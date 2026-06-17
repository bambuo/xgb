package xgb

import (
	"math"
	"sort"
)

// ── GK 加权分位数 Sketch ─────────────────────────────────────
//
// 精确复刻 XGBoost 3.2 的 WXQSketch（src/common/quantile.h）。
// 用于计算直方图分箱的 bin 边界：对每个特征用 Hessian 作为权重
// 构建加权分位数 sketch，然后查询 max_bins 个等间距分位数作为 bin 边界。
//
// 与 XGBoost 一致，内部使用 float32 精度比较值。

// GKEntry 是 Greenwald-Khanna sketch 中的单个摘要条目。
type GKEntry struct {
	Value float32 // 特征值（float32 与 XGBoost 对齐）
	RMin  float64 // 最小排名
	RMax  float64 // 最大排名
	W     float64 // 权重（Hessian）
}

// GKSketch 是 Greenwald-Khanna 加权分位数 sketch。
type GKSketch struct {
	Entries []GKEntry
	Epsilon float64 // 1.0 / (2 * max_bins)
}

// NewGKSketch 创建新的 GK sketch。
// maxBins 控制精度：epsilon = 1.0 / (2 * maxBins)。
func NewGKSketch(maxBins int) *GKSketch {
	if maxBins < 2 {
		maxBins = 2
	}
	return &GKSketch{
		Epsilon: 1.0 / float64(2*maxBins),
		Entries: make([]GKEntry, 0, 256),
	}
}

// Push 向 sketch 中插入一个带权重的值。
// 对应 XGBoost WXQSketch::Push(value, wmin, wmax)。
func (s *GKSketch) Push(value float32, weight float64) {
	if weight <= 0 {
		return
	}

	entry := GKEntry{
		Value: value,
		RMin:  0,
		RMax:  0,
		W:     weight,
	}

	// 找到插入位置（按值排序）
	pos := sort.Search(len(s.Entries), func(i int) bool {
		return s.Entries[i].Value > value
	})

	// 计算 rmin 和 rmax
	if pos < len(s.Entries) {
		entry.RMin = s.Entries[pos].RMin
		entry.RMax = s.Entries[pos].RMax - weight
		if entry.RMax < entry.RMin {
			entry.RMax = entry.RMin
		}
	} else {
		// 末尾插入
		if len(s.Entries) > 0 {
			last := &s.Entries[len(s.Entries)-1]
			entry.RMin = last.RMin + last.W
			entry.RMax = entry.RMin
		}
	}

	// 插入到 pos 位置
	s.Entries = append(s.Entries, GKEntry{})
	copy(s.Entries[pos+1:], s.Entries[pos:])
	s.Entries[pos] = entry

	// 检查是否需要压缩
	maxSize := int(1.0/s.Epsilon) + 1
	if len(s.Entries) > maxSize {
		s.Compress()
	}
}

// Compress 合并 sketch 中的条目以减少大小。
// 对应 XGBoost WXQSketch::Compact()。
func (s *GKSketch) Compress() {
	if len(s.Entries) < 3 {
		return
	}

	threshold := 2.0 * s.Epsilon * s.TotalWeight()

	// 从右向左扫描，合并满足条件的相邻条目
	i := len(s.Entries) - 2
	for i > 0 {
		next := &s.Entries[i+1]
		curr := &s.Entries[i]

		// 合并条件：curr.RMax + next.RMin - curr.W - next.W <= threshold
		// 且 curr.Value != next.Value
		if curr.RMax+next.RMin-curr.W-next.W <= threshold && curr.Value == next.Value {
			next.W += curr.W
			next.RMin = curr.RMin
			if next.RMax < curr.RMax {
				next.RMax = curr.RMax
			}
			// 删除 curr
			s.Entries = append(s.Entries[:i], s.Entries[i+1:]...)
		}
		i--
	}
}

// TotalWeight 返回 sketch 中所有权重之和。
func (s *GKSketch) TotalWeight() float64 {
	var total float64
	for _, e := range s.Entries {
		total += e.W
	}
	return total
}

// Query 查询指定分位数 q 对应的值。
// q 的范围是 [0, 1]。
// 对应 XGBoost WXQSketch::Query(quantile)。
func (s *GKSketch) Query(q float64) float32 {
	if len(s.Entries) == 0 {
		return 0
	}
	if len(s.Entries) == 1 {
		return s.Entries[0].Value
	}

	target := q * s.TotalWeight()
	bestRank := math.MaxFloat64
	bestValue := s.Entries[0].Value

	for _, e := range s.Entries {
		rank := e.RMin + 0.5*e.W
		diff := math.Abs(rank - target)
		if diff < bestRank {
			bestRank = diff
			bestValue = e.Value
		}
	}
	return bestValue
}

// QueryAll 返回 maxBins 个等间距分位数的值（去重后）。
// 用于生成 bin 边界。
func (s *GKSketch) QueryAll(maxBins int) []float32 {
	if len(s.Entries) == 0 {
		return nil
	}

	boundaries := make([]float32, 0, maxBins)
	for i := 1; i <= maxBins; i++ {
		q := float64(i) / float64(maxBins+1)
		v := s.Query(q)
		// 去重
		if len(boundaries) == 0 || boundaries[len(boundaries)-1] != v {
			boundaries = append(boundaries, v)
		}
	}
	return boundaries
}

// ── HistogramCuts：从 sketch 生成 bin 边界 ─────────────────

// HistogramCuts 保存每个特征的 bin 边界。
// 对应 XGBoost 的 HistogramCuts。
type HistogramCuts struct {
	Ptrs      []int     // 每个特征的 boundaries 起始索引
	Values    []float32 // 所有 bin 边界值（连续存储）
	MinValues []float32 // 每个特征的最小值
	NumBins   int       // 目标箱数
}

// BuildHistogramCuts 从训练数据构建 bin 边界。
// hess 是初始 Hessian（来自目标函数的第一轮梯度）。
// 对应 XGBoost 的 SketchOnDMatrix。
func BuildHistogramCuts(data [][]float64, hess []float64, maxBins int) *HistogramCuts {
	if len(data) == 0 {
		return &HistogramCuts{}
	}

	nFeatures := len(data[0])
	cuts := &HistogramCuts{
		Ptrs:      make([]int, nFeatures+1),
		MinValues: make([]float32, nFeatures),
		NumBins:   maxBins,
	}

	for f := 0; f < nFeatures; f++ {
		sketch := NewGKSketch(maxBins)
		minVal := float32(math.MaxFloat32)
		hasValue := false

		for i, row := range data {
			v := row[f]
			if math.IsNaN(v) {
				continue
			}
			v32 := float32(v)
			if v32 < minVal {
				minVal = v32
			}
			hasValue = true

			// 使用 Hessian 作为权重
			w := 1.0
			if hess != nil && i < len(hess) {
				w = hess[i]
			}
			if w > 0 {
				sketch.Push(v32, w)
			}
		}

		if !hasValue {
			cuts.Ptrs[f+1] = len(cuts.Values)
			cuts.MinValues[f] = 0
			continue
		}

		cuts.MinValues[f] = minVal
		boundaries := sketch.QueryAll(maxBins)

		// 将边界值添加到 Values
		for _, b := range boundaries {
			cuts.Values = append(cuts.Values, b)
		}
		cuts.Ptrs[f+1] = len(cuts.Values)
	}

	return cuts
}

// BinForValue 返回给定特征值对应的 bin 索引。
func (c *HistogramCuts) BinForValue(feature int, value float64) int {
	if math.IsNaN(value) {
		return 0 // NaN 总是映射到 bin 0
	}
	start := c.Ptrs[feature]
	end := c.Ptrs[feature+1]
	boundaries := c.Values[start:end]

	v32 := float32(value)
	idx := sort.Search(len(boundaries), func(i int) bool {
		return boundaries[i] > v32
	})
	return idx
}

// NumBinsForFeature 返回某个特征的箱数。
func (c *HistogramCuts) NumBinsForFeature(feature int) int {
	return c.Ptrs[feature+1] - c.Ptrs[feature] + 1
}
