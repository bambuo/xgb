package xgb

import (
	"math"
	"sort"
)

// ── WQ Sketech (Weighted Quantile Sketch) ───────────────────
//
// 精确复刻 XGBoost 3.2 的 WQSummary（src/common/quantile.h）。
// 用于计算直方图分箱的 bin 边界：对每个特征用 Hessian 作为权重
// 构建加权分位数 sketch，然后选择 max_bins 个等间距分位数作为 bin 边界。
//
// 与 XGBoost 一致，内部使用 float32 精度比较值。

// WQEntry 是 WQSummary 中的单个摘要条目。
// 对应 XGBoost WQSummary::Entry。
type WQEntry struct {
	RMin  float64 // 最小排名（累积权重在之前）
	RMax  float64 // 最大排名（累积权重包含自身）
	WMin  float64 // 权重（Hessian）
	Value float32 // 特征值（float32 与 XGBoost 对齐）
}

// RMinNext 返回严格大于当前值时的最小排名估计。
// 对应 XGBoost 的 RMinNext() = rmin + wmin。
func (e *WQEntry) RMinNext() float64 {
	return e.RMin + e.WMin
}

// RMaxPrev 返回严格小于当前值时的最大排名估计。
// 对应 XGBoost 的 RMaxPrev() = rmax - wmin。
func (e *WQEntry) RMaxPrev() float64 {
	return e.RMax - e.WMin
}

// WQSummary 对应 XGBoost 的 WQSummary<DType, RType>。
type WQSummary struct {
	Data []WQEntry
	Size int
}

// NewWQSummary 创建一个新的 WQSummary。
func NewWQSummary(capacity int) *WQSummary {
	return &WQSummary{
		Data: make([]WQEntry, capacity),
	}
}

// CopyFrom 从另一个 summary 复制内容。
func (s *WQSummary) CopyFrom(src *WQSummary) {
	s.Size = src.Size
	copy(s.Data[:s.Size], src.Data[:src.Size])
}

// MaxRank 返回摘要中的最大排名。
func (s *WQSummary) MaxRank() float64 {
	if s.Size == 0 {
		return 0
	}
	return s.Data[s.Size-1].RMax
}

// MaxError 返回摘要的最大误差。
func (s *WQSummary) MaxError() float64 {
	res := s.Data[0].RMax - s.Data[0].RMin - s.Data[0].WMin
	for i := 1; i < s.Size; i++ {
		err := s.Data[i].RMaxPrev() - s.Data[i-1].RMinNext()
		if err > res {
			res = err
		}
		err = s.Data[i].RMax - s.Data[i].RMin - s.Data[i].WMin
		if err > res {
			res = err
		}
	}
	return res
}

// SetPrune 将当前 summary 设置为 src 的剪枝版本。
// 对应 XGBoost 的 WQSummary::SetPrune。
// 从 src 中选择最多 maxsize 个等间距条目。
func (s *WQSummary) SetPrune(src *WQSummary, maxsize int) {
	if src.Size <= maxsize {
		s.CopyFrom(src)
		return
	}

	begin := src.Data[0].RMin
	lastIdx := src.Size - 1
	range_ := src.Data[lastIdx].RMin - begin
	n := maxsize - 1

	// 总是保留第一个条目
	s.Data[0] = src.Data[0]
	s.Size = 1

	i := 1
	lastidx := 0
	for k := 1; k < n; k++ {
		// target = 2 * (k * range / n + begin)
		dx2 := 2.0 * (float64(k)*range_/float64(n) + begin)

		// 找到第一个 i，使得 dx2 < rmax[i+1] + rmin[i+1]
		for i < lastIdx && dx2 >= src.Data[i+1].RMax+src.Data[i+1].RMin {
			i++
		}
		if i == lastIdx {
			break
		}
		if dx2 < src.Data[i].RMinNext()+src.Data[i+1].RMaxPrev() {
			if i != lastidx {
				s.Data[s.Size] = src.Data[i]
				s.Size++
				lastidx = i
			}
		} else {
			if i+1 != lastidx {
				s.Data[s.Size] = src.Data[i+1]
				s.Size++
				lastidx = i + 1
			}
		}
	}
	// 总是保留最后一个条目
	if lastidx != lastIdx {
		s.Data[s.Size] = src.Data[lastIdx]
		s.Size++
	}
}

// SetCombine 将当前 summary 设置为 sa 和 sb 的合并。
// 对应 XGBoost 的 WQSummary::SetCombine。
func (s *WQSummary) SetCombine(sa, sb *WQSummary) {
	if sa.Size == 0 {
		s.CopyFrom(sb)
		return
	}
	if sb.Size == 0 {
		s.CopyFrom(sa)
		return
	}

	a := 0
	b := 0
	dst := 0
	var aprevRMin, bprevRMin float64

	for a < sa.Size && b < sb.Size {
		if sa.Data[a].Value == sb.Data[b].Value {
			s.Data[dst] = WQEntry{
				RMin:  sa.Data[a].RMin + sb.Data[b].RMin,
				RMax:  sa.Data[a].RMax + sb.Data[b].RMax,
				WMin:  sa.Data[a].WMin + sb.Data[b].WMin,
				Value: sa.Data[a].Value,
			}
			aprevRMin = sa.Data[a].RMinNext()
			bprevRMin = sb.Data[b].RMinNext()
			dst++
			a++
			b++
		} else if sa.Data[a].Value < sb.Data[b].Value {
			s.Data[dst] = WQEntry{
				RMin:  sa.Data[a].RMin + bprevRMin,
				RMax:  sa.Data[a].RMax + sb.Data[b].RMaxPrev(),
				WMin:  sa.Data[a].WMin,
				Value: sa.Data[a].Value,
			}
			aprevRMin = sa.Data[a].RMinNext()
			dst++
			a++
		} else {
			s.Data[dst] = WQEntry{
				RMin:  sb.Data[b].RMin + aprevRMin,
				RMax:  sb.Data[b].RMax + sa.Data[a].RMaxPrev(),
				WMin:  sb.Data[b].WMin,
				Value: sb.Data[b].Value,
			}
			bprevRMin = sb.Data[b].RMinNext()
			dst++
			b++
		}
	}

	for a < sa.Size {
		bLastRMax := sb.Data[sb.Size-1].RMax
		s.Data[dst] = WQEntry{
			RMin:  sa.Data[a].RMin + bprevRMin,
			RMax:  sa.Data[a].RMax + bLastRMax,
			WMin:  sa.Data[a].WMin,
			Value: sa.Data[a].Value,
		}
		dst++
		a++
	}

	for b < sb.Size {
		aLastRMax := sa.Data[sa.Size-1].RMax
		s.Data[dst] = WQEntry{
			RMin:  sb.Data[b].RMin + aprevRMin,
			RMax:  sb.Data[b].RMax + aLastRMax,
			WMin:  sb.Data[b].WMin,
			Value: sb.Data[b].Value,
		}
		dst++
		b++
	}

	s.Size = dst

	// 修正可能因浮点误差导致的排序错误
	// XGBoost 中也有同样的修正
	for i := 1; i < s.Size; i++ {
		if s.Data[i].RMin < s.Data[i-1].RMin {
			s.Data[i].RMin = s.Data[i-1].RMin
		}
		if s.Data[i].RMax < s.Data[i-1].RMax {
			s.Data[i].RMax = s.Data[i-1].RMax
		}
	}
}

// BuildWeightedQuantiles 从带权重的值构建加权分位数摘要。
// 这是 XGBoost WQSummary 的核心算法：
// 1. 对 (value, weight) 按值排序
// 2. 合并相同值的权重
// 3. 构建具有累积排名边界的条目
// 返回可直接使用的 WQSummary。
func BuildWeightedQuantiles(values []float32, weights []float64) *WQSummary {
	if len(values) == 0 {
		return NewWQSummary(0)
	}

	// 创建 (value, weight) 对并按值排序
	type pair struct {
		Value float32
		W     float64
	}
	pairs := make([]pair, len(values))
	for i, v := range values {
		pairs[i] = pair{Value: v, W: weights[i]}
	}
	sort.Slice(pairs, func(i, j int) bool {
		return pairs[i].Value < pairs[j].Value
	})

	// 收集去重后的值（相同值权重合并）
	summary := NewWQSummary(len(pairs))
	summary.Size = 0
	wsum := 0.0

	for i := 0; i < len(pairs); {
		j := i + 1
		w := pairs[i].W
		for j < len(pairs) && pairs[j].Value == pairs[i].Value {
			w += pairs[j].W
			j++
		}
		if summary.Size < len(summary.Data) {
			summary.Data[summary.Size] = WQEntry{
				RMin:  wsum,
				RMax:  wsum + w,
				WMin:  w,
				Value: pairs[i].Value,
			}
			summary.Size++
		}
		wsum += w
		i = j
	}

	return summary
}

// ── HistogramCuts：从 WQSummary 生成 bin 边界 ─────────────

// HistogramCuts 保存每个特征的 bin 边界。
// 对应 XGBoost 的 HistogramCuts。
type HistogramCuts struct {
	Ptrs      []int     // 每个特征的 boundaries 起始索引
	Values    []float32 // 所有 bin 边界值（连续存储）
	MinValues []float32 // 每个特征的最小值
	NumBins   int       // 目标箱数
}

// AddCutPoint 将 summary 中的条目添加为 bin 边界。
// 对应 XGBoost 3.2 的 MakeCuts：使用等间距步长选择条目。
// 从 summary.data[1] 开始，每隔 step 个条目选一个。
// 注意：不做跨特征去重，因为 cuts.Values 是全局累积的，而每个特征
// 的边界已由 Ptrs 数组正确分隔。单特征内值严格递增（来自 SetPrune），
// 无需额外去重。
func AddCutPoint(summary *WQSummary, maxBin int, cuts *HistogramCuts) {
	step := summary.Size / maxBin
	if step < 1 {
		step = 1
	}
	for i := 1; i < summary.Size-step+1; i += step {
		cuts.Values = append(cuts.Values, summary.Data[i].Value)
	}
}

// BuildHistogramCuts 从训练数据构建 bin 边界。
// hess 是初始 Hessian（来自目标函数的第一轮梯度）。
// 对应 XGBoost 的 SketchOnDMatrix + MakeCuts。
func BuildHistogramCuts(data [][]float64, hess []float64, maxBins int) *HistogramCuts {
	if len(data) == 0 {
		return &HistogramCuts{}
	}

	nFeatures := len(data[0])
	nSamples := len(data)
	cuts := &HistogramCuts{
		Ptrs:      make([]int, nFeatures+1),
		MinValues: make([]float32, nFeatures),
		NumBins:   maxBins,
	}

	for f := 0; f < nFeatures; f++ {
		minVal := float32(math.MaxFloat32)
		hasValue := false

		// 收集特征值和权重
		var fvals []float32
		var fweights []float64
		for i := 0; i < nSamples; i++ {
			v := data[i][f]
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
				fvals = append(fvals, v32)
				fweights = append(fweights, w)
			}
		}

		if !hasValue {
			cuts.Ptrs[f+1] = len(cuts.Values)
			cuts.MinValues[f] = 0
			continue
		}

		cuts.MinValues[f] = minVal

		// 构建加权分位数摘要
		summary := BuildWeightedQuantiles(fvals, fweights)

		if summary.Size == 0 {
			cuts.Ptrs[f+1] = len(cuts.Values)
			continue
		}

		// 使用 SetPrune 选择最多 maxBins+1 个等间距条目
		pruned := NewWQSummary(maxBins + 1)
		pruned.SetPrune(summary, maxBins+1)

		// 添加 cut points: entries 1..maxBins-1
		AddCutPoint(pruned, maxBins, cuts)

		// 添加 sentinel（对应 XGBoost 的 cpt + |cpt| + 1e-5）
		var cpt float32
		if pruned.Size > 0 {
			cpt = pruned.Data[pruned.Size-1].Value
		} else {
			cpt = cuts.MinValues[f]
		}
		sentinel := cpt + float32(math.Abs(float64(cpt))) + 1e-5
		// 确保 sentinel 严格大于最后一个 cut value
		if len(cuts.Values) == 0 || sentinel > cuts.Values[len(cuts.Values)-1] {
			cuts.Values = append(cuts.Values, sentinel)
		} else {
			// 如果 sentinel 不够大，加倍直到超过
			for sentinel <= cuts.Values[len(cuts.Values)-1] {
				sentinel = sentinel*2 + 1e-5
			}
			cuts.Values = append(cuts.Values, sentinel)
		}

		cuts.Ptrs[f+1] = len(cuts.Values)
	}

	return cuts
}

// BinForValue 返回给定特征值对应的 bin 索引。
// 对应 XGBoost 的 SearchBin：使用 upper_bound 查找，并夹紧到 [0, nBins-1]。
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
	// 夹紧，对应 XGBoost 的 idx -= !!(idx == end)
	if idx >= len(boundaries) {
		idx = len(boundaries) - 1
	}
	return idx
}

// NumBinsForFeature 返回某个特征的箱数（= 边界数，含 sentinel）。
func (c *HistogramCuts) NumBinsForFeature(feature int) int {
	return c.Ptrs[feature+1] - c.Ptrs[feature]
}

// ── GK Sketch（旧实现，保留用于参考）─────────────────────────

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
func (s *GKSketch) Push(value float32, weight float64) {
	if weight <= 0 {
		return
	}

	pos := sort.Search(len(s.Entries), func(i int) bool {
		return s.Entries[i].Value > value
	})

	entry := GKEntry{
		Value: value,
		W:     weight,
	}

	s.Entries = append(s.Entries, GKEntry{})
	copy(s.Entries[pos+1:], s.Entries[pos:])
	s.Entries[pos] = entry

	var cum float64
	for i := range s.Entries {
		s.Entries[i].RMin = cum
		cum += s.Entries[i].W
		s.Entries[i].RMax = cum
	}

	maxSize := int(1.0/s.Epsilon) + 1
	if len(s.Entries) > maxSize {
		s.Compress()
	}
}

// Compress 合并 sketch 中的条目以减少大小。
func (s *GKSketch) Compress() {
	if len(s.Entries) < 3 {
		return
	}

	threshold := 2.0 * s.Epsilon * s.TotalWeight()

	i := len(s.Entries) - 2
	for i > 0 {
		next := &s.Entries[i+1]
		curr := &s.Entries[i]

		if curr.RMax+next.RMin-curr.W-next.W <= threshold || curr.Value == next.Value {
			next.W += curr.W
			next.RMin = curr.RMin
			if next.RMax < curr.RMax {
				next.RMax = curr.RMax
			}
			s.Entries = append(s.Entries[:i], s.Entries[i+1:]...)
		}
		i--
	}

	var cum float64
	for i := range s.Entries {
		s.Entries[i].RMin = cum
		cum += s.Entries[i].W
		s.Entries[i].RMax = cum
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

// QueryAll 使用旧 GK Sketch 从量化位置生成边界值。
// 保留此函数以保持向后兼容性。
func (s *GKSketch) QueryAll(maxBins int) []float32 {
	if len(s.Entries) == 0 {
		return nil
	}

	boundaries := make([]float32, 0, maxBins)
	for i := 1; i < maxBins; i++ {
		q := float64(i) / float64(maxBins+1)
		v := s.Query(q)
		if len(boundaries) == 0 || boundaries[len(boundaries)-1] != v {
			boundaries = append(boundaries, v)
		}
	}
	lastEntry := s.Query(float64(maxBins) / float64(maxBins+1))
	sentinel := lastEntry + float32(math.Abs(float64(lastEntry))) + 1e-5
	if len(boundaries) == 0 || boundaries[len(boundaries)-1] != sentinel {
		boundaries = append(boundaries, sentinel)
	}
	return boundaries
}
