package xgb

import (
	"math"
	"sort"
)

// ── BinMapper ───────────────────────────────────────────────

// BinMapper 将连续特征值映射到离散箱。
type BinMapper struct {
	NumBins     int
	Boundaries  [][]float64
	numFeatures int
}

// NewBinMapper 从训练数据创建箱映射器。
func NewBinMapper(data [][]float64, numBins int) *BinMapper {
	if numBins < 2 {
		numBins = 2
	}
	if numBins > 2048 {
		numBins = 2048
	}
	nFeatures := len(data[0])
	bm := &BinMapper{NumBins: numBins, numFeatures: nFeatures,
		Boundaries: make([][]float64, nFeatures)}
	for f := 0; f < nFeatures; f++ {
		vals := make([]float64, 0, len(data))
		for _, row := range data {
			if v := row[f]; !math.IsNaN(v) {
				vals = append(vals, v)
			}
		}
		if len(vals) < 2 {
			continue
		}
		sort.Float64s(vals)
		n := len(vals)
		bins := numBins
		if bins > n {
			bins = n
		}
		bm.Boundaries[f] = make([]float64, 0, bins-1)
		for i := 1; i < bins; i++ {
			idx := i * n / bins
			if idx >= n {
				idx = n - 1
			}
			if idx > 0 && vals[idx] == vals[idx-1] {
				continue
			}
			bm.Boundaries[f] = append(bm.Boundaries[f], vals[idx])
		}
	}
	return bm
}

// Bin 返回特征值的箱索引（二分查找）。
func (bm *BinMapper) Bin(feature int, value float64) int {
	if math.IsNaN(value) {
		return 0
	}
	bds := bm.Boundaries[feature]
	return sort.Search(len(bds), func(i int) bool { return bds[i] > value })
}

// Threshold 返回箱边界的实际分裂值。
func (bm *BinMapper) Threshold(feature, boundaryIdx int) float64 {
	bds := bm.Boundaries[feature]
	if boundaryIdx < 0 {
		return bds[0]
	}
	if boundaryIdx >= len(bds) {
		return bds[len(bds)-1]
	}
	return bds[boundaryIdx]
}

// NumBinsForFeature 返回某个特征的箱数。
func (bm *BinMapper) NumBinsForFeature(feature int) int {
	return len(bm.Boundaries[feature]) + 1
}

// NewGKBinMapper 使用 Greenwald-Khanna 加权分位数 sketch 创建箱映射器。
// 这与 XGBoost 的 HistogramCuts（SketchOnDMatrix）使用相同的算法。
// hess 用于加权分位数计算，对应 XGBoost 中“用 Hessian 加权”的逻辑。
func NewGKBinMapper(data [][]float64, hess []float64, maxBins int) *BinMapper {
	if maxBins < 2 {
		maxBins = 2
	}
	if maxBins > 2048 {
		maxBins = 2048
	}

	cuts := BuildHistogramCuts(data, hess, maxBins)

	nFeatures := len(data[0])
	bm := &BinMapper{
		NumBins:     maxBins,
		numFeatures: nFeatures,
		Boundaries:  make([][]float64, nFeatures),
	}

	for f := 0; f < nFeatures; f++ {
		start := cuts.Ptrs[f]
		end := cuts.Ptrs[f+1]
		bds := cuts.Values[start:end]
		// 将 float32 边界转换为 float64
		bm.Boundaries[f] = make([]float64, len(bds))
		for i, b := range bds {
			bm.Boundaries[f][i] = float64(b)
		}
	}

	return bm
}

// ── 分裂结果 ────────────────────────────────────────────

type splitResult struct {
	feature     int
	bin         int
	gain        float64
	sumGradLeft float64
	sumHessLeft float64
	defaultLeft bool
}

// ── HistBuilder ─────────────────────────────────────────────

// HistBuilder 使用基于直方图的分裂查找构建树。
type HistBuilder struct {
	tree       *RegTree
	config     *ExactBuilderConfig
	mapper     *BinMapper
	binIndices [][]int
	dm         *DMatrix // 引用原始数据，用于 NaN 检测
	iterSeed   int64    // 当前轮次的随机种子
}

// NewHistBuilder 创建一个新的 HistBuilder。
func NewHistBuilder(cfg *ExactBuilderConfig, mapper *BinMapper) *HistBuilder {
	return &HistBuilder{config: cfg, mapper: mapper}
}

// SetIterSeed 设置当前轮次的随机种子，用于 colsample_bylevel/bynode。
func (b *HistBuilder) SetIterSeed(seed int64) {
	b.iterSeed = seed
}

// precomputeBins 预先计算每个（样本，特征）对的箱索引。
func (b *HistBuilder) precomputeBins(dm *DMatrix) {
	b.dm = dm // 保存引用供 NaN 检测使用
	n, nF := dm.NumRows, dm.NumCols
	b.binIndices = make([][]int, n)
	for i := 0; i < n; i++ {
		row := make([]int, nF)
		for f := 0; f < nF; f++ {
			row[f] = b.mapper.Bin(f, dm.Data[i][f])
		}
		b.binIndices[i] = row
	}
}

// Build 使用基于直方图的分裂查找构建单棵树。
func (b *HistBuilder) Build(dm *DMatrix, grads, hess []float64,
	sampleIndices []int, featureMask []int) error {
	if dm.NumRows == 0 || dm.NumCols == 0 {
		return nil
	}
	if b.mapper == nil {
		b.mapper = NewBinMapper(dm.Data, 256)
	}
	b.precomputeBins(dm)

	b.tree = NewRegTree(dm.NumCols)
	b.tree.Param.MaxDepth = b.config.MaxDepth

	if sampleIndices == nil {
		sampleIndices = make([]int, dm.NumRows)
		for i := range sampleIndices {
			sampleIndices[i] = i
		}
	}

	totalGrad, totalHess := sumGH(grads, hess, sampleIndices)
	rootIdx := b.tree.InitRoot(totalGrad, totalHess)
	b.tree.SetLeaf(rootIdx, calcLeafWeight(totalGrad, totalHess, b.config.Lambda, b.config.Alpha))

	b.splitNode(rootIdx, 0, dm, grads, hess, sampleIndices, featureMask)
	return nil
}

// splitNode 使用直方图扫描递归分裂，并行处理特征。
func (b *HistBuilder) splitNode(nodeIdx, depth int, dm *DMatrix,
	grads, hess []float64, indices []int, featureMask []int) {

	if depth >= b.config.MaxDepth || len(indices) < 2 {
		return
	}

	nodeGrad, nodeHess := b.tree.Nodes[nodeIdx].SumGrad, b.tree.Nodes[nodeIdx].SumHess

	// 应用 colsample_bylevel：每层采样一次特征
	activeFeatures := featureMask
	if b.config.ColSampleByLevel > 0 && b.config.ColSampleByLevel < 1.0 {
		levelSeed := uint32(b.iterSeed) + uint32(depth)*1000
		activeFeatures = mtColSampleFromMask(activeFeatures, b.config.ColSampleByLevel, levelSeed)
	}

	// 应用 colsample_bynode：每节点采样一次特征
	if b.config.ColSampleByNode > 0 && b.config.ColSampleByNode < 1.0 {
		nodeSeed := uint32(b.iterSeed) + uint32(depth)*1000 + uint32(nodeIdx)
		activeFeatures = mtColSampleFromMask(activeFeatures, b.config.ColSampleByNode, nodeSeed)
	}

	// 收集活跃特征（未被屏蔽且有箱）
	type featJob struct{ idx, nBins int }
	var jobs []featJob
	for f := 0; f < dm.NumCols; f++ {
		if activeFeatures != nil && !contains(activeFeatures, f) {
			continue
		}
		if nb := b.mapper.NumBinsForFeature(f); nb > 1 {
			jobs = append(jobs, featJob{f, nb})
		}
	}
	if len(jobs) == 0 {
		return
	}

	// 顺序扫描每个特征
	var best *splitResult
	for _, j := range jobs {
		r := b.findBestSplitForFeature(j.idx, j.nBins, nodeGrad, nodeHess, indices, grads, hess)
		if r == nil {
			continue
		}
		if best == nil || r.gain > best.gain {
			best = r
		}
	}
	if best == nil || best.gain <= 0 {
		return
	}

	// 分裂索引，尊重 defaultLeft 方向
	var leftIdx, rightIdx []int
	threshold := b.mapper.Threshold(best.feature, best.bin)
	for _, idx := range indices {
		v := dm.Data[idx][best.feature]
		if math.IsNaN(v) {
			if best.defaultLeft {
				leftIdx = append(leftIdx, idx)
			} else {
				rightIdx = append(rightIdx, idx)
			}
		} else if v <= threshold {
			leftIdx = append(leftIdx, idx)
		} else {
			rightIdx = append(rightIdx, idx)
		}
	}
	if len(leftIdx) == 0 || len(rightIdx) == 0 {
		return
	}

	// 创建子节点并应用分裂
	leftChild := b.tree.AddNode()
	rightChild := b.tree.AddNode()
	b.tree.SetSplit(nodeIdx, best.feature, threshold, best.gain, leftChild, rightChild, best.defaultLeft)

	sumGradRight := nodeGrad - best.sumGradLeft
	sumHessRight := nodeHess - best.sumHessLeft
	b.tree.Nodes[leftChild].SumGrad = best.sumGradLeft
	b.tree.Nodes[leftChild].SumHess = best.sumHessLeft
	b.tree.Nodes[rightChild].SumGrad = sumGradRight
	b.tree.Nodes[rightChild].SumHess = sumHessRight
	b.tree.SetLeaf(leftChild, calcLeafWeight(best.sumGradLeft, best.sumHessLeft, b.config.Lambda, b.config.Alpha))
	b.tree.SetLeaf(rightChild, calcLeafWeight(sumGradRight, sumHessRight, b.config.Lambda, b.config.Alpha))

	b.splitNode(leftChild, depth+1, dm, grads, hess, leftIdx, featureMask)
	b.splitNode(rightChild, depth+1, dm, grads, hess, rightIdx, featureMask)
}

// findBestSplitForFeature 构建直方图并扫描以找到最佳分裂。
// 实现 Sparsity-Aware Split Finding：对每个候选分裂，
// 分别尝试缺失值→左 和 缺失值→右，取增益更大的方向。
func (b *HistBuilder) findBestSplitForFeature(feature, nBins int, nodeGrad, nodeHess float64,
	indices []int, grads, hess []float64) *splitResult {

	histGrad := make([]float64, nBins)
	histHess := make([]float64, nBins)

	// 单独统计缺失值的梯度/Hessian
	missingGrad := 0.0
	missingHess := 0.0
	for _, idx := range indices {
		if b.dm != nil && math.IsNaN(b.dm.Data[idx][feature]) {
			missingGrad += grads[idx]
			missingHess += hess[idx]
		} else {
			bin := b.binIndices[idx][feature]
			histGrad[bin] += grads[idx]
			histHess[bin] += hess[idx]
		}
	}

	bestGain := -1.0
	bestBin := -1
	bestSumGradLeft := 0.0
	bestSumHessLeft := 0.0
	bestDefaultLeft := false
	sumGradLeft := 0.0
	sumHessLeft := 0.0

	// 非缺失值的总梯度/Hessian
	nonMissingGrad := nodeGrad - missingGrad
	nonMissingHess := nodeHess - missingHess

	for bin := 0; bin < nBins-1; bin++ {
		sumGradLeft += histGrad[bin]
		sumHessLeft += histHess[bin]
		if histHess[bin] == 0 {
			continue
		}

		// 非缺失值部分的右侧
		sumGradRight := nonMissingGrad - sumGradLeft
		sumHessRight := nonMissingHess - sumHessLeft

		// 尝试缺失值→左
		gainLeft := 0.0
		{
			gL := sumGradLeft + missingGrad
			hL := sumHessLeft + missingHess
			gR := sumGradRight
			hR := sumHessRight
			if hL >= b.config.MinChildWeight && hR >= b.config.MinChildWeight {
				gainLeft = calcHistGain(gL, hL, gR, hR, nodeGrad, nodeHess, b.config.Lambda, b.config.Alpha, b.config.Gamma)
			}
		}

		// 尝试缺失值→右
		gainRight := 0.0
		{
			gL := sumGradLeft
			hL := sumHessLeft
			gR := sumGradRight + missingGrad
			hR := sumHessRight + missingHess
			if hL >= b.config.MinChildWeight && hR >= b.config.MinChildWeight {
				gainRight = calcHistGain(gL, hL, gR, hR, nodeGrad, nodeHess, b.config.Lambda, b.config.Alpha, b.config.Gamma)
			}
		}

		gain := gainLeft
		defLeft := true
		if gainRight > gainLeft {
			gain = gainRight
			defLeft = false
		}

		if gain > bestGain {
			bestGain = gain
			bestBin = bin
			bestSumGradLeft = sumGradLeft + func() float64 {
				if defLeft {
					return missingGrad
				}
				return 0
			}()
			bestSumHessLeft = sumHessLeft + func() float64 {
				if defLeft {
					return missingHess
				}
				return 0
			}()
			bestDefaultLeft = defLeft
		}
	}

	if bestGain <= 0 {
		return nil
	}
	return &splitResult{
		feature: feature, bin: bestBin, gain: bestGain,
		sumGradLeft: bestSumGradLeft, sumHessLeft: bestSumHessLeft,
		defaultLeft: bestDefaultLeft,
	}
}

// calcHistGain 使用直方图风格计算增益。
func calcHistGain(gL, hL, gR, hR, gP, hP, lambda, alpha, gamma float64) float64 {
	leftW := gL / (hL + lambda)
	rightW := gR / (hR + lambda)
	parentW := gP / (hP + lambda)

	gl := leftW * gL
	gr := rightW * gR
	gp := parentW * gP

	if alpha > 0 {
		gl = applyAlpha(gl, leftW, gL, alpha)
		gr = applyAlpha(gr, rightW, gR, alpha)
		gp = applyAlpha(gp, parentW, gP, alpha)
	}

	return 0.5*(gl+gr-gp) - gamma
}

// applyAlpha 使用 L1 正则化计算增益。
func applyAlpha(gain, weight, sumGrad, alpha float64) float64 {
	if sumGrad < -alpha {
		return gain + alpha*weight
	}
	if sumGrad > alpha {
		return gain - alpha*weight
	}
	return 0
}

// contains 检查 s 中是否包含 v。
func contains(s []int, v int) bool {
	for i := range s {
		if s[i] == v {
			return true
		}
	}
	return false
}
