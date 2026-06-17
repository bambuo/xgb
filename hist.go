package xgb

import (
	"container/heap"
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

// Bin 返回特征值的箱索引。
// 使用 upper_bound 语义：找到第一个 boundary > value 的位置作为 bin 索引。
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

// NumBinsForFeature 返回某个特征的箱数（= 边界数，含 sentinel）。
func (bm *BinMapper) NumBinsForFeature(feature int) int {
	return len(bm.Boundaries[feature])
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
		// bds 已包含 XGBoost 风格的哨兵（最后一个值），无需额外添加
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

// nodeLevel 保存当前深度需要处理的节点信息。
type nodeLevel struct {
	nodeIdx   int
	indices   []int
	pathGroup int // 当前路径上使用的交互约束组（-1 = 未使用）
}

// nodeSplit 保存 lossguide 增长策略的候选分裂。
type nodeSplit struct {
	gain    float64
	depth   int
	parent  int
	indices []int
	group   int
}

// Build 使用基于直方图的分裂查找构建单棵树。
// 支持 depthwise（level-wise）和 lossguide（max_leaves）两种增长策略。
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
	b.tree.SetLeaf(rootIdx, calcLeafWeight(totalGrad, totalHess, b.config.Lambda, b.config.Alpha, b.config.MaxDeltaStep))

	// 计算交互约束
	featureGroup := make([]int, dm.NumCols)
	for f := range featureGroup {
		featureGroup[f] = -1
	}
	for gidx, group := range b.config.InteractionConstraints {
		for _, f := range group {
			if f >= 0 && f < dm.NumCols {
				featureGroup[f] = gidx
			}
		}
	}

	maxLeaves := b.config.MaxLeaves
	if maxLeaves > 0 {
		// lossguide 策略：基于优先队列的叶节点分裂
		return b.buildLossguide(dm, grads, hess, sampleIndices, featureMask, featureGroup, maxLeaves)
	}

	// depthwise 策略（默认）：层级遍历
	currentLevel := []nodeLevel{{nodeIdx: rootIdx, indices: sampleIndices, pathGroup: -1}}

	for depth := 0; depth < b.config.MaxDepth; depth++ {
		if len(currentLevel) == 0 {
			break
		}

		levelFeatures := featureMask
		if b.config.ColSampleByLevel > 0 && b.config.ColSampleByLevel < 1.0 {
			levelSeed := uint32(b.iterSeed) + uint32(depth)*1000
			levelFeatures = mtColSampleFromMask(levelFeatures, b.config.ColSampleByLevel, levelSeed)
		}

		var nextLevel []nodeLevel

		for _, nl := range currentLevel {
			if len(nl.indices) < 2 {
				continue
			}

			nodeGrad := b.tree.Nodes[nl.nodeIdx].SumGrad
			nodeHess := b.tree.Nodes[nl.nodeIdx].SumHess

			// colsample_bynode：每节点采样一次特征
			activeFeatures := levelFeatures
			if b.config.ColSampleByNode > 0 && b.config.ColSampleByNode < 1.0 {
				nodeSeed := uint32(b.iterSeed) + uint32(depth)*1000 + uint32(nl.nodeIdx)
				activeFeatures = mtColSampleFromMask(activeFeatures, b.config.ColSampleByNode, nodeSeed)
			}

			type featJob struct{ idx, nBins int }
			var jobs []featJob
			for f := 0; f < dm.NumCols; f++ {
				if activeFeatures != nil && !contains(activeFeatures, f) {
					continue
				}
				if nb := b.mapper.NumBinsForFeature(f); nb > 1 {
					if nl.pathGroup >= 0 && featureGroup[f] >= 0 && featureGroup[f] != nl.pathGroup {
						continue
					}
					jobs = append(jobs, featJob{f, nb})
				}
			}
			if len(jobs) == 0 {
				continue
			}

			var best *splitResult
			for _, j := range jobs {
				r := b.findBestSplitForFeature(j.idx, j.nBins, nodeGrad, nodeHess, nl.indices, grads, hess)
				if r == nil {
					continue
				}
				if best == nil || r.gain > best.gain {
					best = r
				}
			}
			if best == nil || best.gain <= b.config.Gamma {
				continue
			}

			var leftIdx, rightIdx []int
			for _, idx := range nl.indices {
				v := dm.Data[idx][best.feature]
				if math.IsNaN(v) {
					if best.defaultLeft {
						leftIdx = append(leftIdx, idx)
					} else {
						rightIdx = append(rightIdx, idx)
					}
				} else {
					bin := b.binIndices[idx][best.feature]
					if bin <= best.bin {
						leftIdx = append(leftIdx, idx)
					} else {
						rightIdx = append(rightIdx, idx)
					}
				}
			}
			if len(leftIdx) == 0 || len(rightIdx) == 0 {
				continue
			}

			leftChild := b.tree.AddNode()
			rightChild := b.tree.AddNode()

			threshold := b.mapper.Threshold(best.feature, best.bin)
			b.tree.SetSplit(nl.nodeIdx, best.feature, threshold, best.gain,
				leftChild, rightChild, best.defaultLeft)

			leftGradSum, leftHessSum := sumGH(grads, hess, leftIdx)
			rightGradSum, rightHessSum := sumGH(grads, hess, rightIdx)
			b.tree.Nodes[leftChild].SumGrad = leftGradSum
			b.tree.Nodes[leftChild].SumHess = leftHessSum
			b.tree.Nodes[rightChild].SumGrad = rightGradSum
			b.tree.Nodes[rightChild].SumHess = rightHessSum
			b.tree.SetLeaf(leftChild, calcLeafWeight(leftGradSum, leftHessSum,
				b.config.Lambda, b.config.Alpha, b.config.MaxDeltaStep))
			b.tree.SetLeaf(rightChild, calcLeafWeight(rightGradSum, rightHessSum,
				b.config.Lambda, b.config.Alpha, b.config.MaxDeltaStep))

			childGroup := nl.pathGroup
			if fg := featureGroup[best.feature]; fg >= 0 {
				childGroup = fg
			}
			nextLevel = append(nextLevel, nodeLevel{nodeIdx: leftChild, indices: leftIdx, pathGroup: childGroup})
			nextLevel = append(nextLevel, nodeLevel{nodeIdx: rightChild, indices: rightIdx, pathGroup: childGroup})
		}

		currentLevel = nextLevel
	}

	return nil
}

// leafHeapItem 是损失引导优先队列的节点项。
type leafHeapItem struct {
	gain      float64
	nodeIdx   int
	depth     int
	indices   []int
	group     int
	bestSplit *splitResult
	index     int // heap 索引
}

// leafHeap 实现 heap.Interface 用于 lossguide 增长策略。
type leafHeap []*leafHeapItem

func (h leafHeap) Len() int           { return len(h) }
func (h leafHeap) Less(i, j int) bool { return h[i].gain > h[j].gain } // 最大堆
func (h leafHeap) Swap(i, j int) {
	h[i], h[j] = h[j], h[i]
	h[i].index = i
	h[j].index = j
}
func (h *leafHeap) Push(x interface{}) {
	n := len(*h)
	item := x.(*leafHeapItem)
	item.index = n
	*h = append(*h, item)
}
func (h *leafHeap) Pop() interface{} {
	old := *h
	n := len(old)
	item := old[n-1]
	old[n-1] = nil
	item.index = -1
	*h = old[0 : n-1]
	return item
}

// buildLossguide 使用 lossguide（损失引导）策略构建树。
// 使用 container/heap 维护优先队列，始终分裂增益最高的叶节点。
// 与 XGBoost C++ 的 lossguide 实现一致。
func (b *HistBuilder) buildLossguide(dm *DMatrix, grads, hess []float64,
	sampleIndices []int, featureMask []int, featureGroup []int, maxLeaves int) error {

	h := &leafHeap{}
	heap.Init(h)

	// 根节点入队
	rootItem := &leafHeapItem{
		nodeIdx: 0, depth: 0,
		indices: sampleIndices, group: -1,
		gain: -1, bestSplit: nil,
	}
	heap.Push(h, rootItem)

	for h.Len() > 0 && h.Len() < maxLeaves {
		leaf := heap.Pop(h).(*leafHeapItem)

		// 如果尚未计算最佳分裂，则计算
		if leaf.bestSplit == nil {
			nodeGrad := b.tree.Nodes[leaf.nodeIdx].SumGrad
			nodeHess := b.tree.Nodes[leaf.nodeIdx].SumHess
			leaf.bestSplit = b.findBestNodeSplit(dm, grads, hess, leaf.indices,
				featureMask, featureGroup, leaf.group, nodeGrad, nodeHess)
			if leaf.bestSplit == nil || leaf.bestSplit.gain <= b.config.Gamma {
				continue // 无法分裂，丢弃该叶节点
			}
			leaf.gain = leaf.bestSplit.gain
			// 重新入队
			heap.Push(h, leaf)
			continue
		}

		best := leaf.bestSplit

		// 划分样本
		var leftIdx, rightIdx []int
		for _, idx := range leaf.indices {
			v := dm.Data[idx][best.feature]
			if math.IsNaN(v) {
				if best.defaultLeft {
					leftIdx = append(leftIdx, idx)
				} else {
					rightIdx = append(rightIdx, idx)
				}
			} else {
				bin := b.binIndices[idx][best.feature]
				if bin <= best.bin {
					leftIdx = append(leftIdx, idx)
				} else {
					rightIdx = append(rightIdx, idx)
				}
			}
		}
		if len(leftIdx) == 0 || len(rightIdx) == 0 {
			// 无法划分，丢弃该叶节点（已在 heap.Pop 时移除）
			continue
		}

		// 创建子节点
		leftChild := b.tree.AddNode()
		rightChild := b.tree.AddNode()

		threshold := b.mapper.Threshold(best.feature, best.bin)
		b.tree.SetSplit(leaf.nodeIdx, best.feature, threshold, best.gain,
			leftChild, rightChild, best.defaultLeft)

		leftGradSum, leftHessSum := sumGH(grads, hess, leftIdx)
		rightGradSum, rightHessSum := sumGH(grads, hess, rightIdx)
		b.tree.Nodes[leftChild].SumGrad = leftGradSum
		b.tree.Nodes[leftChild].SumHess = leftHessSum
		b.tree.Nodes[rightChild].SumGrad = rightGradSum
		b.tree.Nodes[rightChild].SumHess = rightHessSum
		b.tree.SetLeaf(leftChild, calcLeafWeight(leftGradSum, leftHessSum,
			b.config.Lambda, b.config.Alpha, b.config.MaxDeltaStep))
		b.tree.SetLeaf(rightChild, calcLeafWeight(rightGradSum, rightHessSum,
			b.config.Lambda, b.config.Alpha, b.config.MaxDeltaStep))

		childGroup := leaf.group
		if fg := featureGroup[best.feature]; fg >= 0 {
			childGroup = fg
		}

		// 将子节点推入优先队列
		pushChild := func(childNodeIdx int, childIndices []int) {
			if len(childIndices) < 2 {
				return
			}
			child := &leafHeapItem{
				nodeIdx:   childNodeIdx,
				depth:     leaf.depth + 1,
				indices:   childIndices,
				group:     childGroup,
				gain:      -1,
				bestSplit: nil,
			}
			// 预计算最佳分裂（懒加载优化）
			if child.depth < b.config.MaxDepth {
				cg := b.tree.Nodes[childNodeIdx].SumGrad
				ch := b.tree.Nodes[childNodeIdx].SumHess
				if s := b.findBestNodeSplit(dm, grads, hess, childIndices,
					featureMask, featureGroup, childGroup, cg, ch); s != nil && s.gain > b.config.Gamma {
					child.bestSplit = s
					child.gain = s.gain
				}
			}
			heap.Push(h, child)
		}

		pushChild(leftChild, leftIdx)
		pushChild(rightChild, rightIdx)
	}

	return nil
}

// findBestNodeSplit 查找节点在所有特征上的最佳分裂。
func (b *HistBuilder) findBestNodeSplit(dm *DMatrix, grads, hess []float64,
	indices []int, featureMask []int, featureGroup []int, pathGroup int,
	nodeGrad, nodeHess float64) *splitResult {

	if len(indices) < 2 {
		return nil
	}

	levelFeatures := featureMask

	type featJob struct{ idx, nBins int }
	var jobs []featJob
	for f := 0; f < dm.NumCols; f++ {
		if levelFeatures != nil && !contains(levelFeatures, f) {
			continue
		}
		if nb := b.mapper.NumBinsForFeature(f); nb > 1 {
			if pathGroup >= 0 && featureGroup[f] >= 0 && featureGroup[f] != pathGroup {
				continue
			}
			jobs = append(jobs, featJob{f, nb})
		}
	}
	if len(jobs) == 0 {
		return nil
	}

	nodeSeed := uint32(b.iterSeed) + uint32(nodeGrad)*1000
	activeFeatures := levelFeatures
	if b.config.ColSampleByNode > 0 && b.config.ColSampleByNode < 1.0 {
		activeFeatures = mtColSampleFromMask(activeFeatures, b.config.ColSampleByNode, nodeSeed)
	}

	// 过滤 activeFeatures
	var filtered []featJob
	for _, j := range jobs {
		if activeFeatures == nil || contains(activeFeatures, j.idx) {
			filtered = append(filtered, j)
		}
	}
	if len(filtered) == 0 {
		return nil
	}

	var best *splitResult
	for _, j := range filtered {
		r := b.findBestSplitForFeature(j.idx, j.nBins, nodeGrad, nodeHess, indices, grads, hess)
		if r == nil {
			continue
		}
		if best == nil || r.gain > best.gain {
			best = r
		}
	}
	if best != nil && best.gain > b.config.Gamma {
		return best
	}
	return nil
}

// findBestSplitForFeature 构建直方图并扫描以找到最佳分裂。
// 实现 Sparsity-Aware Split Finding：对每个候选分裂，
// 分别尝试缺失值→左 和 缺失值→右，取增益更大的方向。
//
// 使用 float64 精度构建直方图和扫描。
// 支持单调约束（MonotoneConstraints）。
func (b *HistBuilder) findBestSplitForFeature(feature, nBins int, nodeGrad, nodeHess float64,
	indices []int, grads, hess []float64) *splitResult {

	// 获取该特征的单调约束（如有）
	constraint := 0
	if b.config.MonotoneConstraints != nil {
		if c, ok := b.config.MonotoneConstraints[feature]; ok {
			constraint = c
		}
	}

	// 使用 float64 精度构建直方图
	histGrad := make([]float64, nBins)
	histHess := make([]float64, nBins)

	// 单独统计缺失值的梯度/Hessian
	var missingGrad, missingHess float64
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

	var sumGradLeft float64
	var sumHessLeft float64

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
				gainLeft = calcHistGain(gL, hL, gR, hR,
					nodeGrad, nodeHess, b.config.Lambda, b.config.Alpha)
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
				gainRight = calcHistGain(gL, hL, gR, hR,
					nodeGrad, nodeHess, b.config.Lambda, b.config.Alpha)
			}
		}

		gain := gainLeft
		defLeft := true
		if gainRight > gainLeft {
			gain = gainRight
			defLeft = false
		}

		// 单调约束检查：仅当该特征有约束时执行
		if constraint != 0 {
			// 选择分裂方向后的左右梯度/Hessian
			gl := sumGradLeft
			hl := sumHessLeft
			gr := nonMissingGrad - sumGradLeft
			hr := nonMissingHess - sumHessLeft
			if defLeft {
				gl += missingGrad
				hl += missingHess
			} else {
				gr += missingGrad
				hr += missingHess
			}
			// 约束：left_weight <= right_weight（constraint=+1）
			// w = -sum_grad / (sum_hess + lambda)
			// 等价于 gl*(hr+λ) >= gr*(hl+λ)
			if float64(constraint)*(gl*(hr+b.config.Lambda)-gr*(hl+b.config.Lambda)) < 0 {
				gain = -1.0 // 违反约束，拒绝
			}
		}

		if gain > bestGain {
			bestGain = gain
			bestBin = bin
			bestSumGradLeft = sumGradLeft
			bestSumHessLeft = sumHessLeft
			bestDefaultLeft = defLeft
		}
	}

	if bestGain <= b.config.Gamma {
		return nil
	}
	return &splitResult{
		feature: feature, bin: bestBin, gain: bestGain,
		sumGradLeft: bestSumGradLeft, sumHessLeft: bestSumHessLeft,
		defaultLeft: bestDefaultLeft,
	}
}

// ThresholdL1 对应 XGBoost 的 ThresholdL1(w, alpha)：
// 如果 w > +alpha 返回 w - alpha；如果 w < -alpha 返回 w + alpha；否则返回 0。
func ThresholdL1(w, alpha float64) float64 {
	if w > alpha {
		return w - alpha
	}
	if w < -alpha {
		return w + alpha
	}
	return 0
}

// calcHistGain 计算分裂增益，对应 XGBoost 的
// loss_chg = CalcGain(GL, HL) + CalcGain(GR, HR) - CalcGain(GP, HP)，
// 其中 CalcGain(g, h) = ThresholdL1(g, α)² / (h + λ)。
func calcHistGain(gL, hL, gR, hR, gP, hP, lambda, alpha float64) float64 {
	if alpha > 0 {
		gL = ThresholdL1(gL, alpha)
		gR = ThresholdL1(gR, alpha)
		gP = ThresholdL1(gP, alpha)
	}

	leftGain := (gL * gL) / (hL + lambda)
	rightGain := (gR * gR) / (hR + lambda)
	parentGain := (gP * gP) / (hP + lambda)

	return leftGain + rightGain - parentGain
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
