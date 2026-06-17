package xgb

import (
	"math"
	"sort"
)

// SplitCandidate 保存候选分裂点的信息。
type SplitCandidate struct {
	FeatureIndex int     // 分裂的特征
	Threshold    float64 // 分裂值
	Gain         float64 // 损失减少量
	SumGradLeft  float64 // ∑g 左
	SumHessLeft  float64 // ∑h 左
	DefaultLeft  bool    // 缺失值的默认方向
}

// BestSplit 表示构建器找到的最佳分裂。
type BestSplit struct {
	FeatureIndex int
	Threshold    float64
	Gain         float64
	DefaultLeft  bool
}

// ExactBuilder 使用精确贪婪算法构建回归树。
//
// 对应 XGBoost C++ 的 ExactUpdater（src/tree/updater_exact.cc）。
type ExactBuilder struct {
	tree   *RegTree
	config *ExactBuilderConfig
}

// ExactBuilderConfig 保存树构建的配置。
type ExactBuilderConfig struct {
	MaxDepth               int
	Gamma                  float64
	Lambda                 float64
	Alpha                  float64
	MinChildWeight         float64
	MaxDeltaStep           float64 // 叶节点权重最大步长裁剪（0 = 无裁剪）
	MaxLeaves              int     // 最大叶节点数（0 = 使用 MaxDepth）
	NumFeatures            int
	MaxBin                 int         // 直方图最大箱数
	ColSampleByLevel       float64     // 每层特征采样比例
	ColSampleByNode        float64     // 每节点特征采样比例
	MaxCachedHistNode      int         // 直方图缓存节点数上限（0 = 不限制）
	MaxCatToOneHot         int         // one-hot 编码的最大类别数（默认 4）
	MaxCatThreshold        int         // 类别特征分裂的最大类别数
	MonotoneConstraints    map[int]int // 单调约束
	InteractionConstraints [][]int     // 交互约束
}

// NewExactBuilder 创建一个新的 ExactBuilder。
func NewExactBuilder(cfg *ExactBuilderConfig) *ExactBuilder {
	return &ExactBuilder{config: cfg}
}

// Build 从给定的梯度和 Hessian 构建单棵树。
//
// 参数：
//   - dm：训练数据矩阵
//   - grads：每个样本的一阶梯度
//   - hess：每个样本的二阶梯度（Hessian）
//   - sampleIndices：使用的样本索引（nil = 全部）
//   - featureMask：考虑的特征索引（nil = 全部）
func (b *ExactBuilder) Build(dm *DMatrix, grads, hess []float64,
	sampleIndices []int, featureMask []int) error {

	if dm.NumRows == 0 || dm.NumCols == 0 {
		return nil
	}

	b.tree = NewRegTree(dm.NumCols)
	b.tree.Param.MaxDepth = b.config.MaxDepth

	// 若未指定子集，使用全部样本
	if sampleIndices == nil {
		sampleIndices = make([]int, dm.NumRows)
		for i := range sampleIndices {
			sampleIndices[i] = i
		}
	}

	// 计算根节点的总梯度/Hessian
	totalGrad, totalHess := sumGH(grads, hess, sampleIndices)

	// 初始化根节点
	rootIdx := b.tree.InitRoot(totalGrad, totalHess)

	// 设置根叶节点值
	rootValue := calcLeafWeight(totalGrad, totalHess, b.config.Lambda, b.config.Alpha, b.config.MaxDeltaStep)
	b.tree.SetLeaf(rootIdx, rootValue)

	// 递归分裂
	b.splitNode(rootIdx, 0, dm, grads, hess, sampleIndices, featureMask)

	return nil
}

// splitNode 递归分裂节点以构建树。
func (b *ExactBuilder) splitNode(nodeIdx, depth int, dm *DMatrix,
	grads, hess []float64, indices []int, featureMask []int) {

	if depth >= b.config.MaxDepth || len(indices) < 2 {
		return
	}

	nodeGrad, nodeHess := b.tree.Nodes[nodeIdx].SumGrad, b.tree.Nodes[nodeIdx].SumHess

	// 在所有特征中寻找最佳分裂
	best := b.FindBestSplit(dm, grads, hess, indices, featureMask, nodeGrad, nodeHess)

	if best.Gain <= b.config.Gamma {
		return // 无有利分裂
	}

	// 创建子节点
	leftIdx := b.tree.AddNode()
	rightIdx := b.tree.AddNode()

	// 在父节点上设置分裂
	b.tree.SetSplit(nodeIdx, best.FeatureIndex, best.Threshold, best.Gain, leftIdx, rightIdx, best.DefaultLeft)

	// 划分样本
	leftIndices, rightIndices := b.partitionSamples(dm, indices, best.FeatureIndex, best.Threshold, best.DefaultLeft)

	// 设置子节点的叶节点权重
	leftGrad, leftHess := sumGH(grads, hess, leftIndices)
	rightGrad, rightHess := sumGH(grads, hess, rightIndices)

	b.tree.Nodes[leftIdx].SumGrad = leftGrad
	b.tree.Nodes[leftIdx].SumHess = leftHess
	b.tree.SetLeaf(leftIdx, calcLeafWeight(leftGrad, leftHess, b.config.Lambda, b.config.Alpha, b.config.MaxDeltaStep))

	b.tree.Nodes[rightIdx].SumGrad = rightGrad
	b.tree.Nodes[rightIdx].SumHess = rightHess
	b.tree.SetLeaf(rightIdx, calcLeafWeight(rightGrad, rightHess, b.config.Lambda, b.config.Alpha, b.config.MaxDeltaStep))

	// 递归
	b.splitNode(leftIdx, depth+1, dm, grads, hess, leftIndices, featureMask)
	b.splitNode(rightIdx, depth+1, dm, grads, hess, rightIndices, featureMask)
}

// FindBestSplit 在所有特征中搜索最佳分裂。
//
// 对应 src/tree/updater_exact.cc 中的 FindBestSplit()。
func (b *ExactBuilder) FindBestSplit(dm *DMatrix, grads, hess []float64,
	indices []int, featureMask []int, nodeGrad, nodeHess float64) *BestSplit {

	best := &BestSplit{Gain: -math.MaxFloat64}

	// 确定要扫描的特征
	features := featureMask
	if features == nil {
		features = make([]int, dm.NumCols)
		for i := range features {
			features[i] = i
		}
	}

	for _, featIdx := range features {
		candidate := b.EnumerateSplit(dm, featIdx, grads, hess, indices, nodeGrad, nodeHess)
		if candidate != nil && candidate.Gain > best.Gain {
			best.Gain = candidate.Gain
			best.FeatureIndex = candidate.FeatureIndex
			best.Threshold = candidate.Threshold
			best.DefaultLeft = candidate.DefaultLeft
		}
	}

	return best
}

// EnumerateSplit 枚举单个特征的候选分裂。
//
// 对应 src/tree/updater_exact.cc 中的 EnumerateSplit()。
// 实现 Sparsity-Aware Split Finding：对每个候选分裂，
// 分别尝试缺失值→左 和 缺失值→右，取增益更大的方向。
func (b *ExactBuilder) EnumerateSplit(dm *DMatrix, featIdx int,
	grads, hess []float64, indices []int, nodeGrad, nodeHess float64) *SplitCandidate {

	if len(indices) < 2 {
		return nil
	}

	// 获取该特征的单调约束
	constraint := 0
	if b.config.MonotoneConstraints != nil {
		if c, ok := b.config.MonotoneConstraints[featIdx]; ok {
			constraint = c
		}
	}

	// 提取给定样本的特征值及其梯度/Hessian
	type entry struct {
		value float64
		grad  float64
		hess  float64
	}

	entries := make([]entry, 0, len(indices))
	missingGrad := 0.0
	missingHess := 0.0

	for _, idx := range indices {
		v := dm.Data[idx][featIdx]
		if math.IsNaN(v) {
			missingGrad += grads[idx]
			missingHess += hess[idx]
			continue
		}
		entries = append(entries, entry{
			value: v,
			grad:  grads[idx],
			hess:  hess[idx],
		})
	}

	if len(entries) < 2 {
		return nil
	}

	// 类别特征检测和分裂
	// 决策逻辑（与 XGBoost C++ 一致）：
	// 1. 如果 FeatureTypes 标记为 "c"，则视为类别特征
	// 2. 否则通过基数检测：unique_vals ≤ max_cat_to_onehot 用 one-hot，
	//    unique_vals ≤ max_cat_threshold 用类别分裂
	isCatSplit := false
	catThreshold := b.config.MaxCatThreshold
	onehotThreshold := 4 // default max_cat_to_onehot
	if b.config.MaxCatToOneHot > 0 {
		onehotThreshold = b.config.MaxCatToOneHot
	}
	// 检查 FeatureTypes（如果 DMatrix 有设置）
	isExplicitCat := false
	if dm.FeatureTypes != nil && featIdx < len(dm.FeatureTypes) {
		isExplicitCat = dm.FeatureTypes[featIdx] == "c" || dm.FeatureTypes[featIdx] == "categorical"
	}
	if isExplicitCat || catThreshold > 0 {
		uniqueVals := make(map[float64]bool)
		for _, e := range entries {
			uniqueVals[e.value] = true
		}
		nUnique := len(uniqueVals)
		if isExplicitCat && nUnique >= 2 {
			// 显式类别特征：如果基数 ≤ onehotThreshold 则采用 one-hot（由 CatEncoder 处理），
			// 否则用类别分裂
			isCatSplit = nUnique > onehotThreshold && nUnique <= catThreshold
		} else if !isExplicitCat && catThreshold > 0 {
			// 自动检测：基数在 (onehotThreshold, catThreshold] 范围内用类别分裂
			isCatSplit = nUnique > onehotThreshold && nUnique <= catThreshold
		}
	}

	// 按特征值排序
	sort.Slice(entries, func(i, j int) bool {
		return entries[i].value < entries[j].value
	})

	// 扫描候选，维护累积左侧和。
	leftGrad := 0.0
	leftHess := 0.0
	bestGain := -math.MaxFloat64
	bestThreshold := 0.0
	bestDefaultLeft := true

	totalGrad := nodeGrad
	totalHess := nodeHess

	if isCatSplit {
		// 类别分裂：按每个类别聚合梯度，然后按 grad/hess 比值排序后尝试分组
		type catEntry struct {
			value float64
			grad  float64
			hess  float64
		}
		catMap := make(map[float64]*catEntry)
		for _, e := range entries {
			if ce, ok := catMap[e.value]; ok {
				ce.grad += e.grad
				ce.hess += e.hess
			} else {
				catMap[e.value] = &catEntry{value: e.value, grad: e.grad, hess: e.hess}
			}
		}
		catSlice := make([]*catEntry, 0, len(catMap))
		for _, ce := range catMap {
			catSlice = append(catSlice, ce)
		}
		// 按 grad/hess 比值排序（确定性的类别顺序）
		sort.Slice(catSlice, func(i, j int) bool {
			ratioI := catSlice[i].grad / (catSlice[i].hess + 1e-10)
			ratioJ := catSlice[j].grad / (catSlice[j].hess + 1e-10)
			return ratioI < ratioJ
		})

		// 累积方式扫描类别分组
		leftGrad = 0
		leftHess = 0
		for ci := 0; ci < len(catSlice)-1; ci++ {
			leftGrad += catSlice[ci].grad
			leftHess += catSlice[ci].hess

			rightGrad := (totalGrad - missingGrad) - leftGrad
			rightHess := (totalHess - missingHess) - leftHess

			if leftHess >= b.config.MinChildWeight && rightHess >= b.config.MinChildWeight {
				gain := calcGain(leftGrad+missingGrad, leftHess+missingHess, rightGrad, rightHess, b.config.Lambda, b.config.Alpha)
				if gain > bestGain {
					bestGain = gain
					bestThreshold = catSlice[ci].value + 0.5 // 类别边界
					bestDefaultLeft = true
				}
				gain2 := calcGain(leftGrad, leftHess, rightGrad+missingGrad, rightHess+missingHess, b.config.Lambda, b.config.Alpha)
				if gain2 > bestGain {
					bestGain = gain2
					bestThreshold = catSlice[ci].value + 0.5
					bestDefaultLeft = false
				}
			}
		}
	} else {
		for i := 0; i < len(entries)-1; i++ {
			leftGrad += entries[i].grad
			leftHess += entries[i].hess

			if entries[i].value == entries[i+1].value {
				continue
			}

			rightGrad := (totalGrad - missingGrad) - leftGrad
			rightHess := (totalHess - missingHess) - leftHess

			gainLeft := 0.0
			{
				gL := leftGrad + missingGrad
				hL := leftHess + missingHess
				gR := rightGrad
				hR := rightHess
				if hL >= b.config.MinChildWeight && hR >= b.config.MinChildWeight {
					gainLeft = calcGain(gL, hL, gR, hR, b.config.Lambda, b.config.Alpha)
				}
			}

			gainRight := 0.0
			{
				gL := leftGrad
				hL := leftHess
				gR := rightGrad + missingGrad
				hR := rightHess + missingHess
				if hL >= b.config.MinChildWeight && hR >= b.config.MinChildWeight {
					gainRight = calcGain(gL, hL, gR, hR, b.config.Lambda, b.config.Alpha)
				}
			}

			gain := gainLeft
			defLeft := true
			if gainRight > gainLeft {
				gain = gainRight
				defLeft = false
			}

			if constraint != 0 {
				gl := leftGrad
				hl := leftHess
				gr := (totalGrad - missingGrad) - leftGrad
				hr := (totalHess - missingHess) - leftHess
				if defLeft {
					gl += missingGrad
					hl += missingHess
				} else {
					gr += missingGrad
					hr += missingHess
				}
				if float64(constraint)*(gl*(hr+b.config.Lambda)-gr*(hl+b.config.Lambda)) < 0 {
					gain = -1.0
				}
			}

			if gain > bestGain {
				bestGain = gain
				bestThreshold = (entries[i].value + entries[i+1].value) / 2.0
				bestDefaultLeft = defLeft
			}
		}
	}

	if bestGain <= b.config.Gamma {
		return nil
	}

	return &SplitCandidate{
		FeatureIndex: featIdx,
		Threshold:    bestThreshold,
		Gain:         bestGain,
		DefaultLeft:  bestDefaultLeft,
	}
}

// partitionSamples 根据分裂将样本索引分为左和右。
// defaultLeft 指定缺失值（NaN）的默认方向。
func (b *ExactBuilder) partitionSamples(dm *DMatrix, indices []int,
	featIdx int, threshold float64, defaultLeft bool) (left, right []int) {

	for _, idx := range indices {
		v := dm.Data[idx][featIdx]
		if math.IsNaN(v) {
			// 缺失值按 defaultLeft 方向
			if defaultLeft {
				left = append(left, idx)
			} else {
				right = append(right, idx)
			}
		} else if v <= threshold {
			left = append(left, idx)
		} else {
			right = append(right, idx)
		}
	}
	return
}

// sumGH 计算给定样本索引的梯度和 Hessian 之和。
// 使用 float64 精度累积。
func sumGH(grads, hess []float64, indices []int) (float64, float64) {
	var gSum float64
	var hSum float64
	if indices == nil {
		for i := range grads {
			gSum += grads[i]
			hSum += hess[i]
		}
	} else {
		for _, idx := range indices {
			gSum += grads[idx]
			hSum += hess[idx]
		}
	}
	return gSum, hSum
}

// calcLeafWeight 计算最优叶节点权重：w* = -∑g / (∑h + λ)
// 如果 maxDeltaStep > 0，权重会被裁剪到 [-maxDeltaStep, maxDeltaStep]。
func calcLeafWeight(sumGrad, sumHess, lambda, alpha, maxDeltaStep float64) float64 {
	denom := sumHess + lambda
	if denom == 0 {
		return 0
	}

	w := -sumGrad / denom

	// L1 正则化收缩
	if alpha > 0 {
		th := alpha / denom
		if w > th {
			w -= th
		} else if w < -th {
			w += th
		} else {
			w = 0
		}
	}

	// max_delta_step 裁剪
	if maxDeltaStep > 0 {
		w = clamp(w, -maxDeltaStep, maxDeltaStep)
	}

	return w
}

// calcGain 计算分裂的损失减少量（与 XGBoost 3.2 对齐）。
//
// loss_chg = CalcGain(GL, HL) + CalcGain(GR, HR) - CalcGain(GP, HP)
// 其中 CalcGain(g, h, λ, α) = ThresholdL1(g, α)² / (h + λ)
func calcGain(leftGrad, leftHess, rightGrad, rightHess, lambda, alpha float64) float64 {
	gL, gR, gP := leftGrad, rightGrad, leftGrad+rightGrad
	if alpha > 0 {
		gL = ThresholdL1(gL, alpha)
		gR = ThresholdL1(gR, alpha)
		gP = ThresholdL1(gP, alpha)
	}

	denomL := leftHess + lambda
	denomR := rightHess + lambda
	denomP := leftHess + rightHess + lambda

	return gL*gL/denomL + gR*gR/denomR - gP*gP/denomP
}

// refreshLeafValues 遍历树并重新计算所有叶节点的权重。
// 不改变树结构（分裂点、特征等），只更新叶值。
func refreshLeafValues(tree *RegTree, grads, hess []float64, lambda, alpha, maxDeltaStep float64) {
	_ = grads
	_ = hess
	for i := range tree.Nodes {
		if tree.Nodes[i].IsLeaf() {
			w := calcLeafWeight(tree.Nodes[i].SumGrad, tree.Nodes[i].SumHess, lambda, alpha, maxDeltaStep)
			tree.Nodes[i].LeafValue = w
		}
	}
}
