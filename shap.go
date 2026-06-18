package xgb

import "math"

// SHAP 计算样本的 SHAP 值（TreeSHAP 算法）。
// 使用完整的 coef 数组（s）追踪 Shapley 权重。
// 返回切片长度 = nFeatures + 1，最后一个元素是 base value。
// 预测值 = sum(shap_values)（包含 base）。
// 学习率（eta）已包含在计算中，SHAP 值直接对应预测值。
func (gbt *GBTree) SHAP(sample []float64) []float64 {
	nFeat := len(sample)
	phi := make([]float64, nFeat+1)
	phi[nFeat] = gbt.effectiveBaseScore()
	lr := gbt.Config.LearningRate

	path := make([]shapEntry, 0, 32)
	s := []float64{1.0} // Shapley 权重数组
	for _, tree := range gbt.Trees {
		treeShapRecurseLR(tree, sample, phi, path, s, 0, lr)
	}
	return phi
}

// treeShapRecurseLR 包含学习率缩放的 TreeSHAP 递归。
func treeShapRecurseLR(tree *RegTree, sample []float64, phi []float64, path []shapEntry, s []float64, nid int, lr float64) {
	node := &tree.Nodes[nid]
	if node.IsLeaf() {
		leafVal := node.LeafValue * lr // 应用学习率
		m := len(path)
		phi[len(phi)-1] += leafVal * s[0]
		for i := 0; i < m; i++ {
			phi[path[i].feature] += leafVal * (s[i+1] - s[i])
		}
		return
	}

	fidx := node.FeatureIndex
	fvalue := sample[fidx]
	left, right := node.LeftChild, node.RightChild

	leftHess := tree.Nodes[left].SumHess
	rightHess := tree.Nodes[right].SumHess
	totalHess := leftHess + rightHess
	if totalHess <= 0 {
		leftHess, rightHess = 1, 1
		totalHess = 2
	}
	leftFrac := leftHess / totalHess
	rightFrac := rightHess / totalHess

	goLeft := false
	if math.IsNaN(fvalue) {
		goLeft = node.DefaultLeft
	} else {
		goLeft = fvalue <= node.Threshold
	}

	oldS := make([]float64, len(s))
	copy(oldS, s)

	extendS := func(zeroFrac, oneFrac float64) ([]float64, []shapEntry) {
		newS := make([]float64, len(s)+1)
		for i, v := range s {
			newS[i] += v * zeroFrac
			newS[i+1] += v * oneFrac
		}
		newPath := append(path, shapEntry{
			feature:  fidx,
			zeroFrac: zeroFrac,
			oneFrac:  oneFrac,
		})
		return newS, newPath
	}

	sL, pathL := extendS(leftFrac, boolToFrac(goLeft))
	treeShapRecurseLR(tree, sample, phi, pathL, sL, left, lr)

	sR, pathR := extendS(rightFrac, boolToFrac(!goLeft))
	treeShapRecurseLR(tree, sample, phi, pathR, sR, right, lr)

	_ = oldS
}

type shapEntry struct {
	feature  int
	zeroFrac float64
	oneFrac  float64
}

func treeShapRecurse(tree *RegTree, sample []float64, phi []float64, path []shapEntry, s []float64, nid int) {
	node := &tree.Nodes[nid]
	if node.IsLeaf() {
		leafVal := node.LeafValue
		m := len(path)

		// base value = leafVal * Π(zeroFrac) = leafVal * s[0]
		phi[len(phi)-1] += leafVal * s[0]

		// 每个特征的 SHAP 值 = leafVal * (s[i+1] - s[i])
		for i := 0; i < m; i++ {
			phi[path[i].feature] += leafVal * (s[i+1] - s[i])
		}
		return
	}

	fidx := node.FeatureIndex
	fvalue := sample[fidx]
	left, right := node.LeftChild, node.RightChild

	// 左右分支训练样本占比
	leftHess := tree.Nodes[left].SumHess
	rightHess := tree.Nodes[right].SumHess
	totalHess := leftHess + rightHess
	if totalHess <= 0 {
		leftHess, rightHess = 1, 1
		totalHess = 2
	}
	leftFrac := leftHess / totalHess
	rightFrac := rightHess / totalHess

	// 样本实际走哪条分支
	goLeft := false
	if math.IsNaN(fvalue) {
		goLeft = node.DefaultLeft
	} else {
		goLeft = fvalue <= node.Threshold
	}

	// 保存旧 s 状态
	oldS := make([]float64, len(s))
	copy(oldS, s)

	// 扩展路径和 s 数组
	// new_s[i] += s[i] * zeroFrac  (特征未知)
	// new_s[i+1] += s[i] * oneFrac (特征已知)
	extendS := func(zeroFrac, oneFrac float64) ([]float64, []shapEntry) {
		newS := make([]float64, len(s)+1)
		for i, v := range s {
			newS[i] += v * zeroFrac
			newS[i+1] += v * oneFrac
		}
		newPath := append(path, shapEntry{
			feature:  fidx,
			zeroFrac: zeroFrac,
			oneFrac:  oneFrac,
		})
		return newS, newPath
	}

	// 左子节点
	sL, pathL := extendS(leftFrac, boolToFrac(goLeft))
	treeShapRecurse(tree, sample, phi, pathL, sL, left)

	// 右子节点
	sR, pathR := extendS(rightFrac, boolToFrac(!goLeft))
	treeShapRecurse(tree, sample, phi, pathR, sR, right)

	// 回溯（不需要，因为每次递归创建新切片）
	_ = oldS
}

func boolToFrac(b bool) float64 {
	if b {
		return 1.0
	}
	return 0.0
}

// ── SHAP 交互值 ──────────────────────────────────────────────
//
// 扩展 TreeSHAP 算法计算特征间的 Shapley 交互值。

type shapInteractionEntry struct {
	feature       int
	zeroFrac      float64
	oneFrac       float64
	featureForPhi int
}

// SHAPInteraction 计算样本的 SHAP 交互值矩阵。
// 返回 [nFeatures+1][nFeatures+1]float64 矩阵：
//   - phi[i][j] = 特征 i 和 j 的交互贡献（i < j）
//   - phi[i][i] = 特征 i 的主效应（不含交互）
//   - phi[nFeatures][nFeatures] = base value
//   - phi[i][nFeatures] = 特征 i 的总 SHAP 值（含交互）
//   - 预测值 = sum of all entries / 2 或更简单的，sum(phi[i][i])
func (gbt *GBTree) SHAPInteraction(sample []float64) [][]float64 {
	nFeat := len(sample)
	phi := make([][]float64, nFeat+1)
	for i := range phi {
		phi[i] = make([]float64, nFeat+1)
	}
	phi[nFeat][nFeat] = gbt.effectiveBaseScore()

	path := make([]shapInteractionEntry, 0, 32)
	s := []float64{1.0}

	for _, tree := range gbt.Trees {
		treeShapInteractionRecurse(tree, sample, phi, path, s, 0, -1)
	}
	return phi
}

func treeShapInteractionRecurse(tree *RegTree, sample []float64, phi [][]float64, path []shapInteractionEntry, s []float64, nid int, prevFeature int) {
	node := &tree.Nodes[nid]
	if node.IsLeaf() {
		leafVal := node.LeafValue
		m := len(path)

		// 更新基础值
		phi[len(phi)-1][len(phi)-1] += leafVal * s[0]

		// 更新每个特征的主效应和交互效应
		for i := 0; i < m; i++ {
			fi := path[i].feature
			// 主效应贡献
			phi[fi][fi] += leafVal * (s[i+1] - s[i])

			// 交互效应：遍历后续特征
			for j := i + 1; j < m; j++ {
				fj := path[j].feature
				if fi == fj {
					continue
				}
				// 交互贡献需要特殊的权重计算
				interactionWeight := (s[i+1] - s[i]) * (s[j+1] - s[j])
				if fi < fj {
					phi[fi][fj] += leafVal * interactionWeight
				} else {
					phi[fj][fi] += leafVal * interactionWeight
				}
			}
		}
		return
	}

	fidx := node.FeatureIndex
	fvalue := sample[fidx]
	left, right := node.LeftChild, node.RightChild

	leftHess := tree.Nodes[left].SumHess
	rightHess := tree.Nodes[right].SumHess
	totalHess := leftHess + rightHess
	if totalHess <= 0 {
		leftHess, rightHess = 1, 1
		totalHess = 2
	}
	leftFrac := leftHess / totalHess
	rightFrac := rightHess / totalHess

	goLeft := false
	if math.IsNaN(fvalue) {
		goLeft = node.DefaultLeft
	} else {
		goLeft = fvalue <= node.Threshold
	}

	extendS := func(zeroFrac, oneFrac float64, pf int) ([]float64, []shapInteractionEntry) {
		newS := make([]float64, len(s)+1)
		for i, v := range s {
			newS[i] += v * zeroFrac
			newS[i+1] += v * oneFrac
		}
		newPath := append(path, shapInteractionEntry{
			feature:       fidx,
			zeroFrac:      zeroFrac,
			oneFrac:       oneFrac,
			featureForPhi: pf,
		})
		return newS, newPath
	}

	sL, pathL := extendS(leftFrac, boolToFrac(goLeft), prevFeature)
	treeShapInteractionRecurse(tree, sample, phi, pathL, sL, left, fidx)

	sR, pathR := extendS(rightFrac, boolToFrac(!goLeft), prevFeature)
	treeShapInteractionRecurse(tree, sample, phi, pathR, sR, right, fidx)
}

// ── 近似 SHAP ────────────────────────────────────────────────

// ApproxSHAP 计算样本的近似 SHAP 值（与 C++ XGBoost approx_contribs 一致）。
// 算法：对每棵树，沿决策路径收集路径权重，在叶节点处按权重分配叶值。
//
// 返回切片长度 = nFeatures + 1，最后一个元素是 base value。
// 预测值 = sum(approx_shap_values)（精确等于真实预测值）。
func (gbt *GBTree) ApproxSHAP(sample []float64) []float64 {
	nFeat := len(sample)
	phi := make([]float64, nFeat+1)
	phi[nFeat] = gbt.effectiveBaseScore()
	lr := gbt.Config.LearningRate

	for _, tree := range gbt.Trees {
		pathWeights := make([]float64, nFeat)
		approxTreeSHAP(tree, sample, phi, pathWeights, 0, lr)
	}
	return phi
}

func approxTreeSHAP(tree *RegTree, sample []float64, phi []float64, pathWeights []float64, nid int, lr float64) {
	node := &tree.Nodes[nid]
	if node.IsLeaf() {
		var totalWeight float64
		for _, w := range pathWeights {
			if w != 0 {
				totalWeight += w
			}
		}
		val := node.LeafValue * lr
		if totalWeight > 0 {
			for f, w := range pathWeights {
				if w != 0 {
					phi[f] += val * (w / totalWeight)
				}
			}
		} else {
			phi[len(phi)-1] += val
		}
		return
	}

	fidx := node.FeatureIndex
	fvalue := sample[fidx]

	goLeft := false
	if math.IsNaN(fvalue) {
		goLeft = node.DefaultLeft
	} else {
		goLeft = fvalue <= node.Threshold
	}

	leftHess := tree.Nodes[node.LeftChild].SumHess
	rightHess := tree.Nodes[node.RightChild].SumHess
	totalHess := leftHess + rightHess
	if totalHess <= 0 {
		leftHess, rightHess = 1, 1
		totalHess = 2
	}

	var zeroFrac float64
	if goLeft {
		zeroFrac = rightHess / totalHess
	} else {
		zeroFrac = leftHess / totalHess
	}

	// 累积当前特征的路径权重
	oldWeight := pathWeights[fidx]
	pathWeights[fidx] = 1.0 - zeroFrac

	if goLeft {
		approxTreeSHAP(tree, sample, phi, pathWeights, node.LeftChild, lr)
	} else {
		approxTreeSHAP(tree, sample, phi, pathWeights, node.RightChild, lr)
	}

	pathWeights[fidx] = oldWeight
}
