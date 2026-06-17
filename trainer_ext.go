package xgb

import (
	"fmt"
	"math"
)

// splitResultFromHist 保存从累加直方图找到的最佳分裂。
type splitResultFromHist struct {
	feature     int
	bin         int
	gain        float64
	sumGradLeft float64
	sumHessLeft float64
	defaultLeft bool
}

// ExternalTrain 使用外部内存（分块数据）训练模型（与 C++ External Memory 一致）。
// - 逐块加载数据，每块处理完后丢弃（不驻留内存）
// - 每层级重新扫描所有块构建子节点直方图
// - hist 方法专用
func (gbt *GBTree) ExternalTrain(dmc *DMatrixChunked, metrics []Metric) ([]EvalResult, error) {
	if gbt.Config.TreeMethod == "exact" {
		return nil, fmt.Errorf("external memory only supports hist method")
	}
	if gbt.Config.CustomEval != nil {
		metrics = append(metrics, &CustomMetric{Fn: gbt.Config.CustomEval})
	}

	firstChunk, err := dmc.LoadChunk(0)
	if err != nil {
		return nil, fmt.Errorf("load chunk 0: %w", err)
	}
	if firstChunk == nil {
		return nil, fmt.Errorf("no data found")
	}
	nCols, numChunks := firstChunk.NumCols, dmc.NumChunks()

	totalTrees := gbt.Config.NumTrees
	if gbt.Config.NumClass > 1 {
		totalTrees = gbt.Config.NumTrees * gbt.Config.NumClass
	}

	builderCfg := &ExactBuilderConfig{
		MaxDepth:               gbt.Config.MaxDepth,
		Gamma:                  gbt.Config.Gamma,
		Lambda:                 gbt.Config.Lambda,
		Alpha:                  gbt.Config.Alpha,
		MinChildWeight:         gbt.Config.MinChildWeight,
		MaxDeltaStep:           gbt.Config.MaxDeltaStep,
		MaxLeaves:              gbt.Config.MaxLeaves,
		NumFeatures:            nCols,
		MaxBin:                 gbt.Config.MaxBin,
		ColSampleByLevel:       gbt.Config.ColSampleByLevel,
		ColSampleByNode:        gbt.Config.ColSampleByNode,
		MonotoneConstraints:    gbt.Config.MonotoneConstraints,
		InteractionConstraints: gbt.Config.InteractionConstraints,
	}

	// 预计算 bin mapper
	preds := gbt.BoostFromWeights(firstChunk)
	_, initHess, _ := gbt.Objective.GetGradient(preds, firstChunk.Labels, firstChunk.Weights)
	maxBin := gbt.Config.MaxBin
	if maxBin <= 0 {
		maxBin = 256
	}
	binMapper := NewGKBinMapper(firstChunk.Data, initHess, maxBin)

	featureGroup := make([]int, nCols)
	for f := range featureGroup {
		featureGroup[f] = -1
	}
	for gidx, group := range gbt.Config.InteractionConstraints {
		for _, f := range group {
			if f >= 0 && f < nCols {
				featureGroup[f] = gidx
			}
		}
	}

	var history []EvalResult

	for iter := 0; iter < totalTrees; iter++ {
		classIdx := 0
		if gbt.Config.NumClass > 1 {
			classIdx = iter % gbt.Config.NumClass
		}

		tree := buildExternalTree(dmc, numChunks, nCols, classIdx,
			gbt, binMapper, builderCfg, featureGroup, iter)
		if tree == nil {
			continue
		}
		gbt.Trees = append(gbt.Trees, tree)
		gbt.TreeInfo = append(gbt.TreeInfo, classIdx)

		if len(metrics) > 0 {
			evalChunk, _ := dmc.LoadChunk(0)
			if evalChunk != nil {
				evalPreds := predictExternal(gbt, evalChunk.Data)
				ev := EvalResult{Iteration: iter, Metrics: make(map[string]float64)}
				for _, m := range metrics {
					ev.Metrics[m.Name()] = m.Evaluate(evalPreds, evalChunk)
				}
				history = append(history, ev)
			}
		}
	}
	return history, nil
}

// buildExternalTree 构建单棵树（外存模式）。
func buildExternalTree(dmc *DMatrixChunked, numChunks, nCols, classIdx int,
	gbt *GBTree, binMapper *BinMapper, builderCfg *ExactBuilderConfig,
	featureGroup []int, iter int) *RegTree {

	tree := NewRegTree(nCols)
	tree.Param.MaxDepth = builderCfg.MaxDepth

	var totalGrad, totalHess float64
	for ci := 0; ci < numChunks; ci++ {
		chunk, err := dmc.LoadChunk(ci)
		if err != nil || chunk == nil {
			continue
		}
		grads, hess := computeExternalGradients(chunk, gbt, classIdx)
		for i := range grads {
			totalGrad += grads[i]
			totalHess += hess[i]
		}
	}

	rootIdx := tree.InitRoot(totalGrad, totalHess)
	tree.SetLeaf(rootIdx, calcLeafWeight(totalGrad, totalHess,
		builderCfg.Lambda, builderCfg.Alpha, builderCfg.MaxDeltaStep))

	type extNode struct{ nodeIdx, depth, group int }
	current := []extNode{{nodeIdx: rootIdx, depth: 0, group: -1}}

	for depth := 0; depth < builderCfg.MaxDepth; depth++ {
		if len(current) == 0 {
			break
		}
		var next []extNode
		for _, nd := range current {
			nodeGrad, nodeHess := tree.Nodes[nd.nodeIdx].SumGrad, tree.Nodes[nd.nodeIdx].SumHess

			hist := buildExternalHistogram(dmc, numChunks, nCols, classIdx,
				gbt, binMapper, tree, nd.nodeIdx)

			best := findExtBestSplit(hist, builderCfg, featureGroup, nd.group, nodeGrad, nodeHess)
			if best == nil || best.gain <= builderCfg.Gamma {
				continue
			}

			leftChild, rightChild := tree.AddNode(), tree.AddNode()
			threshold := binMapper.Threshold(best.feature, best.bin)
			tree.SetSplit(nd.nodeIdx, best.feature, threshold, best.gain,
				leftChild, rightChild, best.defaultLeft)

			leftGrad, leftHess := best.sumGradLeft, best.sumHessLeft
			rightGrad, rightHess := nodeGrad-leftGrad, nodeHess-leftHess
			tree.Nodes[leftChild].SumGrad = leftGrad
			tree.Nodes[leftChild].SumHess = leftHess
			tree.Nodes[rightChild].SumGrad = rightGrad
			tree.Nodes[rightChild].SumHess = rightHess
			tree.SetLeaf(leftChild, calcLeafWeight(leftGrad, leftHess,
				builderCfg.Lambda, builderCfg.Alpha, builderCfg.MaxDeltaStep))
			tree.SetLeaf(rightChild, calcLeafWeight(rightGrad, rightHess,
				builderCfg.Lambda, builderCfg.Alpha, builderCfg.MaxDeltaStep))

			childGroup := nd.group
			if fg := featureGroup[best.feature]; fg >= 0 {
				childGroup = fg
			}
			next = append(next, extNode{nodeIdx: leftChild, depth: depth + 1, group: childGroup})
			next = append(next, extNode{nodeIdx: rightChild, depth: depth + 1, group: childGroup})
		}
		current = next
	}
	return tree
}

// buildExternalHistogram 为指定节点重新扫描所有 chunk 构建直方图。
func buildExternalHistogram(dmc *DMatrixChunked, numChunks, nCols, classIdx int,
	gbt *GBTree, binMapper *BinMapper, tree *RegTree, nodeIdx int) [][]struct{ grad, hess float64 } {

	hist := make([][]struct{ grad, hess float64 }, nCols)
	for f := 0; f < nCols; f++ {
		hist[f] = make([]struct{ grad, hess float64 }, binMapper.NumBinsForFeature(f))
	}

	for ci := 0; ci < numChunks; ci++ {
		chunk, err := dmc.LoadChunk(ci)
		if err != nil || chunk == nil {
			continue
		}
		grads, hess := computeExternalGradients(chunk, gbt, classIdx)
		for i := 0; i < chunk.NumRows; i++ {
			if !belongsToExternalNode(tree, chunk.Data[i], nodeIdx) {
				continue
			}
			for f := 0; f < nCols; f++ {
				bin := binMapper.Bin(f, chunk.Data[i][f])
				if bin >= 0 && bin < len(hist[f]) {
					hist[f][bin].grad += grads[i]
					hist[f][bin].hess += hess[i]
				}
			}
		}
	}
	return hist
}

// belongsToExternalNode 检查样本是否经过决策路径到达指定节点。
func belongsToExternalNode(tree *RegTree, row []float64, nodeIdx int) bool {
	nid := 0
	for nid != nodeIdx && nid >= 0 {
		node := &tree.Nodes[nid]
		if node.IsLeaf() {
			return false
		}
		val := row[node.FeatureIndex]
		var goLeft bool
		if math.IsNaN(val) {
			goLeft = node.DefaultLeft
		} else {
			goLeft = val <= node.Threshold
		}
		if goLeft {
			nid = node.LeftChild
		} else {
			nid = node.RightChild
		}
	}
	return nid == nodeIdx
}

// computeExternalGradients 计算 chunk 的梯度。
func computeExternalGradients(chunk *DMatrix, gbt *GBTree, classIdx int) ([]float64, []float64) {
	preds := predictExternal(gbt, chunk.Data)
	grads, hess, _ := gbt.Objective.GetGradient(preds, chunk.Labels, chunk.Weights)
	if gbt.Config.NumClass > 1 {
		cg := make([]float64, chunk.NumRows)
		ch := make([]float64, chunk.NumRows)
		for i := 0; i < chunk.NumRows; i++ {
			cg[i] = grads[i*gbt.Config.NumClass+classIdx]
			ch[i] = hess[i*gbt.Config.NumClass+classIdx]
		}
		return cg, ch
	}
	return grads, hess
}

// predictExternal 使用当前所有树计算预测值。
func predictExternal(gbt *GBTree, data [][]float64) []float64 {
	preds := make([]float64, len(data))
	bs := gbt.effectiveBaseScore()
	for i := range preds {
		preds[i] = bs
	}
	lr := gbt.Config.LearningRate
	for _, tree := range gbt.Trees {
		for i, row := range data {
			preds[i] += lr * tree.Predict(row)
		}
	}
	return preds
}

// findExtBestSplit 从直方图找最佳分裂。
func findExtBestSplit(hist [][]struct{ grad, hess float64 },
	cfg *ExactBuilderConfig, featureGroup []int, pathGroup int,
	nodeGrad, nodeHess float64) *splitResultFromHist {

	var best *splitResultFromHist
	bestGain := -math.MaxFloat64

	for f := 0; f < len(hist); f++ {
		nBins := len(hist[f])
		if nBins < 2 {
			continue
		}
		if pathGroup >= 0 && f < len(featureGroup) && featureGroup[f] >= 0 && featureGroup[f] != pathGroup {
			continue
		}

		var sumGradLeft, sumHessLeft float64
		for b := 0; b < nBins-1; b++ {
			sumGradLeft += hist[f][b].grad
			sumHessLeft += hist[f][b].hess
			if hist[f][b].hess == 0 {
				continue
			}

			rightGrad := nodeGrad - sumGradLeft
			rightHess := nodeHess - sumHessLeft
			if sumHessLeft < cfg.MinChildWeight || rightHess < cfg.MinChildWeight {
				continue
			}

			gain := calcGain(sumGradLeft, sumHessLeft, rightGrad, rightHess, cfg.Lambda, cfg.Alpha)
			if gain > bestGain {
				bestGain = gain
				best = &splitResultFromHist{
					feature: f, bin: b, gain: gain,
					sumGradLeft: sumGradLeft, sumHessLeft: sumHessLeft,
					defaultLeft: true,
				}
			}
		}
	}
	if bestGain <= cfg.Gamma {
		return nil
	}
	return best
}
