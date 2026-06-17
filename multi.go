package xgb

import (
	"fmt"
	"math"
)

// MultiTargetModel 实现多目标/多输出梯度提升（与 C++ multi_strategy 一致）。
// C++ 行为：
//   - "one_output_per_tree": 每棵树只优化一个目标（当前实现）
//   - "multi_output_tree": 共享树结构，每个叶节点存储所有目标的叶值（多维叶值）
//
// 本实现：使用 vector leaf（IsVectorLeaf 模式），所有目标共享树结构。

// MultiTargetTree 存储多输出树的向量叶值。
// 每棵树共享结构，叶节点处为每个目标存储独立叶值。
type MultiTargetTree struct {
	Tree       *RegTree
	LeafValues [][]float64 // [nodeIdx] → []float64{target0_leaf, target1_leaf, ...}
}

// MultiTargetModel 实现多目标梯度提升。
type MultiTargetModel struct {
	Config    *Config
	Trees     []*MultiTargetTree
	Objective Objective
	NumTarget int
}

// NewMultiTargetModel 创建多目标模型。
func NewMultiTargetModel(cfg *Config, numTarget int) *MultiTargetModel {
	return &MultiTargetModel{
		Config:    cfg,
		NumTarget: numTarget,
	}
}

// Train 训练多目标模型。
// data: [numRows][numCols] 特征矩阵
// labels: [numRows][numTargets] 标签矩阵
func (mt *MultiTargetModel) Train(data [][]float64, labels [][]float64, metrics []Metric) ([]EvalResult, error) {
	if len(data) == 0 {
		return nil, ErrEmptyData
	}
	n, nFeat := len(data), len(data[0])
	numTarget := mt.NumTarget
	if numTarget <= 0 {
		numTarget = len(labels[0])
	}

	if mt.Config.CustomEval != nil {
		metrics = append(metrics, &CustomMetric{Fn: mt.Config.CustomEval})
	}

	// 初始化目标函数（使用第一个目标的配置）
	if mt.Objective == nil {
		cfg := *mt.Config
		cfg.NumClass = 1 // 每个目标独立处理
		mt.Objective = NewObjective(cfg.Objective, 1, cfg.ScalePosWeight)
	}

	// 预测值 [numRows][numTargets]
	preds := make([][]float64, n)
	for i := range preds {
		preds[i] = make([]float64, numTarget)
		preds[i][0] = mt.Config.BaseScore // 每个目标初始值
	}

	// 构建器配置
	builderCfg := &ExactBuilderConfig{
		MaxDepth: mt.Config.MaxDepth, Gamma: mt.Config.Gamma,
		Lambda: mt.Config.Lambda, Alpha: mt.Config.Alpha,
		MinChildWeight: mt.Config.MinChildWeight, MaxDeltaStep: mt.Config.MaxDeltaStep,
		NumFeatures: nFeat, MaxBin: mt.Config.MaxBin,
	}

	// bin mapper
	var binMapper *BinMapper
	if mt.Config.TreeMethod == "hist" || mt.Config.TreeMethod == "" {
		flatLabels := make([]float64, n)
		for i := range flatLabels {
			flatLabels[i] = labels[i][0]
		}
		dm, _ := NewDMatrix(data, flatLabels)
		_, initHess, _ := mt.Objective.GetGradient(preds[0], flatLabels, nil)
		maxBin := mt.Config.MaxBin
		if maxBin <= 0 {
			maxBin = 256
		}
		binMapper = NewGKBinMapper(dm.Data, initHess, maxBin)
	}

	var history []EvalResult
	lr := mt.Config.LearningRate

	for iter := 0; iter < mt.Config.NumTrees; iter++ {
		// C++ multi_output_tree: 同时计算所有目标的梯度
		allGrads := make([][]float64, numTarget)
		allHess := make([][]float64, numTarget)
		for t := 0; t < numTarget; t++ {
			targetPreds := make([]float64, n)
			for i := 0; i < n; i++ {
				targetPreds[i] = preds[i][t]
			}
			targetLabels := make([]float64, n)
			for i := 0; i < n; i++ {
				targetLabels[i] = labels[i][t]
			}
			allGrads[t], allHess[t], _ = mt.Objective.GetGradient(targetPreds, targetLabels, nil)
		}

		// 构建共享树结构（使用第一个目标的梯度）
		builder := NewHistBuilder(builderCfg, binMapper)
		builder.SetIterSeed(mt.Config.Seed + int64(iter))

		dm, _ := NewDMatrix(data, make([]float64, n))
		if err := builder.Build(dm, allGrads[0], allHess[0], nil, nil); err != nil {
			return history, fmt.Errorf("tree build at %d: %w", iter, err)
		}

		// 为每个节点计算所有目标的叶值
		leafVals := make([][]float64, len(builder.tree.Nodes))
		for ni := range builder.tree.Nodes {
			if builder.tree.Nodes[ni].IsLeaf() {
				leafVals[ni] = make([]float64, numTarget)
				// 计算从根到该叶节点的样本索引
				var sampleIdx []int
				for i := 0; i < n; i++ {
					if belongsToNode(builder.tree, data[i], ni) {
						sampleIdx = append(sampleIdx, i)
					}
				}
				if len(sampleIdx) == 0 {
					continue
				}
				for t := 0; t < numTarget; t++ {
					g, h := sumGHTarget(allGrads[t], allHess[t], sampleIdx)
					leafVals[ni][t] = calcLeafWeight(g, h, mt.Config.Lambda, mt.Config.Alpha, mt.Config.MaxDeltaStep)
				}
			}
		}

		// 保存多输出树
		mtt := &MultiTargetTree{
			Tree:       builder.tree,
			LeafValues: leafVals,
		}
		mt.Trees = append(mt.Trees, mtt)

		// 更新预测值（所有目标）
		for i := 0; i < n; i++ {
			leafIdx := builder.tree.GetLeafIndex(data[i])
			for t := 0; t < numTarget; t++ {
				preds[i][t] += lr * leafVals[leafIdx][t]
			}
		}

		// 评估
		ev := EvalResult{Iteration: iter, Metrics: make(map[string]float64)}
		if len(metrics) > 0 {
			for t := 0; t < numTarget; t++ {
				targetPreds := make([]float64, n)
				for i := 0; i < n; i++ {
					targetPreds[i] = preds[i][t]
				}
				targetLabels := make([]float64, n)
				for i := 0; i < n; i++ {
					targetLabels[i] = labels[i][t]
				}
				tdm, _ := NewDMatrix(data, targetLabels)
				for _, m := range metrics {
					ev.Metrics[fmt.Sprintf("%s_t%d", m.Name(), t)] = m.Evaluate(targetPreds, tdm)
				}
			}
		}
		history = append(history, ev)
	}
	return history, nil
}

// Predict 返回单个样本在所有目标上的预测值。
func (mt *MultiTargetModel) Predict(sample []float64) []float64 {
	preds := make([]float64, mt.NumTarget)
	for _, mtt := range mt.Trees {
		leafIdx := mtt.Tree.GetLeafIndex(sample)
		for t := 0; t < mt.NumTarget && t < len(mtt.LeafValues[leafIdx]); t++ {
			preds[t] += mt.Config.LearningRate * mtt.LeafValues[leafIdx][t]
		}
	}
	return preds
}

// PredictBatch 返回多个样本的预测值。
func (mt *MultiTargetModel) PredictBatch(data [][]float64) [][]float64 {
	preds := make([][]float64, len(data))
	for i, row := range data {
		preds[i] = mt.Predict(row)
	}
	return preds
}

// belongsToNode 检查样本是否属于指定节点。
func belongsToNode(tree *RegTree, row []float64, nodeIdx int) bool {
	nid := 0
	for nid != nodeIdx {
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
	return true
}

// sumGHTarget 计算梯度和 Hessian 的和。
func sumGHTarget(grads, hess []float64, indices []int) (float64, float64) {
	var g, h float64
	for _, idx := range indices {
		if idx < len(grads) {
			g += grads[idx]
			h += hess[idx]
		}
	}
	return g, h
}
