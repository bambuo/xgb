package xgb

import (
	"fmt"
	"math"
)

// GBTree 实现了梯度提升树模型。
//
// 对应 XGBoost C++ 的 GBTree（src/gbm/gbtree.cc）。
type GBTree struct {
	Config    *Config
	Trees     []*RegTree
	TreeInfo  []int // 每棵树的类别索引（多分类）
	Objective Objective
}

// NewGBTree 用给定的配置和目标函数创建新的 GBTree 模型。
func NewGBTree(cfg *Config) *GBTree {
	objective := NewObjective(cfg.Objective, cfg.NumClass)
	return &GBTree{
		Config:    cfg,
		Objective: objective,
	}
}

// BoostFromWeights 从基础分数初始化预测值。
func (gbt *GBTree) BoostFromWeights(dm *DMatrix) []float64 {
	n := dm.NumRows
	total := n
	if gbt.Config.NumClass > 1 {
		total = n * gbt.Config.NumClass
	}

	preds := make([]float64, total)
	for i := range preds {
		preds[i] = gbt.Config.BaseScore
	}

	// 如果存在基础边距则应用
	if dm.BaseMargin != nil {
		for i := 0; i < n; i++ {
			if gbt.Config.NumClass == 1 {
				preds[i] = dm.BaseMargin[i]
			} else {
				base := i * gbt.Config.NumClass
				for k := 0; k < gbt.Config.NumClass; k++ {
					preds[base+k] = dm.BaseMargin[i*int(math.Min(float64(gbt.Config.NumClass),
						float64(len(dm.BaseMargin)/n)))]
				}
			}
		}
	}

	return preds
}

// Train 运行完整的提升训练循环。
//
// 返回每个评估指标的历史记录列表，用于监控。
func (gbt *GBTree) Train(dm *DMatrix, metrics []Metric) error {
	return gbt.trainLoop(dm, metrics, 0)
}

// ContinueTrain 向现有模型添加更多树。
// 保留现有模型的树，额外的提升轮次从当前预测状态继续。
func (gbt *GBTree) ContinueTrain(dm *DMatrix, additionalRounds int, metrics []Metric) error {
	prevTrees := gbt.Config.NumTrees
	gbt.Config.NumTrees = additionalRounds
	defer func() { gbt.Config.NumTrees = prevTrees + additionalRounds }()
	return gbt.trainLoop(dm, metrics, len(gbt.Trees))
}

func (gbt *GBTree) trainLoop(dm *DMatrix, metrics []Metric, startIter int) error {
	if dm == nil {
		return ErrEmptyData
	}

	// 将标签裁剪为 {0, 1} 为二分类目标函数。
	// 特征引擎可能产生 -1/0/1 标签，但 binary:logistic
	// 期望 0（负类）/ 1（正类）。任何 <= 0 的标签 → 类 0。
	if gbt.Config.NumClass <= 1 {
		for i, v := range dm.Labels {
			if v > 0 {
				dm.Labels[i] = 1
			} else {
				dm.Labels[i] = 0
			}
		}
	}

	// 总树数（多分类下按类别计）
	totalTrees := gbt.Config.NumTrees + startIter
	if gbt.Config.NumClass > 1 {
		totalTrees = (gbt.Config.NumTrees + startIter) * gbt.Config.NumClass
	}

	// 从现有树初始化预测值
	preds := gbt.BoostFromWeights(dm)

	// 添加现有树的预测值
	for ti, tree := range gbt.Trees {
		classIdx := 0
		if ti < len(gbt.TreeInfo) {
			classIdx = gbt.TreeInfo[ti]
		}
		for i := 0; i < dm.NumRows; i++ {
			lv := tree.Predict(dm.Data[i])
			if gbt.Config.NumClass > 1 {
				preds[i*gbt.Config.NumClass+classIdx] += gbt.Config.LearningRate * lv
			} else {
				preds[i] += gbt.Config.LearningRate * lv
			}
		}
	}

	// 构建树构建器配置
	builderCfg := &ExactBuilderConfig{
		MaxDepth:         gbt.Config.MaxDepth,
		Gamma:            gbt.Config.Gamma,
		Lambda:           gbt.Config.Lambda,
		Alpha:            gbt.Config.Alpha,
		MinChildWeight:   gbt.Config.MinChildWeight,
		NumFeatures:      dm.NumCols,
		MaxBin:           gbt.Config.MaxBin,
		ColSampleByLevel: gbt.Config.ColSampleByLevel,
		ColSampleByNode:  gbt.Config.ColSampleByNode,
	}

	// 计算初始 Hessian 用于加权分位数 bin 映射器
	// 与 XGBoost 一致：bin 边界在训练前确定，使用初始 Hessian 加权
	_, initHess, _ := gbt.Objective.GetGradient(preds, dm.Labels, dm.Weights)
	if gbt.Config.NumClass > 1 {
		// 多分类时使用第一个类别的 Hessian
		classHess := make([]float64, dm.NumRows)
		for i := 0; i < dm.NumRows; i++ {
			classHess[i] = initHess[i*gbt.Config.NumClass]
		}
		initHess = classHess
	}

	// 预先计算箱映射器（GK 加权分位数，数据在树之间不变）
	maxBin := gbt.Config.MaxBin
	if maxBin <= 0 {
		maxBin = 256
	}
	bm := NewGKBinMapper(dm.Data, initHess, maxBin)

	for iter := startIter; iter < totalTrees; iter++ {
		// 确定多分类的类别
		classIdx := 0
		if gbt.Config.NumClass > 1 {
			classIdx = iter % gbt.Config.NumClass
		}

		// 计算本轮梯度
		grads, hess, err := gbt.Objective.GetGradient(preds, dm.Labels, dm.Weights)
		if err != nil {
			return fmt.Errorf("gradient computation at iter %d: %w", iter, err)
		}

		// 将梯度截断为 float32 精度，与 XGBoost GradientPair（float32）对齐
		truncateGradients(grads, hess)

		// 对于多分类，提取每个类别的梯度
		var classGrads, classHess []float64
		if gbt.Config.NumClass > 1 {
			classGrads = make([]float64, dm.NumRows)
			classHess = make([]float64, dm.NumRows)
			for i := 0; i < dm.NumRows; i++ {
				base := i * gbt.Config.NumClass
				classGrads[i] = grads[base+classIdx]
				classHess[i] = hess[base+classIdx]
			}
		} else {
			classGrads = grads
			classHess = hess
		}

		// 应用行子采样（使用 mt19937 与 XGBoost 对齐）
		sampleIndices := mtRowSample(dm.NumRows, gbt.Config.Subsample, uint32(gbt.Config.Seed)+uint32(iter))

		// 应用列子采样
		featureMask := mtColSample(dm.NumCols, gbt.Config.ColSampleByTree, uint32(gbt.Config.Seed)+uint32(iter))

		// 构建单棵树
		builder := NewHistBuilder(builderCfg, bm)
		builder.SetIterSeed(gbt.Config.Seed + int64(iter))
		if err := builder.Build(dm, classGrads, classHess, sampleIndices, featureMask); err != nil {
			return fmt.Errorf("tree build at iter %d: %w", iter, err)
		}

		// 将树添加到集成
		gbt.Trees = append(gbt.Trees, builder.tree)
		gbt.TreeInfo = append(gbt.TreeInfo, classIdx)

		// 更新预测值
		for i := 0; i < dm.NumRows; i++ {
			row := dm.Data[i]
			leafValue := builder.tree.Predict(row)

			if gbt.Config.NumClass > 1 {
				base := i * gbt.Config.NumClass
				preds[base+classIdx] += gbt.Config.LearningRate * leafValue
			} else {
				preds[i] += gbt.Config.LearningRate * leafValue
			}
		}

		// 定期打印评估指标
		if gbt.Config.Verbosity >= 2 && iter%10 == 0 {
			score := fmt.Sprintf("[%d/%d]", iter+1, totalTrees)
			for _, m := range metrics {
				score += fmt.Sprintf(" %s=%.4f", m.Name(), m.Evaluate(preds, dm))
			}
			fmt.Println(score)
		}
	}

	return nil
}

// Predict 返回单个样本的原始预测值。
// 对于 binary:logistic，返回对数几率；调用 PredTransform 获取概率。
func (gbt *GBTree) Predict(sample []float64) float64 {
	if len(gbt.Trees) == 0 {
		return gbt.Config.BaseScore
	}

	score := gbt.Config.BaseScore
	for _, tree := range gbt.Trees {
		score += gbt.Config.LearningRate * tree.Predict(sample)
	}
	return score
}

// PredictProb 返回二分类的概率值（0-1）。
func (gbt *GBTree) PredictProb(sample []float64) float64 {
	return sigmoid(gbt.Predict(sample))
}

// PredictBatch 返回多个样本的预测值。
func (gbt *GBTree) PredictBatch(data [][]float64) []float64 {
	out := make([]float64, len(data))
	for i, row := range data {
		out[i] = gbt.Predict(row)
	}
	return out
}

// rowSample 为子采样生成行索引的随机子集。
func rowSample(numRows int, ratio float64, seed int64) []int {
	if ratio >= 1.0 {
		indices := make([]int, numRows)
		for i := range indices {
			indices[i] = i
		}
		return indices
	}

	// 使用线性同余生成器进行简单确定性采样
	// 这与 XGBoost 的采样方式一致以保证可重现性
	indices := make([]int, 0, int(float64(numRows)*ratio))
	state := uint64(seed)

	for i := 0; i < numRows; i++ {
		state = state*6364136223846793005 + 1442695040888963407
		rnd := float64((state>>33)&0x3FFFFFFF) / float64(0x3FFFFFFF)
		if rnd < ratio {
			indices = append(indices, i)
		}
	}
	return indices
}

// colSample 为列子采样生成特征索引的随机子集。
func colSample(numCols int, ratio float64, seed int64) []int {
	if ratio >= 1.0 {
		features := make([]int, numCols)
		for i := range features {
			features[i] = i
		}
		return features
	}

	features := make([]int, 0, int(float64(numCols)*ratio))
	state := uint64(seed)

	for i := 0; i < numCols; i++ {
		state = state*6364136223846793005 + 1442695040888963407
		rnd := float64((state>>33)&0x3FFFFFFF) / float64(0x3FFFFFFF)
		if rnd < ratio {
			features = append(features, i)
		}
	}
	return features
}

// colSampleFromMask 从已有特征掩码中进行子采样。
// 用于 colsample_bylevel 和 colsample_bynode。
func colSampleFromMask(mask []int, ratio float64, seed int64) []int {
	if mask == nil || ratio >= 1.0 {
		return mask
	}

	features := make([]int, 0, int(float64(len(mask))*ratio))
	state := uint64(seed)

	for _, f := range mask {
		state = state*6364136223846793005 + 1442695040888963407
		rnd := float64((state>>33)&0x3FFFFFFF) / float64(0x3FFFFFFF)
		if rnd < ratio {
			features = append(features, f)
		}
	}
	return features
}
