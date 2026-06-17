package xgb

import (
	"fmt"
	"io"
	"math"
)

// EvalResult 保存单轮评估结果。
type EvalResult struct {
	Iteration int
	Metrics   map[string]float64
}

// GBTree 实现了梯度提升树模型。
//
// 对应 XGBoost C++ 的 GBTree（src/gbm/gbtree.cc）。
type GBTree struct {
	Config    *Config
	Trees     []*RegTree
	TreeInfo  []int // 每棵树的类别索引（多分类）
	Objective Objective

	// 早停状态
	bestEvalScore float64 // 最佳验证集指标值
	bestIteration int     // 最佳指标对应的迭代轮次
}

// NewGBTree 用给定的配置和目标函数创建新的 GBTree 模型。
func NewGBTree(cfg *Config) *GBTree {
	var objective Objective
	if cfg.CustomObjective != nil {
		objective = &CustomObj{
			ObjName: "custom",
			Fn:      cfg.CustomObjective,
		}
	} else {
		objective = NewObjective(cfg.Objective, cfg.NumClass, cfg.ScalePosWeight)
	}

	// 为可配置的 objective 设置特定参数
	if ph, ok := objective.(*PseudoHuberError); ok {
		ph.Delta = cfg.HuberSlope
	}
	if qe, ok := objective.(*QuantileError); ok {
		qe.Alpha = cfg.QuantileAlpha
	}
	if tr, ok := objective.(*TweedieRegression); ok {
		tr.VariancePower = cfg.TweedieVariancePower
	}
	if rnk, ok := objective.(*RankNDCG); ok {
		rnk.PairMethod = cfg.LambdaRankPairMethod
		rnk.NumPairPerSample = cfg.LambdaRankNumPairPerSample
		rnk.NDCGExpGain = cfg.NDCGExpGain
		rnk.Normalization = cfg.LambdaRankNormalization
	}

	return &GBTree{
		Config:    cfg,
		Objective: objective,
	}
}

// logWriter 返回配置的日志输出目标。
func (gbt *GBTree) logWriter() io.Writer {
	if gbt.Config.LogWriter != nil {
		return gbt.Config.LogWriter
	}
	return nil // stdout
}

// logPrintf 根据日志级别打印消息。
func (gbt *GBTree) logPrintf(level int, format string, args ...interface{}) {
	if gbt.Config.Verbosity < level {
		return
	}
	w := gbt.logWriter()
	msg := fmt.Sprintf(format, args...)
	if w != nil {
		fmt.Fprint(w, msg)
	} else {
		fmt.Print(msg)
	}
}

// BoostFromWeights 从基础分数初始化预测值。
func (gbt *GBTree) BoostFromWeights(dm *DMatrix) []float64 {
	n := dm.NumRows
	total := n
	if gbt.Config.NumClass > 1 {
		total = n * gbt.Config.NumClass
	}

	initScore := gbt.effectiveBaseScore()
	preds := make([]float64, total)
	for i := range preds {
		preds[i] = initScore
	}

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
// 返回每个评估指标的历史记录列表。
func (gbt *GBTree) Train(dm *DMatrix, metrics []Metric) ([]EvalResult, error) {
	if gbt.Config.CustomEval != nil {
		metrics = append(metrics, &CustomMetric{Fn: gbt.Config.CustomEval})
	}
	return gbt.trainLoop(dm, metrics, 0)
}

// ContinueTrain 向现有模型添加更多树。
// 保留现有模型的树，额外的提升轮次从当前预测状态继续。
func (gbt *GBTree) ContinueTrain(dm *DMatrix, additionalRounds int, metrics []Metric) ([]EvalResult, error) {
	prevTrees := gbt.Config.NumTrees
	gbt.Config.NumTrees = additionalRounds
	defer func() { gbt.Config.NumTrees = prevTrees + additionalRounds }()
	return gbt.trainLoop(dm, metrics, len(gbt.Trees))
}

func (gbt *GBTree) trainLoop(dm *DMatrix, metrics []Metric, startIter int) ([]EvalResult, error) {
	if dm == nil {
		return nil, ErrEmptyData
	}

	// 将标签裁剪为 {0, 1} 为二分类目标函数。
	if gbt.Config.Objective == ObjBinaryLogistic || gbt.Config.Objective == ObjBinaryLogitRaw {
		for i, v := range dm.Labels {
			if v > 0 {
				dm.Labels[i] = 1
			} else {
				dm.Labels[i] = 0
			}
		}
	}

	// 从数据自动估计 BaseScore（对应 XGBoost 的 boost_from_average）
	if gbt.Config.BoostFromAverage && startIter == 0 {
		meanLabel := 0.0
		for _, v := range dm.Labels {
			meanLabel += v
		}
		meanLabel /= float64(len(dm.Labels))

		switch gbt.Config.Objective {
		case ObjBinaryLogistic, ObjBinaryLogitRaw:
			if meanLabel > 0 && meanLabel < 1 {
				gbt.Config.BaseScore = meanLabel
			} else {
				gbt.Config.BaseScore = 0
			}
		case ObjRegSquareError, ObjRegLogistic:
			gbt.Config.BaseScore = meanLabel
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
				idx := i*gbt.Config.NumClass + classIdx
				preds[idx] += gbt.Config.LearningRate * lv
			} else {
				preds[i] += gbt.Config.LearningRate * lv
			}
		}
	}

	// 构建树构建器配置
	builderCfg := &ExactBuilderConfig{
		MaxDepth:               gbt.Config.MaxDepth,
		Gamma:                  gbt.Config.Gamma,
		Lambda:                 gbt.Config.Lambda,
		Alpha:                  gbt.Config.Alpha,
		MinChildWeight:         gbt.Config.MinChildWeight,
		MaxDeltaStep:           gbt.Config.MaxDeltaStep,
		MaxLeaves:              gbt.Config.MaxLeaves,
		NumFeatures:            dm.NumCols,
		MaxBin:                 gbt.Config.MaxBin,
		ColSampleByLevel:       gbt.Config.ColSampleByLevel,
		ColSampleByNode:        gbt.Config.ColSampleByNode,
		MaxCachedHistNode:      gbt.Config.MaxCachedHistNode,
		MaxCatToOneHot:         gbt.Config.MaxCatToOneHot,
		MaxCatThreshold:        gbt.Config.MaxCatThreshold,
		MonotoneConstraints:    gbt.Config.MonotoneConstraints,
		InteractionConstraints: gbt.Config.InteractionConstraints,
	}

	// 仅 hist 方法需要 bin mapper
	var binMapper *BinMapper
	if gbt.Config.TreeMethod == "hist" || gbt.Config.TreeMethod == "" {
		var initHess []float64
		if rnk, ok := gbt.Objective.(*RankNDCG); ok && len(dm.Group) > 0 {
			_, initHess, _ = rnk.GetGradientWithGroup(preds, dm.Labels, dm.Weights, dm.Group, 1.0)
		} else if rnk, ok := gbt.Objective.(*RankMAP); ok && len(dm.Group) > 0 {
			_, initHess, _ = rnk.GetGradientWithGroup(preds, dm.Labels, dm.Weights, dm.Group, 1.0)
		} else {
			_, initHess, _ = gbt.Objective.GetGradient(preds, dm.Labels, dm.Weights)
		}
		if gbt.Config.NumClass > 1 {
			classHess := make([]float64, dm.NumRows)
			for i := 0; i < dm.NumRows; i++ {
				classHess[i] = initHess[i*gbt.Config.NumClass]
			}
			initHess = classHess
		}

		maxBin := gbt.Config.MaxBin
		if maxBin <= 0 {
			maxBin = 256
		}
		binMapper = NewGKBinMapper(dm.Data, initHess, maxBin)
	}

	// 初始化验证集预测（用于早停）
	var evalPreds []float64
	if gbt.Config.EarlyStoppingRounds > 0 && gbt.Config.EvalData != nil {
		evalPreds = gbt.BoostFromWeights(gbt.Config.EvalData)
	}

	// 历史记录
	var history []EvalResult

	for iter := startIter; iter < totalTrees; iter++ {
		classIdx := 0
		if gbt.Config.NumClass > 1 {
			classIdx = iter % gbt.Config.NumClass
		}

		// DART：选择丢弃的树，并计算丢弃后预测值
		predsForGrad := preds
		var dropped []int
		var dartNorm float64 = 1.0
		if gbt.Config.BoostType == BoostDART && len(gbt.Trees) > 1 {
			dropped, dartNorm = gbt.selectDroppedTrees(iter)
			predsForGrad = make([]float64, len(preds))
			copy(predsForGrad, gbt.BoostFromWeights(dm))
			for ti, tree := range gbt.Trees {
				isDropped := false
				for _, d := range dropped {
					if ti == d {
						isDropped = true
						break
					}
				}
				if !isDropped {
					for i := 0; i < dm.NumRows; i++ {
						predsForGrad[i] += gbt.Config.LearningRate * tree.Predict(dm.Data[i])
					}
				}
			}
		}

		// 计算本轮梯度
		var grads, hess []float64
		var err error
		computedWithGroup := false
		if rnk, ok := gbt.Objective.(*RankNDCG); ok && len(dm.Group) > 0 {
			grads, hess, err = rnk.GetGradientWithGroup(predsForGrad, dm.Labels, dm.Weights, dm.Group, 1.0)
			computedWithGroup = true
		}
		if rnk, ok := gbt.Objective.(*RankMAP); ok && len(dm.Group) > 0 {
			grads, hess, err = rnk.GetGradientWithGroup(predsForGrad, dm.Labels, dm.Weights, dm.Group, 1.0)
			computedWithGroup = true
		}
		if !computedWithGroup {
			grads, hess, err = gbt.Objective.GetGradient(predsForGrad, dm.Labels, dm.Weights)
		}
		if err != nil {
			return history, fmt.Errorf("gradient computation at iter %d: %w", iter, err)
		}

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

		// 应用行子采样（支持 uniform 和 gradient_based）
		rowSeed := deriveSampleSeed(gbt.Config.Seed, iter, 0)
		var sampleIndices []int
		if gbt.Config.SamplingMethod == "gradient_based" && gbt.Config.Subsample < 1.0 {
			// 基于梯度幅度的采样：概率与 |grad| 成正比
			sampleIndices = mtRowSampleGradient(dm.NumRows, gbt.Config.Subsample, rowSeed, classGrads)
		} else {
			sampleIndices = mtRowSample(dm.NumRows, gbt.Config.Subsample, rowSeed)
		}

		// 应用列子采样（支持 feature_weights）
		colSeed := deriveSampleSeed(gbt.Config.Seed, iter, 1)
		var featureMask []int
		if len(gbt.Config.FeatureWeights) >= dm.NumCols {
			featureMask = mtColSampleWeighted(dm.NumCols, gbt.Config.ColSampleByTree, colSeed, gbt.Config.FeatureWeights)
		} else {
			featureMask = mtColSample(dm.NumCols, gbt.Config.ColSampleByTree, colSeed)
		}

		// 并行树（num_parallel_tree）
		numParallel := gbt.Config.NumParallelTree
		if numParallel <= 0 {
			numParallel = 1
		}

		lr := gbt.Config.LearningRate
		for pt := 0; pt < numParallel; pt++ {
			var treeSampleIndices []int
			var treeFeatureMask []int
			if numParallel > 1 {
				// Bootstrap 采样（用 mt19937 替换 LCG，与 XGBoost 一致）
				bootSeed := deriveSampleSeed(gbt.Config.Seed, iter, uint32(100+pt))
				treeSampleIndices = bootstrapSampleMT(dm.NumRows, bootSeed)
				// 每棵并行树独立列采样（XGBoost 随机森林行为）
				colSeed2 := deriveSampleSeed(gbt.Config.Seed, iter+pt*1000, uint32(101+pt))
				treeFeatureMask = mtColSample(dm.NumCols, gbt.Config.ColSampleByTree, colSeed2)
			} else {
				treeSampleIndices = sampleIndices
				treeFeatureMask = featureMask
			}

			var tree *RegTree
			switch gbt.Config.TreeMethod {
			case "exact":
				builder := NewExactBuilder(builderCfg)
				if err := builder.Build(dm, classGrads, classHess, treeSampleIndices, treeFeatureMask); err != nil {
					return history, fmt.Errorf("tree build at iter %d: %w", iter, err)
				}
				tree = builder.tree
			default:
				builder := NewHistBuilder(builderCfg, binMapper)
				builder.SetIterSeed(gbt.Config.Seed + int64(iter) + int64(pt)*100000)
				if err := builder.Build(dm, classGrads, classHess, treeSampleIndices, treeFeatureMask); err != nil {
					return history, fmt.Errorf("tree build at iter %d: %w", iter, err)
				}
				tree = builder.tree
			}

			// refresh_leaf：重新计算叶节点值（不改变树结构）
			if gbt.Config.RefreshLeaf && numParallel <= 1 {
				refreshLeafValues(tree, classGrads, classHess, builderCfg.Lambda, builderCfg.Alpha, builderCfg.MaxDeltaStep)
			}

			gbt.Trees = append(gbt.Trees, tree)
			gbt.TreeInfo = append(gbt.TreeInfo, classIdx)

			for i := 0; i < dm.NumRows; i++ {
				leafValue := tree.Predict(dm.Data[i])
				contribution := lr * leafValue * dartNorm / float64(numParallel)
				if gbt.Config.NumClass > 1 {
					base := i * gbt.Config.NumClass
					preds[base+classIdx] += contribution
				} else {
					preds[i] += contribution
				}
			}
		}

		// 收集训练指标历史
		evalResult := EvalResult{
			Iteration: iter,
			Metrics:   make(map[string]float64),
		}
		for _, m := range metrics {
			evalResult.Metrics[m.Name()] = m.Evaluate(preds, dm)
		}
		history = append(history, evalResult)

		// 早停检查
		if gbt.Config.EarlyStoppingRounds > 0 && gbt.Config.EvalData != nil && len(metrics) > 0 {
			for i := 0; i < gbt.Config.EvalData.NumRows; i++ {
				// 使用最后构建的树更新 evalPreds
				lastTree := gbt.Trees[len(gbt.Trees)-1]
				leafValue := lastTree.Predict(gbt.Config.EvalData.Data[i])
				evalPreds[i] += lr * leafValue
			}

			m := metrics[0]
			score := m.Evaluate(evalPreds, gbt.Config.EvalData)

			lowerBetter := true
			switch m.Name() {
			case "auc", "aucpr":
				lowerBetter = false
			}

			bestScore := gbt.bestEvalScore
			if iter == startIter || (lowerBetter && score < bestScore) || (!lowerBetter && score > bestScore) {
				gbt.bestEvalScore = score
				gbt.bestIteration = iter
			}

			if iter-gbt.bestIteration >= gbt.Config.EarlyStoppingRounds {
				nKeep := gbt.bestIteration - startIter + 1
				gbt.Trees = gbt.Trees[:nKeep]
				if gbt.bestIteration < iter {
					return history, nil
				}
			}
		}

		// 定期打印评估指标
		if gbt.Config.Verbosity >= 2 && iter%10 == 0 {
			score := fmt.Sprintf("[%d/%d]", iter+1, totalTrees)
			for _, m := range metrics {
				score += fmt.Sprintf(" %s=%.4f", m.Name(), m.Evaluate(preds, dm))
			}
			gbt.logPrintf(2, "%s\n", score)
		}

		// 回调函数
		for _, cb := range gbt.Config.Callbacks {
			if cb(iter, preds, dm) {
				return history, nil
			}
		}
	}

	return history, nil
}

// effectiveBaseScore 将 BaseScore（概率空间）转换为原始空间的初始预测值。
func (gbt *GBTree) effectiveBaseScore() float64 {
	bs := gbt.Config.BaseScore
	if gbt.Config.Objective == ObjBinaryLogistic || gbt.Config.Objective == ObjBinaryLogitRaw {
		if bs <= 0 || bs >= 1 {
			return 0.0
		}
		return math.Log(bs / (1 - bs))
	}
	return bs
}

// Predict 返回单个样本的原始预测值。
// iterationRange 可选参数指定树范围 [start, end)，为空时使用所有树。
func (gbt *GBTree) Predict(sample []float64, iterationRange ...int) float64 {
	score := gbt.effectiveBaseScore()
	start, end := 0, len(gbt.Trees)
	if len(iterationRange) > 0 {
		start = iterationRange[0]
	}
	if len(iterationRange) > 1 {
		end = iterationRange[1]
	}
	if start < 0 {
		start = 0
	}
	if end > len(gbt.Trees) {
		end = len(gbt.Trees)
	}
	for i := start; i < end; i++ {
		score += gbt.Config.LearningRate * gbt.Trees[i].Predict(sample)
	}
	return score
}

// PredictProb 返回二分类的概率值（0-1）。
// iterationRange 可选参数指定树范围。
func (gbt *GBTree) PredictProb(sample []float64, iterationRange ...int) float64 {
	return sigmoid(gbt.Predict(sample, iterationRange...))
}

// PredictBatch 返回多个样本的预测值。
// iterationRange 可选参数指定树范围。
func (gbt *GBTree) PredictBatch(data [][]float64, iterationRange ...int) []float64 {
	out := make([]float64, len(data))
	for i, row := range data {
		out[i] = gbt.Predict(row, iterationRange...)
	}
	return out
}

// GetLeafIndex 返回样本所在的叶节点索引。
// iterationRange 可选参数指定树范围。
func (gbt *GBTree) GetLeafIndex(sample []float64, iterationRange ...int) []int {
	start, end := 0, len(gbt.Trees)
	if len(iterationRange) > 0 {
		start = iterationRange[0]
	}
	if len(iterationRange) > 1 {
		end = iterationRange[1]
	}
	if start < 0 {
		start = 0
	}
	if end > len(gbt.Trees) {
		end = len(gbt.Trees)
	}
	leaves := make([]int, end-start)
	for i := start; i < end; i++ {
		leaves[i-start] = gbt.Trees[i].GetLeafIndex(sample)
	}
	return leaves
}

// selectDroppedTrees 为 DART 选择本轮要丢弃的树。
// 支持 sample_type（uniform/weighted）和 normalize_type（tree/forest）。
func (gbt *GBTree) selectDroppedTrees(iter int) ([]int, float64) {
	numTrees := len(gbt.Trees)
	if numTrees <= 1 {
		return nil, 1.0
	}

	seed := deriveSampleSeed(gbt.Config.Seed, iter, 100)
	rng := NewMT19937(seed)

	if gbt.Config.SkipDrop > 0 && rng.Uniform() < gbt.Config.SkipDrop {
		return nil, 1.0
	}

	sampleType := gbt.Config.SampleType
	if sampleType == "" {
		sampleType = "uniform"
	}

	dropped := make([]int, 0)

	if sampleType == "weighted" {
		// Weighted sampling: 使用叶子权重和作为采样概率
		weights := make([]float64, numTrees)
		totalWeight := 0.0
		for ti, tree := range gbt.Trees {
			for _, node := range tree.Nodes {
				if node.IsLeaf() {
					w := math.Abs(node.LeafValue)
					weights[ti] += w
				}
			}
			totalWeight += weights[ti]
		}
		if totalWeight > 0 {
			for ti := 0; ti < numTrees; ti++ {
				if gbt.Config.OneDrop && len(dropped) >= 1 {
					break
				}
				if rng.Uniform() < gbt.Config.RateDrop*(weights[ti]/totalWeight)*float64(numTrees) {
					dropped = append(dropped, ti)
				}
			}
		} else {
			// 回退到 uniform
			for ti := 0; ti < numTrees; ti++ {
				if gbt.Config.OneDrop && len(dropped) >= 1 {
					break
				}
				if rng.Uniform() < gbt.Config.RateDrop {
					dropped = append(dropped, ti)
				}
			}
		}
	} else {
		// Uniform sampling
		for ti := 0; ti < numTrees; ti++ {
			if gbt.Config.OneDrop && len(dropped) >= 1 {
				break
			}
			if rng.Uniform() < gbt.Config.RateDrop {
				dropped = append(dropped, ti)
			}
		}
	}

	if len(dropped) >= numTrees {
		dropped = dropped[:numTrees-1]
	}

	kept := numTrees - len(dropped)
	// normalize_type: "tree" (default) or "forest"
	normalizeType := gbt.Config.NormalizeType
	if normalizeType == "" {
		normalizeType = "tree"
	}

	var norm float64
	if normalizeType == "forest" {
		norm = 1.0 / (1.0 + gbt.Config.RateDrop)
	} else {
		norm = 1.0 / float64(kept+1)
	}

	return dropped, norm
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

// bootstrapSampleMT 使用 mt19937 生成有放回的行索引采样（随机森林）。
// 与 XGBoost C++ 的行为一致：对 n 个样本进行有放回采样，每次从 [0, n) 均匀抽取。
func bootstrapSampleMT(n int, seed uint32) []int {
	rng := NewMT19937(seed)
	indices := make([]int, n)
	for i := range indices {
		indices[i] = int(rng.Uniform() * float64(n))
		if indices[i] >= n {
			indices[i] = n - 1
		}
	}
	return indices
}

// bootstrapSample 使用 LCG 生成有放回的行索引采样（旧版，保留兼容）。
func bootstrapSample(n int, seed int64) []int {
	indices := make([]int, n)
	state := uint64(seed)
	for i := range indices {
		state = state*6364136223846793005 + 1442695040888963407
		idx := int((state >> 33) % uint64(n))
		if idx < 0 {
			idx = -idx % n
		}
		indices[i] = idx
	}
	return indices
}

// colSampleFromMask 从已有特征掩码中进行子采样。
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
