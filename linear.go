package xgb

// GBLinear 实现线性提升器（gblinear）。
//
// 对应 XGBoost C++ 的 src/gbm/gblinear.cc。
// 使用坐标下降法（shotgun/coord_descent）优化线性权重。
type GBLinear struct {
	Config    *Config
	Weights   []float64 // 特征权重（最后一个是偏置 bias）
	GradSum   []float64 // 每个坐标的梯度累积
	HessSum   []float64 // 每个坐标的 Hessian 累积
	Objective Objective
	Updater   string // "shotgun"（随机顺序）或 "coord_descent"（固定顺序）
}

// NewGBLinear 创建线性提升器。
func NewGBLinear(cfg *Config) *GBLinear {
	var objective Objective
	if cfg.CustomObjective != nil {
		objective = &CustomObj{ObjName: "custom", Fn: cfg.CustomObjective}
	} else {
		objective = NewObjective(cfg.Objective, cfg.NumClass, cfg.ScalePosWeight)
	}
	// 配置 objective 参数
	if ph, ok := objective.(*PseudoHuberError); ok {
		ph.Delta = cfg.HuberSlope
	}
	if qe, ok := objective.(*QuantileError); ok {
		qe.Alpha = cfg.QuantileAlpha
	}
	if tr, ok := objective.(*TweedieRegression); ok {
		tr.VariancePower = cfg.TweedieVariancePower
	}

	return &GBLinear{
		Config:    cfg,
		Objective: objective,
	}
}

// Train 训练线性模型（使用坐标下降法）。
func (gl *GBLinear) Train(dm *DMatrix, metrics []Metric) ([]EvalResult, error) {
	if gl.Config.CustomEval != nil {
		metrics = append(metrics, &CustomMetric{Fn: gl.Config.CustomEval})
	}

	n := dm.NumRows
	nFeat := dm.NumCols

	// 初始化权重（含偏置）
	gl.Weights = make([]float64, nFeat+1) // +1 for bias
	gl.GradSum = make([]float64, nFeat+1)
	gl.HessSum = make([]float64, nFeat+1)

	// 基础预测
	baseScore := gl.Config.BaseScore
	if gl.Config.BoostFromAverage {
		var mean float64
		for _, v := range dm.Labels {
			mean += v
		}
		mean /= float64(n)
		baseScore = mean
	}
	gl.Weights[nFeat] = baseScore // bias

	lr := gl.Config.LearningRate
	lambda := gl.Config.Lambda
	alpha := gl.Config.Alpha
	history := make([]EvalResult, 0, gl.Config.NumTrees)

	for iter := 0; iter < gl.Config.NumTrees; iter++ {
		// 当前预测值
		preds := make([]float64, n)
		for i := 0; i < n; i++ {
			preds[i] = gl.Predict(dm.Data[i])
		}

		// 计算梯度
		grads, hess, err := gl.Objective.GetGradient(preds, dm.Labels, dm.Weights)
		if err != nil {
			return history, err
		}

		// 清零累积
		for j := range gl.GradSum {
			gl.GradSum[j] = 0
			gl.HessSum[j] = 0
		}

		// 计算每个坐标的梯度/Hessian 和（带偏置）
		for i := 0; i < n; i++ {
			for j := 0; j < nFeat; j++ {
				gl.GradSum[j] += grads[i] * dm.Data[i][j]
				gl.HessSum[j] += hess[i] * dm.Data[i][j] * dm.Data[i][j]
			}
			gl.GradSum[nFeat] += grads[i]
			gl.HessSum[nFeat] += hess[i]
		}

		// 坐标下降更新（与 C++ ShotgunUpdater 一致）
		// C++ 顺序：1) 先更新 bias（偏置），2) 用 FeatureSelector 更新特征
		// 偏置更新（C++: CoordinateDeltaBias）
		if gl.HessSum[nFeat] != 0 {
			denom := gl.HessSum[nFeat] + lambda
			if denom != 0 {
				db := -gl.GradSum[nFeat] / denom
				if alpha > 0 {
					th := alpha / denom
					if db > th {
						db -= th
					} else if db < -th {
						db += th
					} else {
						db = 0
					}
				}
				gl.Weights[nFeat] += lr * db
			}
		}

		// 特征更新顺序（C++: FeatureSelector.NextFeature 决定顺序）
		featOrder := make([]int, nFeat)
		for j := range featOrder {
			featOrder[j] = j
		}
		if gl.Updater == "shotgun" || gl.Updater == "" {
			// Shotgun: 每轮随机打乱特征顺序（C++: feature_selector=kShuffle）
			dropSeed := deriveSampleSeed(int64(gl.Config.Seed), iter, 100)
			rng := NewMT19937(dropSeed)
			for i := len(featOrder) - 1; i > 0; i-- {
				j := int(rng.Uniform() * float64(i+1))
				if j > i {
					j = i
				}
				featOrder[i], featOrder[j] = featOrder[j], featOrder[i]
			}
		}
		// coord_descent: 保持固定顺序（C++: feature_selector=kCyclic）

		// 特征权重更新（C++: CoordinateDelta）
		for _, j := range featOrder {
			if gl.HessSum[j] == 0 {
				continue
			}
			denom := gl.HessSum[j] + lambda
			if denom == 0 {
				continue
			}
			w := -gl.GradSum[j] / denom
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
			gl.Weights[j] += lr * w
		}

		// 收集指标
		ev := EvalResult{
			Iteration: iter,
			Metrics:   make(map[string]float64),
		}
		finalPreds := make([]float64, n)
		for i := 0; i < n; i++ {
			finalPreds[i] = gl.Predict(dm.Data[i])
		}
		for _, m := range metrics {
			ev.Metrics[m.Name()] = m.Evaluate(finalPreds, dm)
		}
		history = append(history, ev)
	}

	return history, nil
}

// Predict 返回单个样本的线性预测值。
func (gl *GBLinear) Predict(sample []float64) float64 {
	score := gl.Weights[len(gl.Weights)-1] // bias
	for j, w := range gl.Weights[:len(gl.Weights)-1] {
		if j < len(sample) {
			score += w * sample[j]
		}
	}
	return score
}

// PredictBatch 返回多个样本的预测值。
func (gl *GBLinear) PredictBatch(data [][]float64) []float64 {
	out := make([]float64, len(data))
	for i, row := range data {
		out[i] = gl.Predict(row)
	}
	return out
}

// GetWeights 返回模型权重（前 n 个为特征权重，最后一个为偏置）。
func (gl *GBLinear) GetWeights() []float64 {
	return gl.Weights
}
