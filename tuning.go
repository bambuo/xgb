package xgb

import (
	"fmt"
	"math"
	"math/rand"
)

// GridSearchParam 定义单个参数的一组候选值。
type GridSearchParam struct {
	Name   string
	Values []interface{}
}

// GridSearchConfig 定义网格搜索的参数空间。
type GridSearchConfig struct {
	Params  []GridSearchParam
	Metrics []Metric
	NFolds  int
	Shuffle bool
	Seed    int64
}

// GridSearchResult 保存网格搜索的一组参数配置及其评估结果。
type GridSearchResult struct {
	Params  map[string]interface{}
	Scores  map[string]float64 // 指标名 → 验证集均值
	StdDevs map[string]float64 // 指标名 → 验证集标准差
}

// GridSearch 运行超参数网格搜索。
//
// baseCfg 是基础配置（固定参数），params 定义要搜索的参数空间。
// 返回按第一个指标排序的结果列表（最优在前）。
func GridSearch(baseCfg *Config, dm *DMatrix, gc GridSearchConfig) ([]GridSearchResult, error) {
	// 生成所有参数组合
	combinations := generateCombinations(gc.Params)
	var results []GridSearchResult

	for _, combo := range combinations {
		cfg := copyConfig(baseCfg)
		applyParams(cfg, combo)

		cvResults, err := CV(cfg, dm, gc.Metrics, gc.NFolds, gc.Shuffle, gc.Seed)
		if err != nil {
			return nil, fmt.Errorf("CV with params %v: %w", combo, err)
		}

		result := GridSearchResult{
			Params:  make(map[string]interface{}),
			Scores:  make(map[string]float64),
			StdDevs: make(map[string]float64),
		}
		for _, p := range combo {
			result.Params[p.Name] = p.Value
		}

		// 聚合所有 fold 的指标
		for _, m := range gc.Metrics {
			name := m.Name()
			var vals []float64
			for _, cvr := range cvResults {
				if v, ok := cvr.TestEval[name]; ok {
					vals = append(vals, v)
				}
			}
			if len(vals) > 0 {
				mean, std := meanStd(vals)
				result.Scores[name] = mean
				result.StdDevs[name] = std
			}
		}

		results = append(results, result)
	}

	// 按第一个指标排序（如果指标数量 > 0）
	if len(gc.Metrics) > 0 {
		primaryMetric := gc.Metrics[0].Name()
		// 判断是否为高优指标（auc, aucpr）
		higherBetter := primaryMetric == "auc" || primaryMetric == "aucpr"
		sortResults(results, primaryMetric, higherBetter)
	}

	return results, nil
}

// RandomSearch 运行超参数随机搜索。
// nIter 是随机采样的参数配置数。
func RandomSearch(baseCfg *Config, dm *DMatrix, gc GridSearchConfig, nIter int) ([]GridSearchResult, error) {
	rng := rand.New(rand.NewSource(gc.Seed))
	var results []GridSearchResult

	for i := 0; i < nIter; i++ {
		// 从每个参数中随机选择一个值
		combo := make([]paramCombination, len(gc.Params))
		for j, p := range gc.Params {
			if len(p.Values) == 0 {
				continue
			}
			idx := rng.Intn(len(p.Values))
			combo[j] = paramCombination{Name: p.Name, Value: p.Values[idx]}
		}

		cfg := copyConfig(baseCfg)
		applyParams(cfg, combo)

		cvResults, err := CV(cfg, dm, gc.Metrics, gc.NFolds, gc.Shuffle, gc.Seed+int64(i))
		if err != nil {
			return nil, fmt.Errorf("CV iter %d: %w", i, err)
		}

		result := GridSearchResult{
			Params:  make(map[string]interface{}),
			Scores:  make(map[string]float64),
			StdDevs: make(map[string]float64),
		}
		for _, p := range combo {
			result.Params[p.Name] = p.Value
		}

		for _, m := range gc.Metrics {
			name := m.Name()
			var vals []float64
			for _, cvr := range cvResults {
				if v, ok := cvr.TestEval[name]; ok {
					vals = append(vals, v)
				}
			}
			if len(vals) > 0 {
				mean, std := meanStd(vals)
				result.Scores[name] = mean
				result.StdDevs[name] = std
			}
		}

		results = append(results, result)
	}

	if len(gc.Metrics) > 0 {
		primaryMetric := gc.Metrics[0].Name()
		higherBetter := primaryMetric == "auc" || primaryMetric == "aucpr"
		sortResults(results, primaryMetric, higherBetter)
	}

	return results, nil
}

// ── 辅助函数 ──────────────────────────────────────────────────

// paramCombination 保存单个参数配置。
type paramCombination struct {
	Name  string
	Value interface{}
}

func generateCombinations(params []GridSearchParam) [][]paramCombination {
	if len(params) == 0 {
		return [][]paramCombination{{}}
	}

	first := params[0]
	rest := generateCombinations(params[1:])

	var result [][]paramCombination
	for _, v := range first.Values {
		for _, r := range rest {
			combo := make([]paramCombination, 0, len(r)+1)
			combo = append(combo, paramCombination{Name: first.Name, Value: v})
			combo = append(combo, r...)
			result = append(result, combo)
		}
	}
	return result
}

func applyParams(cfg *Config, params []paramCombination) {
	for _, p := range params {
		switch p.Name {
		case "learning_rate", "eta":
			if v, ok := p.Value.(float64); ok {
				cfg.LearningRate = v
			}
		case "max_depth":
			if v, ok := p.Value.(int); ok {
				cfg.MaxDepth = v
			}
		case "num_trees", "n_estimators":
			if v, ok := p.Value.(int); ok {
				cfg.NumTrees = v
			}
		case "subsample":
			if v, ok := p.Value.(float64); ok {
				cfg.Subsample = v
			}
		case "colsample_bytree":
			if v, ok := p.Value.(float64); ok {
				cfg.ColSampleByTree = v
			}
		case "colsample_bylevel":
			if v, ok := p.Value.(float64); ok {
				cfg.ColSampleByLevel = v
			}
		case "colsample_bynode":
			if v, ok := p.Value.(float64); ok {
				cfg.ColSampleByNode = v
			}
		case "min_child_weight":
			if v, ok := p.Value.(float64); ok {
				cfg.MinChildWeight = v
			}
		case "gamma":
			if v, ok := p.Value.(float64); ok {
				cfg.Gamma = v
			}
		case "lambda", "reg_lambda":
			if v, ok := p.Value.(float64); ok {
				cfg.Lambda = v
			}
		case "alpha", "reg_alpha":
			if v, ok := p.Value.(float64); ok {
				cfg.Alpha = v
			}
		case "max_delta_step":
			if v, ok := p.Value.(float64); ok {
				cfg.MaxDeltaStep = v
			}
		case "max_bin":
			if v, ok := p.Value.(int); ok {
				cfg.MaxBin = v
			}
		case "scale_pos_weight":
			if v, ok := p.Value.(float64); ok {
				cfg.ScalePosWeight = v
			}
		}
	}
}

func copyConfig(cfg *Config) *Config {
	// 创建一个新 Config，复制所有值
	cp := *cfg
	// 复制 map 和 slice
	if cfg.MonotoneConstraints != nil {
		cp.MonotoneConstraints = make(map[int]int)
		for k, v := range cfg.MonotoneConstraints {
			cp.MonotoneConstraints[k] = v
		}
	}
	if cfg.InteractionConstraints != nil {
		cp.InteractionConstraints = make([][]int, len(cfg.InteractionConstraints))
		for i, g := range cfg.InteractionConstraints {
			cp.InteractionConstraints[i] = make([]int, len(g))
			copy(cp.InteractionConstraints[i], g)
		}
	}
	if cfg.FeatureWeights != nil {
		cp.FeatureWeights = make([]float64, len(cfg.FeatureWeights))
		copy(cp.FeatureWeights, cfg.FeatureWeights)
	}
	cp.Callbacks = nil // 不复制回调
	cp.CustomObjective = nil
	cp.CustomEval = nil
	cp.EvalData = nil
	return &cp
}

func meanStd(vals []float64) (float64, float64) {
	n := float64(len(vals))
	if n == 0 {
		return 0, 0
	}
	var mean float64
	for _, v := range vals {
		mean += v
	}
	mean /= n
	var std float64
	for _, v := range vals {
		d := v - mean
		std += d * d
	}
	std = math.Sqrt(std / n)
	return mean, std
}

func sortResults(results []GridSearchResult, metric string, higherBetter bool) {
	for i := 1; i < len(results); i++ {
		for j := i; j > 0; j-- {
			a := results[j-1].Scores[metric]
			b := results[j].Scores[metric]
			swap := false
			if higherBetter {
				swap = b > a
			} else {
				swap = b < a
			}
			if swap {
				results[j], results[j-1] = results[j-1], results[j]
			}
		}
	}
}
