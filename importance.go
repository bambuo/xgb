package xgb

import (
	"math"
	"sort"
)

// ImportanceType 定义特征重要性计算方式。
type ImportanceType int

const (
	// ImportanceWeight 按分裂次数统计（默认）。
	// 对应 XGBoost 的 "weight"。
	ImportanceWeight ImportanceType = iota

	// ImportanceGain 按总增益统计。
	// 对应 XGBoost 的 "gain"。
	ImportanceGain

	// ImportanceCover 按总覆盖样本数（sum_hess）统计。
	// 对应 XGBoost 的 "cover"。
	ImportanceCover
)

// FeatureImportance 保存单个特征的重要性信息。
type FeatureImportance struct {
	FeatureIndex int
	Score        float64
	Rank         int
}

// GetScore 返回特征重要性评分。
// importanceType 指定计算方式：ImportanceWeight、ImportanceGain、ImportanceCover。
// 返回 map[featureIndex]score，特征索引从 0 开始。
func (gbt *GBTree) GetScore(importanceType ImportanceType) map[int]float64 {
	result := make(map[int]float64)
	for _, tree := range gbt.Trees {
		for _, node := range tree.Nodes {
			if node.IsLeaf() {
				continue
			}
			fidx := node.FeatureIndex
			switch importanceType {
			case ImportanceGain:
				result[fidx] += node.LossChange
			case ImportanceCover:
				result[fidx] += node.SumHess
			default: // ImportanceWeight
				result[fidx]++
			}
		}
	}
	return result
}

// GetFScore 返回按分裂次数统计的特征重要性。
// 等价于 GetScore(ImportanceWeight)。
func (gbt *GBTree) GetFScore() map[int]int {
	result := make(map[int]int)
	for _, tree := range gbt.Trees {
		for _, node := range tree.Nodes {
			if !node.IsLeaf() {
				result[node.FeatureIndex]++
			}
		}
	}
	return result
}

// ImportanceRanking 返回按分数降序排列的特征重要性列表。
// 可用于绘制条形图。
func (gbt *GBTree) ImportanceRanking(importanceType ImportanceType) []FeatureImportance {
	scores := gbt.GetScore(importanceType)
	ranking := make([]FeatureImportance, 0, len(scores))
	for fidx, score := range scores {
		ranking = append(ranking, FeatureImportance{
			FeatureIndex: fidx,
			Score:        score,
		})
	}
	sort.Slice(ranking, func(i, j int) bool {
		return ranking[i].Score > ranking[j].Score
	})
	for i := range ranking {
		ranking[i].Rank = i + 1
	}
	return ranking
}

// PDPPoint 保存偏依赖图上的一个点。
type PDPPoint struct {
	FeatureValue float64 // 目标特征值
	Prediction   float64 // 平均预测值
	LowerBound   float64 // 置信下界（2.5%）
	UpperBound   float64 // 置信上界（97.5%）
}

// PartialDependence 计算单个特征的偏依赖值。
// featureIdx 是目标特征索引。
// data 是用于计算的数据集（nil = 使用全部训练数据）。
// gridPoints 指定在特征值范围内均匀采样的点数。
//
// 返回 []PDPPoint，每个点包含特征值和对应的平均预测值。
func (gbt *GBTree) PartialDependence(featureIdx int, data [][]float64, gridPoints int) []PDPPoint {
	if data == nil || len(data) == 0 {
		return nil
	}
	if gridPoints <= 0 {
		gridPoints = 50
	}

	// 找到特征的最小/最大值
	minVal, maxVal := data[0][featureIdx], data[0][featureIdx]
	for _, row := range data {
		v := row[featureIdx]
		if math.IsNaN(v) {
			continue
		}
		if v < minVal {
			minVal = v
		}
		if v > maxVal {
			maxVal = v
		}
	}

	if maxVal <= minVal {
		// 常数特征，只返回一个点
		avg := gbt.computeAveragePrediction(data, featureIdx, minVal)
		return []PDPPoint{{FeatureValue: minVal, Prediction: avg}}
	}

	// 在特征值范围内均匀采样
	points := make([]PDPPoint, gridPoints)
	for i := 0; i < gridPoints; i++ {
		fv := minVal + (maxVal-minVal)*float64(i)/float64(gridPoints-1)
		avg := gbt.computeAveragePrediction(data, featureIdx, fv)

		// 计算预测值的标准差作为置信区间
		var sumSq float64
		var count int
		for _, row := range data {
			orig := row[featureIdx]
			row[featureIdx] = fv
			p := gbt.Predict(row)
			row[featureIdx] = orig
			d := p - avg
			sumSq += d * d
			count++
		}

		points[i] = PDPPoint{
			FeatureValue: fv,
			Prediction:   avg,
			LowerBound:   avg - 1.96*math.Sqrt(sumSq/float64(count))/math.Sqrt(float64(count)),
			UpperBound:   avg + 1.96*math.Sqrt(sumSq/float64(count))/math.Sqrt(float64(count)),
		}
	}

	return points
}

// LearningCurvePoint 保存学习曲线上的一个点。
type LearningCurvePoint struct {
	Iteration int
	Metrics   map[string]float64
}

// LearningCurve 从训练历史中提取学习曲线数据。
// history 是 Train() 返回的评估历史。
// metricName 指定要提取的指标名（为空时返回所有指标）。
//
// 返回的学习曲线可用于训练进度可视化。
func LearningCurve(history []EvalResult, metricName string) []LearningCurvePoint {
	points := make([]LearningCurvePoint, 0, len(history))
	for _, hr := range history {
		if metricName != "" {
			if v, ok := hr.Metrics[metricName]; ok {
				points = append(points, LearningCurvePoint{
					Iteration: hr.Iteration,
					Metrics:   map[string]float64{metricName: v},
				})
			}
		} else {
			m := make(map[string]float64)
			for k, v := range hr.Metrics {
				m[k] = v
			}
			points = append(points, LearningCurvePoint{
				Iteration: hr.Iteration,
				Metrics:   m,
			})
		}
	}
	return points
}

// BestIteration 从训练历史中找到指定指标的最佳迭代轮次。
// lowerBetter 指示该指标是否越低越好。
func BestIteration(history []EvalResult, metricName string, lowerBetter bool) (int, float64) {
	if len(history) == 0 {
		return -1, 0
	}
	bestIdx := 0
	bestVal := history[0].Metrics[metricName]
	for i := 1; i < len(history); i++ {
		v := history[i].Metrics[metricName]
		if lowerBetter && v < bestVal {
			bestVal = v
			bestIdx = i
		} else if !lowerBetter && v > bestVal {
			bestVal = v
			bestIdx = i
		}
	}
	return history[bestIdx].Iteration, bestVal
}

// computeAveragePrediction 将所有样本的特征 featureIdx 替换为 value 后的平均预测值。
func (gbt *GBTree) computeAveragePrediction(data [][]float64, featureIdx int, value float64) float64 {
	var sum float64
	for _, row := range data {
		orig := row[featureIdx]
		row[featureIdx] = value
		sum += gbt.Predict(row)
		row[featureIdx] = orig
	}
	return sum / float64(len(data))
}
