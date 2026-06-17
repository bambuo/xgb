package xgb

import (
	"math"
	"sort"
)

// Metric 是用于监控训练进度的评估指标。
type Metric interface {
	// Evaluate returns the metric value given predictions and data.
	Evaluate(preds []float64, dm *DMatrix) float64

	// Name returns the metric name (e.g., "rmse", "logloss").
	Name() string
}

// RMSEMetric 计算均方根误差。
type RMSEMetric struct{}

func (m *RMSEMetric) Name() string { return "rmse" }

func (m *RMSEMetric) Evaluate(preds []float64, dm *DMatrix) float64 {
	var sumSq float64
	for i, y := range dm.Labels {
		diff := preds[i] - y
		sumSq += diff * diff
	}
	return math.Sqrt(sumSq / float64(dm.NumRows))
}

// MAEMetric 计算平均绝对误差。
type MAEMetric struct{}

func (m *MAEMetric) Name() string { return "mae" }

func (m *MAEMetric) Evaluate(preds []float64, dm *DMatrix) float64 {
	var sum float64
	for i, y := range dm.Labels {
		sum += math.Abs(preds[i] - y)
	}
	return sum / float64(dm.NumRows)
}

// LogLossMetric 计算二分类对数损失。
type LogLossMetric struct{}

func (m *LogLossMetric) Name() string { return "logloss" }

func (m *LogLossMetric) Evaluate(preds []float64, dm *DMatrix) float64 {
	var sum float64
	eps := 1e-15
	for i, y := range dm.Labels {
		p := preds[i]
		// 裁剪概率值
		if p < eps {
			p = eps
		}
		if p > 1.0-eps {
			p = 1.0 - eps
		}
		sum += -y*math.Log(p) - (1.0-y)*math.Log(1.0-p)
	}
	return sum / float64(dm.NumRows)
}

// ErrorMetric 计算分类错误率。
type ErrorMetric struct{}

func (m *ErrorMetric) Name() string { return "error" }

func (m *ErrorMetric) Evaluate(preds []float64, dm *DMatrix) float64 {
	var errors float64
	for i, y := range dm.Labels {
		pred := 0.0
		if preds[i] >= 0.5 {
			pred = 1.0
		}
		if pred != y {
			errors++
		}
	}
	return errors / float64(dm.NumRows)
}

// AUCMetric 计算 ROC 曲线下的面积。
type AUCMetric struct{}

func (m *AUCMetric) Name() string { return "auc" }

func (m *AUCMetric) Evaluate(preds []float64, dm *DMatrix) float64 {
	n := dm.NumRows
	if n < 2 {
		return 0
	}

	// 创建 (pred, label) 对并按预测值降序排序
	type pair struct {
		pred  float64
		label float64
	}
	pairs := make([]pair, n)
	for i := 0; i < n; i++ {
		pairs[i] = pair{pred: preds[i], label: dm.Labels[i]}
	}
	sort.Slice(pairs, func(i, j int) bool {
		return pairs[i].pred > pairs[j].pred
	})

	// 使用 Wilcoxon-Mann-Whitney 方法计算 AUC
	var sumRanks, nPos, nNeg float64
	for i := 0; i < n; i++ {
		if pairs[i].label == 1 {
			nPos++
			sumRanks += float64(i + 1) // 1-基索引的排名
		} else {
			nNeg++
		}
	}

	if nPos == 0 || nNeg == 0 {
		return 0.5 // 随机猜测
	}

	// 正类的 U 统计量
	u := sumRanks - nPos*(nPos+1)/2.0
	// AUC = 1 - U/(nPos*nNeg) 或等价地：
	auc := 1.0 - u/(nPos*nNeg)
	return auc
}

// ParseMetric 根据名称创建指标。
func ParseMetric(name string) (Metric, bool) {
	switch name {
	case "rmse":
		return &RMSEMetric{}, true
	case "mae":
		return &MAEMetric{}, true
	case "logloss":
		return &LogLossMetric{}, true
	case "error":
		return &ErrorMetric{}, true
	case "auc":
		return &AUCMetric{}, true
	default:
		return nil, false
	}
}
