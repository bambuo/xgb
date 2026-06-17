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

// RMSLEMetric 计算均方根对数误差。
type RMSLEMetric struct{}

func (m *RMSLEMetric) Name() string { return "rmsle" }

func (m *RMSLEMetric) Evaluate(preds []float64, dm *DMatrix) float64 {
	var sumSq float64
	for i, y := range dm.Labels {
		lp := log1pSafe(preds[i])
		ll := log1pSafe(y)
		diff := lp - ll
		sumSq += diff * diff
	}
	return math.Sqrt(sumSq / float64(dm.NumRows))
}

// MAPEMetric 计算平均绝对百分比误差。
type MAPEMetric struct{}

func (m *MAPEMetric) Name() string { return "mape" }

func (m *MAPEMetric) Evaluate(preds []float64, dm *DMatrix) float64 {
	var sum float64
	eps := 1e-15
	for i, y := range dm.Labels {
		absY := math.Abs(y)
		if absY < eps {
			absY = eps
		}
		sum += math.Abs(preds[i]-y) / absY
	}
	return sum / float64(dm.NumRows) * 100.0
}

// MPHMetric 计算平均 Pseudo Huber 误差。
type MPHMetric struct {
	Delta float64
}

func (m *MPHMetric) Name() string { return "mphe" }

func (m *MPHMetric) Evaluate(preds []float64, dm *DMatrix) float64 {
	delta := m.Delta
	if delta <= 0 {
		delta = 1.0
	}
	var sum float64
	for i, y := range dm.Labels {
		diff := preds[i] - y
		sum += delta * delta * (math.Sqrt(1.0+(diff/delta)*(diff/delta)) - 1.0)
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

// ErrorMetric 计算二分类错误率。
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

// ErrorAtMetric 计算指定阈值的分类错误率。
type ErrorAtMetric struct {
	Threshold float64
}

func (m *ErrorAtMetric) Name() string { return "error@t" }

func (m *ErrorAtMetric) Evaluate(preds []float64, dm *DMatrix) float64 {
	var errors float64
	t := m.Threshold
	if t <= 0 {
		t = 0.5
	}
	for i, y := range dm.Labels {
		pred := 0.0
		if preds[i] >= t {
			pred = 1.0
		}
		if pred != y {
			errors++
		}
	}
	return errors / float64(dm.NumRows)
}

// MErrorMetric 计算多分类错误率。
type MErrorMetric struct{}

func (m *MErrorMetric) Name() string { return "merror" }

func (m *MErrorMetric) Evaluate(preds []float64, dm *DMatrix) float64 {
	n := dm.NumRows
	numClass := len(preds) / n
	var errors float64
	for i := 0; i < n; i++ {
		base := i * numClass
		maxIdx := 0
		maxVal := preds[base]
		for k := 1; k < numClass; k++ {
			if preds[base+k] > maxVal {
				maxVal = preds[base+k]
				maxIdx = k
			}
		}
		if float64(maxIdx) != dm.Labels[i] {
			errors++
		}
	}
	return errors / float64(n)
}

// MLogLossMetric 计算多分类对数损失。
type MLogLossMetric struct{}

func (m *MLogLossMetric) Name() string { return "mlogloss" }

func (m *MLogLossMetric) Evaluate(preds []float64, dm *DMatrix) float64 {
	n := dm.NumRows
	numClass := len(preds) / n
	var sum float64
	eps := 1e-15
	for i := 0; i < n; i++ {
		base := i * numClass
		label := int(dm.Labels[i])
		if label < 0 || label >= numClass {
			continue
		}
		p := preds[base+label]
		if p < eps {
			p = eps
		}
		if p > 1.0-eps {
			p = 1.0 - eps
		}
		sum += -math.Log(p)
	}
	return sum / float64(n)
}

// AUCMetric 计算 ROC 曲线下的面积。
type AUCMetric struct{}

func (m *AUCMetric) Name() string { return "auc" }

func (m *AUCMetric) Evaluate(preds []float64, dm *DMatrix) float64 {
	n := dm.NumRows
	if n < 2 {
		return 0
	}

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

	var sumRanks, nPos, nNeg float64
	for i := 0; i < n; i++ {
		if pairs[i].label == 1 {
			nPos++
			sumRanks += float64(i + 1)
		} else {
			nNeg++
		}
	}

	if nPos == 0 || nNeg == 0 {
		return 0.5
	}

	u := sumRanks - nPos*(nPos+1)/2.0
	auc := 1.0 - u/(nPos*nNeg)
	return auc
}

// AUCPRMetric 计算 Precision-Recall 曲线下的面积。
type AUCPRMetric struct{}

func (m *AUCPRMetric) Name() string { return "aucpr" }

func (m *AUCPRMetric) Evaluate(preds []float64, dm *DMatrix) float64 {
	n := dm.NumRows
	if n < 2 {
		return 0
	}

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

	var totalPos float64
	for _, p := range pairs {
		if p.label == 1 {
			totalPos++
		}
	}
	if totalPos == 0 {
		return 0
	}

	var cumPos, cumTotal float64
	var ap float64
	prevPrec := 1.0
	for _, p := range pairs {
		cumTotal++
		if p.label == 1 {
			cumPos++
			prec := cumPos / cumTotal
			if prec > prevPrec {
				prec = prevPrec
			}
			ap += prec
			prevPrec = prec
		}
	}
	return ap / totalPos
}

// NDCGMetric 计算 NDCG@k。
type NDCGMetric struct {
	K int
}

func (m *NDCGMetric) Name() string { return "ndcg" }

func (m *NDCGMetric) Evaluate(preds []float64, dm *DMatrix) float64 {
	n := dm.NumRows
	if n < 2 {
		return 1.0
	}

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

	k := m.K
	if k <= 0 || k > n {
		k = n
	}

	var dcg float64
	for i := 0; i < k; i++ {
		rel := math.Pow(2, pairs[i].label) - 1
		dcg += rel / math.Log2(float64(i+2))
	}

	sortedLabels := make([]float64, n)
	copy(sortedLabels, dm.Labels)
	sort.Slice(sortedLabels, func(i, j int) bool {
		return sortedLabels[i] > sortedLabels[j]
	})
	var idcg float64
	for i := 0; i < k; i++ {
		rel := math.Pow(2, sortedLabels[i]) - 1
		idcg += rel / math.Log2(float64(i+2))
	}

	if idcg <= 0 {
		return 0
	}
	return dcg / idcg
}

// MAPMetric 计算 Mean Average Precision。
type MAPMetric struct {
	K int
}

func (m *MAPMetric) Name() string { return "map" }

func (m *MAPMetric) Evaluate(preds []float64, dm *DMatrix) float64 {
	n := dm.NumRows
	if n < 2 {
		return 0
	}

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

	k := m.K
	if k <= 0 || k > n {
		k = n
	}

	var totalPos float64
	for _, p := range pairs {
		if p.label > 0 {
			totalPos++
		}
	}
	if totalPos == 0 {
		return 0
	}

	var cumPos, ap float64
	for i := 0; i < k; i++ {
		if pairs[i].label > 0 {
			cumPos++
			ap += cumPos / float64(i+1)
		}
	}
	return ap / math.Min(totalPos, float64(k))
}

// PrecisionMetric 计算 Precision@k。
type PrecisionMetric struct {
	K int
}

func (m *PrecisionMetric) Name() string { return "pre" }

func (m *PrecisionMetric) Evaluate(preds []float64, dm *DMatrix) float64 {
	n := dm.NumRows
	if n < 2 {
		return 0
	}

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

	k := m.K
	if k <= 0 || k > n {
		k = n
	}

	var relevant float64
	for i := 0; i < k; i++ {
		if pairs[i].label > 0 {
			relevant++
		}
	}
	return relevant / float64(k)
}

// PoissonNLogLikMetric 计算 Poisson 负对数似然。
type PoissonNLogLikMetric struct{}

func (m *PoissonNLogLikMetric) Name() string { return "poisson-nloglik" }

func (m *PoissonNLogLikMetric) Evaluate(preds []float64, dm *DMatrix) float64 {
	var sum float64
	eps := 1e-15
	for i, y := range dm.Labels {
		p := preds[i]
		if p < eps {
			p = eps
		}
		// Stirling 近似处理 log(label!)
		logFact := 0.0
		if y > 0 {
			logFact = y*math.Log(y) - y + 0.5*math.Log(2*math.Pi*y)
		}
		sum += p - y*math.Log(p) + logFact
	}
	return sum / float64(dm.NumRows)
}

// GammaNLogLikMetric 计算 Gamma 负对数似然。
type GammaNLogLikMetric struct{}

func (m *GammaNLogLikMetric) Name() string { return "gamma-nloglik" }

func (m *GammaNLogLikMetric) Evaluate(preds []float64, dm *DMatrix) float64 {
	var sum float64
	eps := 1e-15
	for i, y := range dm.Labels {
		p := preds[i]
		if p < eps {
			p = eps
		}
		if y < eps {
			y = eps
		}
		sum += math.Log(p) + y/p
	}
	return sum / float64(dm.NumRows)
}

// GammaDevianceMetric 计算 Gamma 偏差。
type GammaDevianceMetric struct{}

func (m *GammaDevianceMetric) Name() string { return "gamma-deviance" }

func (m *GammaDevianceMetric) Evaluate(preds []float64, dm *DMatrix) float64 {
	var sum float64
	eps := 1e-15
	for i, y := range dm.Labels {
		p := preds[i]
		if p < eps {
			p = eps
		}
		if y < eps {
			y = eps
		}
		sum += 2.0 * (math.Log(p/y) + (y-p)/p)
	}
	return sum / float64(dm.NumRows)
}

// CoxNLogLikMetric 计算 Cox 负偏对数似然。
type CoxNLogLikMetric struct{}

func (m *CoxNLogLikMetric) Name() string { return "cox-nloglik" }

func (m *CoxNLogLikMetric) Evaluate(preds []float64, dm *DMatrix) float64 {
	n := dm.NumRows
	idx := make([]int, n)
	for i := range idx {
		idx[i] = i
	}
	sort.Slice(idx, func(i, j int) bool {
		return preds[idx[i]] > preds[idx[j]]
	})

	var logLik float64
	var cumExp float64
	for i := n - 1; i >= 0; i-- {
		j := idx[i]
		expPred := math.Exp(preds[j])
		cumExp += expPred
		if dm.Labels[j] > 0 {
			logLik += preds[j] - math.Log(cumExp)
		}
	}
	return -logLik / float64(n)
}

// TweedieNLogLikMetric 计算 Tweedie 负对数似然。
type TweedieNLogLikMetric struct {
	VariancePower float64
}

func (m *TweedieNLogLikMetric) Name() string { return "tweedie-nloglik" }

func (m *TweedieNLogLikMetric) Evaluate(preds []float64, dm *DMatrix) float64 {
	p := m.VariancePower
	if p <= 1 || p >= 2 {
		p = 1.5
	}
	var sum float64
	eps := 1e-15
	for i, y := range dm.Labels {
		pred := preds[i]
		if pred < eps {
			pred = eps
		}
		if y < eps {
			y = eps
		}
		term1 := -y * math.Pow(pred, 1-p) / (1 - p)
		term2 := math.Pow(pred, 2-p) / (2 - p)
		sum += term1 + term2
	}
	return -sum / float64(dm.NumRows)
}

// ParseMetric 根据名称创建指标。
func ParseMetric(name string) (Metric, bool) {
	switch name {
	case "rmse":
		return &RMSEMetric{}, true
	case "mae":
		return &MAEMetric{}, true
	case "rmsle":
		return &RMSLEMetric{}, true
	case "mape":
		return &MAPEMetric{}, true
	case "mphe":
		return &MPHMetric{Delta: 1.0}, true
	case "logloss":
		return &LogLossMetric{}, true
	case "error":
		return &ErrorMetric{}, true
	case "error@t":
		return &ErrorAtMetric{Threshold: 0.5}, true
	case "merror":
		return &MErrorMetric{}, true
	case "mlogloss":
		return &MLogLossMetric{}, true
	case "auc":
		return &AUCMetric{}, true
	case "aucpr":
		return &AUCPRMetric{}, true
	case "ndcg":
		return &NDCGMetric{K: -1}, true
	case "map":
		return &MAPMetric{K: -1}, true
	case "pre":
		return &PrecisionMetric{K: -1}, true
	case "poisson-nloglik":
		return &PoissonNLogLikMetric{}, true
	case "gamma-nloglik":
		return &GammaNLogLikMetric{}, true
	case "gamma-deviance":
		return &GammaDevianceMetric{}, true
	case "tweedie-nloglik":
		return &TweedieNLogLikMetric{VariancePower: 1.5}, true
	case "cox-nloglik":
		return &CoxNLogLikMetric{}, true
	default:
		return nil, false
	}
}
