package xgb

import (
	"math"
)

// Objective 计算每个预测的梯度和 Hessian。
//
// 对应 XGBoost C++ 的 src/objective/。
type Objective interface {
	// GetGradient 计算一阶和二阶梯度。
	// pred：模型原始预测值（变换前）
	// labels：真实标签
	// weights：样本权重（nil = 均匀权重）
	GetGradient(pred, labels, weights []float64) (grad, hess []float64, err error)

	// PredTransform 将原始预测值转换为最终输出。
	PredTransform(pred []float64) []float64

	// Name 返回目标函数名称。
	Name() string
}

// SquaredError 是回归目标函数：reg:squarederror。
type SquaredError struct{}

func (o *SquaredError) Name() string { return "reg:squarederror" }

func (o *SquaredError) GetGradient(pred, labels, weights []float64) ([]float64, []float64, error) {
	n := len(pred)
	grad := make([]float64, n)
	hess := make([]float64, n)

	for i := 0; i < n; i++ {
		diff := pred[i] - labels[i]
		grad[i] = diff // g = y∧ - y
		hess[i] = 1.0  // h = 1

		if weights != nil {
			grad[i] *= weights[i]
			hess[i] *= weights[i]
		}
	}

	return grad, hess, nil
}

func (o *SquaredError) PredTransform(pred []float64) []float64 {
	// 回归无需变换
	out := make([]float64, len(pred))
	copy(out, pred)
	return out
}

// ── Poisson 回归 ────────────────────────────────────────────
//
// 对应 XGBoost 的 count:poisson 目标函数。
// 使用 log 链接函数：pred = exp(linear_pred)
// 梯度：g = pred - label
// Hessian：h = pred

type PoissonRegression struct{}

func (o *PoissonRegression) Name() string { return "count:poisson" }

func (o *PoissonRegression) GetGradient(pred, labels, weights []float64) ([]float64, []float64, error) {
	n := len(pred)
	grad := make([]float64, n)
	hess := make([]float64, n)
	for i := 0; i < n; i++ {
		p := math.Exp(pred[i]) // log 链接：pred = exp(raw)
		grad[i] = p - labels[i]
		hess[i] = p
		if weights != nil {
			grad[i] *= weights[i]
			hess[i] *= weights[i]
		}
	}
	return grad, hess, nil
}

func (o *PoissonRegression) PredTransform(pred []float64) []float64 {
	out := make([]float64, len(pred))
	for i, v := range pred {
		out[i] = math.Exp(v)
	}
	return out
}

// ── Gamma 回归 ──────────────────────────────────────────────
//
// 对应 XGBoost 的 reg:gamma 目标函数。
// 使用 log 链接函数：pred = exp(linear_pred)
// 梯度：g = 1 - label / pred
// Hessian：h = label / pred²

type GammaRegression struct{}

func (o *GammaRegression) Name() string { return "reg:gamma" }

func (o *GammaRegression) GetGradient(pred, labels, weights []float64) ([]float64, []float64, error) {
	n := len(pred)
	grad := make([]float64, n)
	hess := make([]float64, n)
	for i := 0; i < n; i++ {
		p := math.Exp(pred[i])
		grad[i] = 1.0 - labels[i]/p
		hess[i] = labels[i] / (p * p)
		if weights != nil {
			grad[i] *= weights[i]
			hess[i] *= weights[i]
		}
	}
	return grad, hess, nil
}

func (o *GammaRegression) PredTransform(pred []float64) []float64 {
	out := make([]float64, len(pred))
	for i, v := range pred {
		out[i] = math.Exp(v)
	}
	return out
}

// ── Tweedie 回归 ─────────────────────────────────────────────
//
// 对应 XGBoost 的 reg:tweedie 目标函数。
// pred = exp(linear_pred)
// 梯度/海森矩阵涉及方差幂参数 p（由 Config.TweedieVariancePower 控制）。

type TweedieRegression struct {
	VariancePower float64
}

func (o *TweedieRegression) Name() string { return "reg:tweedie" }

func (o *TweedieRegression) GetGradient(pred, labels, weights []float64) ([]float64, []float64, error) {
	n := len(pred)
	grad := make([]float64, n)
	hess := make([]float64, n)
	p := o.VariancePower
	for i := 0; i < n; i++ {
		predExp := math.Exp(pred[i])
		if predExp < 1e-6 {
			predExp = 1e-6
		}
		var rho float64
		if labels[i] > 0 {
			rho = labels[i] / predExp
		}
		grad[i] = predExp*math.Pow(rho, 2-p) - math.Pow(rho, 1-p)
		hess[i] = -predExp*math.Pow(rho, 2-p) +
			(2-p)*predExp*math.Pow(rho, 1-p) +
			(p-1)*math.Pow(rho, -p)
		if weights != nil {
			grad[i] *= weights[i]
			hess[i] *= weights[i]
		}
	}
	return grad, hess, nil
}

func (o *TweedieRegression) PredTransform(pred []float64) []float64 {
	out := make([]float64, len(pred))
	for i, v := range pred {
		out[i] = math.Exp(v)
	}
	return out
}

// LogisticRegression 是目标函数：reg:logistic, binary:logistic
type LogisticRegression struct {
	ScalePosWeight float64
}

func (o *LogisticRegression) Name() string { return "binary:logistic" }

func (o *LogisticRegression) GetGradient(pred, labels, weights []float64) ([]float64, []float64, error) {
	n := len(pred)
	grad := make([]float64, n)
	hess := make([]float64, n)

	for i := 0; i < n; i++ {
		p := sigmoid(pred[i])   // p = σ(y∧)
		grad[i] = p - labels[i] // g = p - y
		hess[i] = p * (1.0 - p) // h = p·(1-p)

		// scale_pos_weight：正类样本权重
		if labels[i] > 0 && o.ScalePosWeight != 1.0 {
			grad[i] *= o.ScalePosWeight
			hess[i] *= o.ScalePosWeight
		}

		if weights != nil {
			grad[i] *= weights[i]
			hess[i] *= weights[i]
		}
	}

	return grad, hess, nil
}

func (o *LogisticRegression) PredTransform(pred []float64) []float64 {
	out := make([]float64, len(pred))
	for i, v := range pred {
		out[i] = sigmoid(v)
	}
	return out
}

// LogisticRaw 是目标函数：binary:logitraw
type LogisticRaw struct {
	ScalePosWeight float64
}

func (o *LogisticRaw) Name() string { return "binary:logitraw" }

func (o *LogisticRaw) GetGradient(pred, labels, weights []float64) ([]float64, []float64, error) {
	n := len(pred)
	grad := make([]float64, n)
	hess := make([]float64, n)

	for i := 0; i < n; i++ {
		// 梯度公式与 logistic 相同，但输出为原始值
		p := sigmoid(pred[i])
		grad[i] = p - labels[i]
		hess[i] = p * (1.0 - p)

		// scale_pos_weight：正类样本权重
		if labels[i] > 0 && o.ScalePosWeight != 1.0 {
			grad[i] *= o.ScalePosWeight
			hess[i] *= o.ScalePosWeight
		}

		if weights != nil {
			grad[i] *= weights[i]
			hess[i] *= weights[i]
		}
	}

	return grad, hess, nil
}

func (o *LogisticRaw) PredTransform(pred []float64) []float64 {
	out := make([]float64, len(pred))
	copy(out, pred) // logitraw 无需变换
	return out
}

// MulticlassSoftmax 是目标函数：multi:softmax, multi:softprob
type MulticlassSoftmax struct {
	NumClass int
}

func (o *MulticlassSoftmax) Name() string { return "multi:softmax" }

func (o *MulticlassSoftmax) GetGradient(pred, labels, weights []float64) ([]float64, []float64, error) {
	nSamples := len(labels)
	grad := make([]float64, len(pred))
	hess := make([]float64, len(pred))

	for i := 0; i < nSamples; i++ {
		base := i * o.NumClass
		// 该样本的 Softmax 概率
		ps := softmax(pred[base : base+o.NumClass])

		// 启发式：使用 Hessian 近似的多分类重新归一化
		for k := 0; k < o.NumClass; k++ {
			pk := ps[k]
			target := 0.0
			if k == int(labels[i]) {
				target = 1.0
			}

			grad[base+k] = pk - target
			hess[base+k] = pk * (1.0 - pk)

			if weights != nil {
				grad[base+k] *= weights[i]
				hess[base+k] *= weights[i]
			}
		}
	}

	return grad, hess, nil
}

func (o *MulticlassSoftmax) PredTransform(pred []float64) []float64 {
	out := make([]float64, len(pred))
	nSamples := len(pred) / o.NumClass

	for i := 0; i < nSamples; i++ {
		base := i * o.NumClass
		ps := softmax(pred[base : base+o.NumClass])
		copy(out[base:base+o.NumClass], ps)
	}
	return out
}

// ── LambdaRank NDCG ─────────────────────────────────────────
//
// 对应 XGBoost 的 rank:ndcg 目标函数。
// 需要 DMatrix 中设置 Group 信息。

// computeDCG 计算排序列表的 DCG（Discounted Cumulative Gain）。
// expGain=true 时使用指数增益 2^rel-1，否则使用线性增益 rel。
func computeDCG(labels []float64, offset int, expGain bool) float64 {
	var sum float64
	for i, l := range labels {
		var rel float64
		if expGain {
			rel = math.Pow(2, l) - 1
		} else {
			rel = l
		}
		sum += rel / math.Log2(float64(offset+i+2))
	}
	return sum
}

// RankNDCG 实现 LambdaRank 算法，使用 NDCG 优化排序。
type RankNDCG struct {
	PairMethod       string // "topk" or "mean"
	NumPairPerSample int    // 0 = all pairs
	NDCGExpGain      bool   // true = use 2^rel-1 gain
	Normalization    bool   // true = normalize lambda gradients
}

func (o *RankNDCG) Name() string { return "rank:ndcg" }

func (o *RankNDCG) GetGradient(pred, labels, weights []float64) ([]float64, []float64, error) {
	panic("rank:ndcg requires group info — use GetGradientWithGroup")
}

func (o *RankNDCG) PredTransform(pred []float64) []float64 {
	return pred
}

// GetGradientWithGroup 为排序计算梯度，需要分组信息。
func (o *RankNDCG) GetGradientWithGroup(pred, labels, weights []float64, groups []int, sigma float64) ([]float64, []float64, error) {
	n := len(pred)
	grad := make([]float64, n)
	hess := make([]float64, n)
	if sigma <= 0 {
		sigma = 1.0
	}
	start := 0
	for _, gSize := range groups {
		end := start + gSize
		o.computeGradHess(pred[start:end], labels[start:end], grad[start:end], hess[start:end], sigma)
		if weights != nil {
			for i := start; i < end; i++ {
				grad[i] *= weights[i]
				hess[i] *= weights[i]
			}
		}
		start = end
	}
	return grad, hess, nil
}

// computeGradHess 为单个组计算梯度/Hessian。
func (o *RankNDCG) computeGradHess(pred, labels, grad, hess []float64, sigma float64) {
	m := len(pred)
	if m < 2 {
		return
	}

	// 按预测值降序排序的索引
	idx := make([]int, m)
	for i := range idx {
		idx[i] = i
	}
	for i := 1; i < m; i++ {
		for j := i; j > 0 && pred[idx[j]] > pred[idx[j-1]]; j-- {
			idx[j], idx[j-1] = idx[j-1], idx[j]
		}
	}

	// 计算 IDCG
	sortedLabels := make([]float64, m)
	for i, j := range idx {
		sortedLabels[i] = labels[j]
	}
	expGain := o.NDCGExpGain
	idcg := computeDCG(sortedLabels, 0, expGain)
	if idcg <= 0 {
		idcg = 1
	}

	// 当前 NDCG
	ndcgBefore := computeDCG(sortedLabels, 0, expGain) / idcg

	// 遍历所有 pair
	for a := 0; a < m; a++ {
		for b := a + 1; b < m; b++ {
			i, j := idx[a], idx[b]

			// 计算交换 a,b 后的 NDCG
			swapped := make([]float64, m)
			for k := 0; k < m; k++ {
				if k == a {
					swapped[k] = labels[idx[b]]
				} else if k == b {
					swapped[k] = labels[idx[a]]
				} else {
					swapped[k] = labels[idx[k]]
				}
			}
			ndcgAfter := computeDCG(swapped, 0, expGain) / idcg
			deltaNDCG := math.Abs(ndcgAfter - ndcgBefore)

			diff := pred[i] - pred[j]
			expTerm := math.Exp(sigma * diff)
			lambda := sigma * deltaNDCG * (1.0/(1.0+expTerm) - 1.0/(1.0+1.0/expTerm))
			w := sigma * sigma * deltaNDCG * expTerm / ((1.0 + expTerm) * (1.0 + expTerm))

			grad[i] += lambda
			grad[j] -= lambda
			hess[i] += w
			hess[j] += w
		}
	}
}

// NewObjective 根据类型创建 Objective。
// scalePosWeight 是二分类正类样本权重（仅在 logistic 目标中使用）。
func NewObjective(objType ObjectiveType, numClass int, scalePosWeight float64) Objective {
	switch objType {
	case ObjRegSquareError:
		return &SquaredError{}
	case ObjRegLogistic:
		return &LogisticRegression{ScalePosWeight: scalePosWeight}
	case ObjBinaryLogistic:
		return &LogisticRegression{ScalePosWeight: scalePosWeight}
	case ObjBinaryLogitRaw:
		return &LogisticRaw{ScalePosWeight: scalePosWeight}
	case ObjRegPoisson:
		return &PoissonRegression{}
	case ObjRegGamma:
		return &GammaRegression{}
	case ObjRegTweedie:
		return &TweedieRegression{VariancePower: 1.5} // 默认 p=1.5
	case ObjRankNDCG:
		return &RankNDCG{NDCGExpGain: true, Normalization: true}
	case ObjMultiSoftmax:
		return &MulticlassSoftmax{NumClass: numClass}
	case ObjMultiSoftProb:
		return &MulticlassSoftmax{NumClass: numClass}
	case ObjRegSquaredLogError:
		return &SquaredLogError{}
	case ObjRegAbsoluteError:
		return &AbsoluteError{}
	case ObjRegPseudoHuberError:
		return &PseudoHuberError{Delta: 1.0}
	case ObjRegQuantileError:
		return &QuantileError{Alpha: 0.5}
	case ObjBinaryHinge:
		return &BinaryHinge{}
	case ObjRankMAP:
		return &RankMAP{}
	case ObjRankPairwise:
		return &RankPairwise{}
	case ObjSurvivalCox:
		return &SurvivalCox{}
	case ObjSurvivalAFT:
		return &SurvivalAFT{Distribution: "normal"}
	default:
		return &SquaredError{} // 回退默认值
	}
}

// 辅助函数

// sigmoid 计算逻辑函数：1 / (1 + exp(-x))。
func sigmoid(x float64) float64 {
	var p float64
	if x >= 0 {
		p = 1.0 / (1.0 + math.Exp(-x))
	} else {
		// 针对大负值的数值稳定版本
		ex := math.Exp(x)
		p = ex / (1.0 + ex)
	}
	return p
}

// softmax 计算向量的 softmax 函数。
func softmax(x []float64) []float64 {
	n := len(x)
	out := make([]float64, n)

	// 找到最大值以保证数值稳定性
	maxVal := x[0]
	for _, v := range x[1:] {
		if v > maxVal {
			maxVal = v
		}
	}

	var sum float64
	for i, v := range x {
		out[i] = math.Exp(v - maxVal)
		sum += out[i]
	}

	for i := range out {
		out[i] /= sum
	}

	return out
}

// ── 平方对数误差回归 ─────────────────────────────────────────
//
// 对应 XGBoost 的 reg:squaredlogerror 目标函数。
// 梯度使用 log1p 变换：g = (log(p+1) - log(l+1)) / (p+1)
// Hessian：h = (1 - log(p+1) + log(l+1)) / (p+1)²

type SquaredLogError struct{}

func (o *SquaredLogError) Name() string { return "reg:squaredlogerror" }

func (o *SquaredLogError) GetGradient(pred, labels, weights []float64) ([]float64, []float64, error) {
	n := len(pred)
	grad := make([]float64, n)
	hess := make([]float64, n)
	for i := 0; i < n; i++ {
		p := pred[i]
		lp := log1pSafe(p)
		ll := log1pSafe(labels[i])
		diff := lp - ll
		denom := 1.0 + p
		grad[i] = diff / denom
		hess[i] = (1.0 - diff) / (denom * denom)
		if weights != nil {
			grad[i] *= weights[i]
			hess[i] *= weights[i]
		}
	}
	return grad, hess, nil
}

func (o *SquaredLogError) PredTransform(pred []float64) []float64 {
	// 返回预测值（原始空间）
	out := make([]float64, len(pred))
	copy(out, pred)
	return out
}

// ── 绝对误差回归（L1）────────────────────────────────────────
//
// 对应 XGBoost 的 reg:absoluteerror 目标函数。
// 使用 L1 损失的平滑近似：
// 梯度：g = sign(pred - label)（当 |pred-label| > eps 时）
// Hessian：h = 1/|pred-label|（近似）

type AbsoluteError struct{}

func (o *AbsoluteError) Name() string { return "reg:absoluteerror" }

func (o *AbsoluteError) GetGradient(pred, labels, weights []float64) ([]float64, []float64, error) {
	n := len(pred)
	grad := make([]float64, n)
	hess := make([]float64, n)
	eps := 1e-8
	for i := 0; i < n; i++ {
		diff := pred[i] - labels[i]
		absDiff := math.Abs(diff)
		if absDiff < eps {
			grad[i] = 0.0
			hess[i] = 1.0
		} else {
			grad[i] = diff / absDiff // sign
			hess[i] = 1.0 / absDiff
		}
		if weights != nil {
			grad[i] *= weights[i]
			hess[i] *= weights[i]
		}
	}
	return grad, hess, nil
}

func (o *AbsoluteError) PredTransform(pred []float64) []float64 {
	out := make([]float64, len(pred))
	copy(out, pred)
	return out
}

// ── Pseudo Huber 误差 ─────────────────────────────────────────
//
// 对应 XGBoost 的 reg:pseudohubererror 目标函数。
// 平滑 L1：L = delta² * (sqrt(1 + ((p-l)/delta)²) - 1)
// g = delta * tanh(delta * (p-l)) / sqrt(delta² + (p-l)²) 的替代形式

type PseudoHuberError struct {
	Delta float64
}

func (o *PseudoHuberError) Name() string { return "reg:pseudohubererror" }

func (o *PseudoHuberError) GetGradient(pred, labels, weights []float64) ([]float64, []float64, error) {
	n := len(pred)
	delta := o.Delta
	if delta <= 0 {
		delta = 1.0
	}
	grad := make([]float64, n)
	hess := make([]float64, n)
	for i := 0; i < n; i++ {
		diff := pred[i] - labels[i]
		denom := math.Sqrt(delta*delta + diff*diff)
		grad[i] = delta * diff / denom
		hess[i] = delta * delta / (denom * denom * denom)
		if weights != nil {
			grad[i] *= weights[i]
			hess[i] *= weights[i]
		}
	}
	return grad, hess, nil
}

func (o *PseudoHuberError) PredTransform(pred []float64) []float64 {
	out := make([]float64, len(pred))
	copy(out, pred)
	return out
}

// ── 分位数回归 ─────────────────────────────────────────────────
//
// 对应 XGBoost 的 reg:quantileerror 目标函数。
// pinball loss：
// g = alpha - 1 (当 pred < label 时), alpha (当 pred >= label 时)
// h = 1（恒定 Hessian，与 XGBoost 一致）

type QuantileError struct {
	Alpha float64
}

func (o *QuantileError) Name() string { return "reg:quantileerror" }

func (o *QuantileError) GetGradient(pred, labels, weights []float64) ([]float64, []float64, error) {
	n := len(pred)
	grad := make([]float64, n)
	hess := make([]float64, n)
	alpha := o.Alpha
	if alpha <= 0 || alpha >= 1 {
		alpha = 0.5
	}
	for i := 0; i < n; i++ {
		if pred[i] >= labels[i] {
			grad[i] = alpha
		} else {
			grad[i] = alpha - 1.0
		}
		hess[i] = 1.0
		if weights != nil {
			grad[i] *= weights[i]
			hess[i] *= weights[i]
		}
	}
	return grad, hess, nil
}

func (o *QuantileError) PredTransform(pred []float64) []float64 {
	out := make([]float64, len(pred))
	copy(out, pred)
	return out
}

// ── 二分类 Hinge 损失 ─────────────────────────────────────────
//
// 对应 XGBoost 的 binary:hinge 目标函数。
// g = -label（当 label*pred < 1 时）, 0（否则）
// h = 1（恒定）

type BinaryHinge struct{}

func (o *BinaryHinge) Name() string { return "binary:hinge" }

func (o *BinaryHinge) GetGradient(pred, labels, weights []float64) ([]float64, []float64, error) {
	n := len(pred)
	grad := make([]float64, n)
	hess := make([]float64, n)
	for i := 0; i < n; i++ {
		label := labels[i]
		// 将标签从 {0,1} 映射到 {-1,1}（hinge 损失期望）
		y := 2.0*label - 1.0
		if y*pred[i] < 1.0 {
			grad[i] = -y
		} else {
			grad[i] = 0.0
		}
		hess[i] = 1.0
		if weights != nil {
			grad[i] *= weights[i]
			hess[i] *= weights[i]
		}
	}
	return grad, hess, nil
}

func (o *BinaryHinge) PredTransform(pred []float64) []float64 {
	out := make([]float64, len(pred))
	for i, v := range pred {
		if v >= 0 {
			out[i] = 1.0
		} else {
			out[i] = 0.0
		}
	}
	return out
}

// ── LambdaRank MAP ────────────────────────────────────────────
//
// 对应 XGBoost 的 rank:map 目标函数。
// 与 rank:ndcg 共享架构，但使用 Average Precision 替代 NDCG。

type RankMAP struct{}

func (o *RankMAP) Name() string { return "rank:map" }

func (o *RankMAP) GetGradient(pred, labels, weights []float64) ([]float64, []float64, error) {
	panic("rank:map requires group info — use GetGradientWithGroup")
}

func (o *RankMAP) PredTransform(pred []float64) []float64 {
	return pred
}

// GetGradientWithGroup 为 rank:map 计算梯度。
func (o *RankMAP) GetGradientWithGroup(pred, labels, weights []float64, groups []int, sigma float64) ([]float64, []float64, error) {
	n := len(pred)
	grad := make([]float64, n)
	hess := make([]float64, n)
	if sigma <= 0 {
		sigma = 1.0
	}
	start := 0
	for _, gSize := range groups {
		end := start + gSize
		o.computeMAPGradHess(pred[start:end], labels[start:end], grad[start:end], hess[start:end], sigma)
		if weights != nil {
			for i := start; i < end; i++ {
				grad[i] *= weights[i]
				hess[i] *= weights[i]
			}
		}
		start = end
	}
	return grad, hess, nil
}

// computeAP 计算 Average Precision。
func computeAP(labels []float64) float64 {
	totalPos := 0.0
	for _, l := range labels {
		if l > 0 {
			totalPos++
		}
	}
	if totalPos == 0 {
		return 0
	}
	var ap, cumPos float64
	for i, l := range labels {
		if l > 0 {
			cumPos++
			ap += cumPos / float64(i+1)
		}
	}
	return ap / totalPos
}

func (o *RankMAP) computeMAPGradHess(pred, labels, grad, hess []float64, sigma float64) {
	m := len(pred)
	if m < 2 {
		return
	}

	// 按预测值降序排序
	idx := make([]int, m)
	for i := range idx {
		idx[i] = i
	}
	for i := 1; i < m; i++ {
		for j := i; j > 0 && pred[idx[j]] > pred[idx[j-1]]; j-- {
			idx[j], idx[j-1] = idx[j-1], idx[j]
		}
	}

	// 当前 AP
	sortedLabels := make([]float64, m)
	for i, j := range idx {
		sortedLabels[i] = labels[j]
	}
	apBefore := computeAP(sortedLabels)

	// 遍历所有 pair
	for a := 0; a < m; a++ {
		for b := a + 1; b < m; b++ {
			i, j := idx[a], idx[b]

			// 计算交换 a,b 后的 AP
			swapped := make([]float64, m)
			for k := 0; k < m; k++ {
				if k == a {
					swapped[k] = labels[idx[b]]
				} else if k == b {
					swapped[k] = labels[idx[a]]
				} else {
					swapped[k] = labels[idx[k]]
				}
			}
			apAfter := computeAP(swapped)
			deltaMAP := math.Abs(apAfter - apBefore)

			diff := pred[i] - pred[j]
			expTerm := math.Exp(sigma * diff)
			lambda := sigma * deltaMAP * (1.0/(1.0+expTerm) - 1.0/(1.0+1.0/expTerm))
			w := sigma * sigma * deltaMAP * expTerm / ((1.0 + expTerm) * (1.0 + expTerm))

			grad[i] += lambda
			grad[j] -= lambda
			hess[i] += w
			hess[j] += w
		}
	}
}

// ── Pairwise 排序 ─────────────────────────────────────────────
//
// 对应 XGBoost 的 rank:pairwise 目标函数。
// 简单的成对排序损失，不需要 Group 信息。
// 使用所有可能的 pair 或随机采样。

type RankPairwise struct{}

func (o *RankPairwise) Name() string { return "rank:pairwise" }

func (o *RankPairwise) GetGradient(pred, labels, weights []float64) ([]float64, []float64, error) {
	n := len(pred)
	grad := make([]float64, n)
	hess := make([]float64, n)
	sigma := 1.0

	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			if labels[i] == labels[j] {
				continue
			}
			// 确保 i 是更相关的
			if labels[i] < labels[j] {
				i, j = j, i
			}
			diff := pred[i] - pred[j]
			expTerm := math.Exp(sigma * diff)
			lambda := -sigma / (1.0 + expTerm)
			w := sigma * sigma * expTerm / ((1.0 + expTerm) * (1.0 + expTerm))

			grad[i] += lambda
			grad[j] -= lambda
			hess[i] += w
			hess[j] += w
		}
	}

	if weights != nil {
		for i := range grad {
			grad[i] *= weights[i]
			hess[i] *= weights[i]
		}
	}

	return grad, hess, nil
}

func (o *RankPairwise) PredTransform(pred []float64) []float64 {
	return pred
}

// GetGradientWithGroup 为 rank:pairwise 按组计算梯度。
func (o *RankPairwise) GetGradientWithGroup(pred, labels, weights []float64, groups []int, _ float64) ([]float64, []float64, error) {
	return o.GetGradient(pred, labels, weights)
}

// ── Cox 比例风险回归 ──────────────────────────────────────────
//
// 对应 XGBoost 的 survival:cox 目标函数。
// 使用 Efron 近似的偏似然。

type SurvivalCox struct{}

func (o *SurvivalCox) Name() string { return "survival:cox" }

func (o *SurvivalCox) GetGradient(pred, labels, weights []float64) ([]float64, []float64, error) {
	n := len(pred)
	grad := make([]float64, n)
	hess := make([]float64, n)

	// 按预测值降序排序用于风险集计算
	type entry struct {
		pred   float64
		label  float64
		weight float64
		idx    int
	}
	entries := make([]entry, n)
	for i := 0; i < n; i++ {
		w := 1.0
		if weights != nil {
			w = weights[i]
		}
		entries[i] = entry{pred: pred[i], label: labels[i], weight: w, idx: i}
	}
	// 按预测值降序排序
	for i := 1; i < n; i++ {
		for j := i; j > 0 && entries[j].pred > entries[j-1].pred; j-- {
			entries[j], entries[j-1] = entries[j-1], entries[j]
		}
	}

	// Efron 近似：反向扫描风险集
	var riskGradSum, riskHessSum float64
	for i := n - 1; i >= 0; i-- {
		riskGradSum += entries[i].weight * math.Exp(entries[i].pred)
		riskHessSum += entries[i].weight * math.Exp(entries[i].pred)
	}

	// 正向扫描用于梯度
	var cumExp, cumExp2 float64
	// Efron 绑定的计数
	var tieCount int
	var tieExp, tieExp2 float64

	for i := 0; i < n; i++ {
		expPred := math.Exp(entries[i].pred)
		if entries[i].label > 0 { // 事件发生
			// 简化 Efron 近似（无绑定的情况）
			grad[entries[i].idx] = -entries[i].weight + entries[i].weight*expPred*(riskGradSum-cumExp)/(riskHessSum-cumExp2)
			hess[entries[i].idx] = entries[i].weight * expPred * (riskGradSum - cumExp) / (riskHessSum - cumExp2)
		} else { // 右删失
			grad[entries[i].idx] = 0
			hess[entries[i].idx] = 0
		}
		cumExp += entries[i].weight * expPred
		cumExp2 += entries[i].weight * expPred
		_ = tieCount
		_ = tieExp
		_ = tieExp2
	}

	return grad, hess, nil
}

func (o *SurvivalCox) PredTransform(pred []float64) []float64 {
	// Cox 预测为风险比 exp(pred)
	out := make([]float64, len(pred))
	for i, v := range pred {
		out[i] = math.Exp(v)
	}
	return out
}

// ── AFT 生存分析 ──────────────────────────────────────────────
//
// 对应 XGBoost 的 survival:aft 目标函数。
// 使用对数正态分布的参数化 AFT 模型。

// SurvivalAFT 实现加速失效时间模型（Accelerated Failure Time）。
type SurvivalAFT struct {
	// AFT 损失分布类型："normal", "logistic", "extreme"
	Distribution string
	// 是否使用对数变换
	LogInput bool
}

func (o *SurvivalAFT) Name() string { return "survival:aft" }

func (o *SurvivalAFT) GetGradient(pred, labels, weights []float64) ([]float64, []float64, error) {
	n := len(pred)
	grad := make([]float64, n)
	hess := make([]float64, n)

	dist := o.Distribution
	if dist == "" {
		dist = "normal"
	}

	for i := 0; i < n; i++ {
		y := labels[i]
		if y < 0 {
			// 负标签表示右删失，边界处理
			y = -y
		}
		if y <= 0 {
			y = 1e-6
		}

		// 对数变换
		logT := math.Log(y)
		standardized := (logT - pred[i]) // 假设 sigma=1 简化

		var g, h float64
		switch dist {
		case "logistic":
			// 逻辑分布：L = log(1 + exp(-s)) - (1-delta)*log(1 + exp(-s))
			// 简化：使用标准逻辑分布
			expS := math.Exp(standardized)
			g = -(1.0 / (1.0 + expS)) + 1.0
			h = expS / ((1.0 + expS) * (1.0 + expS))
		case "extreme":
			// 极值分布（Gumbel）
			expS := math.Exp(standardized)
			g = math.Exp(standardized-expS) - 1.0 + expS
			h = expS * (1 + expS - math.Exp(standardized-expS))
		default: // "normal"
			// 对数正态分布
			g = standardized
			h = 1.0
		}

		grad[i] = -g
		hess[i] = h
		if hess[i] < 0 {
			hess[i] = 0.5 // 确保正定
		}

		if weights != nil {
			grad[i] *= weights[i]
			hess[i] *= weights[i]
		}
	}

	return grad, hess, nil
}

func (o *SurvivalAFT) PredTransform(pred []float64) []float64 {
	// AFT 预测 exp(pred)（预计生存时间）
	out := make([]float64, len(pred))
	for i, v := range pred {
		out[i] = math.Exp(v)
	}
	return out
}

// ── 自定义目标函数、评估函数和回调 ──────────────────────────

// CustomObjFunc 是自定义目标函数的签名。
type CustomObjFunc func(pred, labels, weights []float64) (grad, hess []float64, err error)

// CustomObj 包装自定义目标函数为 Objective 接口。
type CustomObj struct {
	ObjName string
	Fn      CustomObjFunc
}

func (o *CustomObj) Name() string { return o.ObjName }

func (o *CustomObj) GetGradient(pred, labels, weights []float64) ([]float64, []float64, error) {
	return o.Fn(pred, labels, weights)
}

func (o *CustomObj) PredTransform(pred []float64) []float64 {
	return pred
}

// CustomEvalFunc 是自定义评估函数的签名。
type CustomEvalFunc func(pred, labels []float64) (name string, value float64)

// CustomMetric 包装自定义评估函数为 Metric 接口。
type CustomMetric struct {
	Fn CustomEvalFunc
}

func (m *CustomMetric) Name() string { return "custom" }

func (m *CustomMetric) Evaluate(preds []float64, dm *DMatrix) float64 {
	_, v := m.Fn(preds, dm.Labels)
	return v
}

// CallbackFunc 是训练回调函数的签名。
// 在每轮迭代后调用。如果返回 true，训练将提前停止。
// iter 是当前迭代轮数（从 0 开始），preds 是当前预测值。
type CallbackFunc func(iter int, preds []float64, dm *DMatrix) bool
