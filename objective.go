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

// LogisticRegression 是目标函数：reg:logistic, binary:logistic
type LogisticRegression struct{}

func (o *LogisticRegression) Name() string { return "binary:logistic" }

func (o *LogisticRegression) GetGradient(pred, labels, weights []float64) ([]float64, []float64, error) {
	n := len(pred)
	grad := make([]float64, n)
	hess := make([]float64, n)

	for i := 0; i < n; i++ {
		p := sigmoid(pred[i])   // p = σ(y∧)
		grad[i] = p - labels[i] // g = p - y
		hess[i] = p * (1.0 - p) // h = p·(1-p)

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
type LogisticRaw struct{}

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

// NewObjective 根据类型创建 Objective。
func NewObjective(objType ObjectiveType, numClass int) Objective {
	switch objType {
	case ObjRegSquareError:
		return &SquaredError{}
	case ObjRegLogistic:
		return &LogisticRegression{}
	case ObjBinaryLogistic:
		return &LogisticRegression{}
	case ObjBinaryLogitRaw:
		return &LogisticRaw{}
	case ObjMultiSoftmax:
		return &MulticlassSoftmax{NumClass: numClass}
	case ObjMultiSoftProb:
		return &MulticlassSoftmax{NumClass: numClass}
	default:
		return &SquaredError{} // 回退默认值
	}
}

// 辅助函数

// sigmoid 计算逻辑函数：1 / (1 + exp(-x))
func sigmoid(x float64) float64 {
	if x >= 0 {
		return 1.0 / (1.0 + math.Exp(-x))
	}
	// 针对大负值的数值稳定版本
	ex := math.Exp(x)
	return ex / (1.0 + ex)
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
