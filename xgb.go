// Package xgb 提供了 XGBoost 的梯度提升决策树（GBDT）训练和推理引擎的纯 Go 实现。
//
// 本实现是对 XGBoost C++ 源码的逐行翻译，
// 目标是与原始实现达到 100% 精度一致。
//
// 功能特性：
//   - 精确贪婪树构建器
//   - 回归、二分类和多分类目标函数
//   - 完整的正则化（gamma、lambda、alpha）
//   - 行和列子采样
//   - 模型序列化（JSON）
//
// 精度目标：预测值与 XGBoost C++ 的偏差 < 1e-8。
package xgb

import "errors"

// BoostType 定义提升算法类型。
type BoostType int

const (
	BoostGBTree BoostType = iota
	BoostDART
	BoostGBLinear
)

// ObjectiveType 枚举支持的目标函数类型。
// 这些对应 XGBoost 的目标字符串参数。
type ObjectiveType int

const (
	ObjRegSquareError ObjectiveType = iota // reg:squarederror
	ObjRegLogistic                         // reg:logistic
	ObjBinaryLogistic                      // binary:logistic
	ObjBinaryLogitRaw                      // binary:logitraw
	ObjRegGamma                            // reg:gamma
	ObjMultiSoftmax                        // multi:softmax
	ObjMultiSoftProb                       // multi:softprob
)

// String 返回目标类型的 XGBoost 参数字符串。
func (o ObjectiveType) String() string {
	switch o {
	case ObjRegSquareError:
		return "reg:squarederror"
	case ObjRegLogistic:
		return "reg:logistic"
	case ObjBinaryLogistic:
		return "binary:logistic"
	case ObjBinaryLogitRaw:
		return "binary:logitraw"
	case ObjRegGamma:
		return "reg:gamma"
	case ObjMultiSoftmax:
		return "multi:softmax"
	case ObjMultiSoftProb:
		return "multi:softprob"
	default:
		return "unknown"
	}
}

// ParseObjectiveType 将目标字符串解析为 ObjectiveType 枚举。
func ParseObjectiveType(s string) (ObjectiveType, error) {
	switch s {
	case "reg:squarederror":
		return ObjRegSquareError, nil
	case "reg:logistic":
		return ObjRegLogistic, nil
	case "binary:logistic":
		return ObjBinaryLogistic, nil
	case "binary:logitraw":
		return ObjBinaryLogitRaw, nil
	case "reg:gamma":
		return ObjRegGamma, nil
	case "multi:softmax":
		return ObjMultiSoftmax, nil
	case "multi:softprob":
		return ObjMultiSoftProb, nil
	default:
		return -1, errors.New("xgb: unknown objective type: " + s)
	}
}
