// Package xgb 提供了 XGBoost 梯度提升决策树（GBDT）训练和推理引擎的纯 Go 实现。
//
// 本实现参考 XGBoost C++ 源码的算法逻辑，使用纯 float64 精度（高于 XGBoost 的 float32）。
// 核心算法（层级遍历、bin-based 分区、加权分位数边界）与 C++ 版本一致。
//
// 功能特性：
//
// 树方法：
//   - hist：直方图方法（层级遍历 + bin-based 分区），默认
//   - exact：精确贪婪算法（逐值枚举最佳分裂）
//
// 目标函数（9 种）：
//   - reg:squarederror — 平方损失回归
//   - reg:logistic — Logistic 回归
//   - binary:logistic — 二分类逻辑回归
//   - binary:logitraw — 二分类原始输出
//   - count:poisson — Poisson 回归（log 链接）
//   - reg:gamma — Gamma 回归（log 链接）
//   - reg:tweedie — Tweedie 回归（方差幂参数可配置）
//   - multi:softmax / multi:softprob — 多分类（softmax）
//   - rank:ndcg — LambdaRank 排序（需设置 Group）
//
// Booster（2 种）：
//   - gbtree — 传统梯度提升（默认）
//   - dart — 带 Dropout 的梯度提升（rate_drop / one_drop / skip_drop）
//
// 训练控制：
//   - 学习率（eta）· 最大深度 · 最小分裂增益（gamma）
//   - L1/L2 正则化（alpha / lambda）· 最小叶节点权重
//   - 行子采样（subsample）· 列子采样（colsample_bytree/level/node）
//   - 正类权重（scale_pos_weight）· 单调约束 · 交互约束
//   - 早停（early_stopping_rounds + EvalData）
//   - 自定义目标函数（CustomObjFunc）· 自定义评估函数（CustomEvalFunc）
//
// 数据格式：
//   - 稠密 DMatrix — NewDMatrix
//   - CSR 稀疏矩阵 — NewDMatrixFromCSR
//   - 分组信息 — SetGroup（排序任务）
//
// 模型管理：
//   - JSON 序列化（SaveModel / LoadModel）
//   - 继续训练（ContinueTrain）
//   - 特征重要性（GetScore / GetFScore）
//   - SHAP 值归因（SHAP）
//
// 精度说明：
//
//	Go 实现使用纯 float64 精度（梯度、直方图、叶值计算全部在 float64 中完成），
//	因此预测值不会与 XGBoost（内部用 float32）bit-exact 一致，
//	但在算法逻辑（树结构、分裂点选择、叶值权重）上完全等价。
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
	ObjRegSquareError      ObjectiveType = iota // reg:squarederror
	ObjRegLogistic                              // reg:logistic
	ObjBinaryLogistic                           // binary:logistic
	ObjBinaryLogitRaw                           // binary:logitraw
	ObjRegGamma                                 // reg:gamma
	ObjMultiSoftmax                             // multi:softmax
	ObjMultiSoftProb                            // multi:softprob
	ObjRegPoisson                               // count:poisson
	ObjRegTweedie                               // reg:tweedie
	ObjRankNDCG                                 // rank:ndcg
	ObjRegSquaredLogError                       // reg:squaredlogerror
	ObjRegAbsoluteError                         // reg:absoluteerror
	ObjRegPseudoHuberError                      // reg:pseudohubererror
	ObjRegQuantileError                         // reg:quantileerror
	ObjBinaryHinge                              // binary:hinge
	ObjRankMAP                                  // rank:map
	ObjRankPairwise                             // rank:pairwise
	ObjSurvivalCox                              // survival:cox
	ObjSurvivalAFT                              // survival:aft
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
	case ObjRegPoisson:
		return "count:poisson"
	case ObjRegTweedie:
		return "reg:tweedie"
	case ObjRankNDCG:
		return "rank:ndcg"
	case ObjMultiSoftmax:
		return "multi:softmax"
	case ObjMultiSoftProb:
		return "multi:softprob"
	case ObjRegSquaredLogError:
		return "reg:squaredlogerror"
	case ObjRegAbsoluteError:
		return "reg:absoluteerror"
	case ObjRegPseudoHuberError:
		return "reg:pseudohubererror"
	case ObjRegQuantileError:
		return "reg:quantileerror"
	case ObjBinaryHinge:
		return "binary:hinge"
	case ObjRankMAP:
		return "rank:map"
	case ObjRankPairwise:
		return "rank:pairwise"
	case ObjSurvivalCox:
		return "survival:cox"
	case ObjSurvivalAFT:
		return "survival:aft"
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
	case "count:poisson":
		return ObjRegPoisson, nil
	case "reg:tweedie":
		return ObjRegTweedie, nil
	case "rank:ndcg":
		return ObjRankNDCG, nil
	case "multi:softmax":
		return ObjMultiSoftmax, nil
	case "multi:softprob":
		return ObjMultiSoftProb, nil
	case "reg:squaredlogerror":
		return ObjRegSquaredLogError, nil
	case "reg:absoluteerror":
		return ObjRegAbsoluteError, nil
	case "reg:pseudohubererror":
		return ObjRegPseudoHuberError, nil
	case "reg:quantileerror":
		return ObjRegQuantileError, nil
	case "binary:hinge":
		return ObjBinaryHinge, nil
	case "rank:map":
		return ObjRankMAP, nil
	case "rank:pairwise":
		return ObjRankPairwise, nil
	case "survival:cox":
		return ObjSurvivalCox, nil
	case "survival:aft":
		return ObjSurvivalAFT, nil
	default:
		return -1, errors.New("xgb: unknown objective type: " + s)
	}
}
