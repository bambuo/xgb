package xgb

import "errors"

// Config 保存所有训练超参数。
//
// 默认值与 XGBoost 默认值保持一致。
type Config struct {
	// BoostType 选择提升算法（默认：BoostGBTree）。
	BoostType BoostType

	// NumTrees 是提升轮数（n_estimators），默认 100。
	NumTrees int

	// LearningRate 是步长收缩率（eta），默认 0.3。
	LearningRate float64

	// MaxDepth 是最大树深度，默认 6。
	MaxDepth int

	// Gamma 是分裂所需的最小损失减少量，默认 0。
	Gamma float64

	// Lambda 是 L2 正则化权重，默认 1。
	Lambda float64

	// Alpha 是 L1 正则化权重，默认 0。
	Alpha float64

	// MinChildWeight 是子节点中 Hessian 的最小和，默认 1。
	MinChildWeight float64

	// Subsample 是行采样比例，默认 1。
	Subsample float64

	// ColSampleByTree 是每棵树的特征采样比例，默认 1。
	ColSampleByTree float64

	// ColSampleByLevel 是每层的特征采样比例，默认 1。
	ColSampleByLevel float64

	// ColSampleByNode 是每个节点的特征采样比例，默认 1。
	ColSampleByNode float64

	// BaseScore 是初始预测分数，默认 0.5。
	BaseScore float64

	// NumClass 是类别数（回归/二分类时为 1），默认 1。
	NumClass int

	// Objective 选择损失函数。
	Objective ObjectiveType

	// Seed 是用于可重现性的随机种子（0 = 随机）。
	Seed int64

	// Verbosity 控制日志输出：0=静默，1=警告，2=信息，3=调试。
	Verbosity int

	// MaxBin 是直方图分箱的最大箱数，默认 256。
	// 对应 XGBoost 的 max_bin 参数。
	MaxBin int
}

// DefaultConfig 返回一个包含 XGBoost 默认值的 Config。
func DefaultConfig() *Config {
	return &Config{
		BoostType:        BoostGBTree,
		NumTrees:         100,
		LearningRate:     0.3,
		MaxDepth:         6,
		Gamma:            0.0,
		Lambda:           1.0,
		Alpha:            0.0,
		MinChildWeight:   1.0,
		Subsample:        1.0,
		ColSampleByTree:  1.0,
		ColSampleByLevel: 1.0,
		ColSampleByNode:  1.0,
		BaseScore:        0.0, // 对数几率空间：0 → sigmoid(0)=0.5 概率
		NumClass:         1,
		Objective:        ObjRegSquareError,
		Seed:             0,
		Verbosity:        1,
		MaxBin:           256,
	}
}

// Validate 检查配置参数是否在有效范围内。
func (c *Config) Validate() error {
	if c.NumTrees <= 0 {
		return errors.New("xgb: NumTrees must be positive")
	}
	if c.LearningRate <= 0 {
		return errors.New("xgb: LearningRate must be positive")
	}
	if c.MaxDepth <= 0 {
		return errors.New("xgb: MaxDepth must be positive")
	}
	if c.Gamma < 0 {
		return errors.New("xgb: Gamma must be non-negative")
	}
	if c.Lambda < 0 {
		return errors.New("xgb: Lambda must be non-negative")
	}
	if c.Alpha < 0 {
		return errors.New("xgb: Alpha must be non-negative")
	}
	if c.MinChildWeight < 0 {
		return errors.New("xgb: MinChildWeight must be non-negative")
	}
	if c.Subsample <= 0 || c.Subsample > 1 {
		return errors.New("xgb: Subsample must be in (0, 1]")
	}
	if c.ColSampleByTree <= 0 || c.ColSampleByTree > 1 {
		return errors.New("xgb: ColSampleByTree must be in (0, 1]")
	}
	if c.ColSampleByLevel <= 0 || c.ColSampleByLevel > 1 {
		return errors.New("xgb: ColSampleByLevel must be in (0, 1]")
	}
	if c.ColSampleByNode <= 0 || c.ColSampleByNode > 1 {
		return errors.New("xgb: ColSampleByNode must be in (0, 1]")
	}
	if c.NumClass < 1 {
		return errors.New("xgb: NumClass must be >= 1")
	}
	return nil
}
