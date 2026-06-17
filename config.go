package xgb

import (
	"errors"
	"io"
)

// Config 保存所有训练超参数。
//
// 默认值与 XGBoost 默认值保持一致。
type Config struct {
	// TreeMethod 选择树构建算法："hist"（直方图）或 "exact"（精确贪婪）。
	// 对应 XGBoost 的 tree_method 参数。
	TreeMethod string

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

	// MaxDeltaStep 是叶节点权重的最大步长裁剪，默认 0（无裁剪）。
	// 对应 XGBoost 的 max_delta_step 参数。
	// 对不平衡分类任务有保护作用。
	MaxDeltaStep float64

	// BoostFromAverage 控制是否从标签均值自动估计 BaseScore。
	// 对应 XGBoost 的 boost_from_average 参数（默认 true）。
	BoostFromAverage bool

	// Subsample 是行采样比例，默认 1。
	Subsample float64

	// SamplingMethod 是行采样方法："uniform" 或 "gradient_based"。
	// 对应 XGBoost 的 sampling_method 参数。默认 "uniform"。
	SamplingMethod string

	// FeatureWeights 是特征级别的采样权重，用于列采样。
	// 对应 XGBoost 的 feature_weights 参数。nil 表示均匀权重。
	FeatureWeights []float64

	// ColSampleByTree 是每棵树的特征采样比例，默认 1。
	ColSampleByTree float64

	// ColSampleByLevel 是每层的特征采样比例，默认 1。
	ColSampleByLevel float64

	// ColSampleByNode 是每个节点的特征采样比例，默认 1。
	ColSampleByNode float64

	// BaseScore 是初始预测分数（概率空间，0~1），默认 0.5。
	// 回归目标直接使用此值作为初始预测。
	// 二分类目标自动转换为对数几率（log-odds）。
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
	// 对应 XGBoost 的 max_bin 参数。仅 hist 方法使用。
	MaxBin int

	// MaxLeaves 是树的最大叶节点数，0 = 不限制（使用 MaxDepth）。
	// 对应 XGBoost 的 max_leaves 参数（lossguide 增长策略）。
	MaxLeaves int

	// NumParallelTree 是每轮构建的并行树数，1 = 标准提升。
	// 对应 XGBoost 的 num_parallel_tree 参数。
	NumParallelTree int

	// ProcessType 控制训练模式："default"（正常训练）或 "update"（更新已有树）。
	// 对应 XGBoost 的 process_type 参数。
	ProcessType string

	// RefreshLeaf 控制是否在每轮构建后刷新叶节点值（不改变树结构）。
	// 对应 XGBoost 的 refresh_leaf 参数。
	RefreshLeaf bool

	// MaxCachedHistNode 是直方图缓存的节点数上限，0 = 不限制。
	// 对应 XGBoost 的 max_cached_hist_node 参数。
	MaxCachedHistNode int

	// EarlyStoppingRounds 早停轮数，0 = 禁用。
	// 当验证集指标连续 N 轮不改善时停止训练。
	EarlyStoppingRounds int

	// EvalData 是可选验证集，用于早停和指标监控。
	EvalData *DMatrix

	// EvalMetricName 是早停使用的指标名（"rmse", "mae", "logloss", "error", "auc"）。
	// 为空时使用第一个 metrics。
	EvalMetricName string

	// ScalePosWeight 是二分类正类样本权重，用于处理不平衡数据。
	// 对应 XGBoost 的 scale_pos_weight 参数。默认 1.0。
	ScalePosWeight float64

	// MonotoneConstraints 单调约束。key=特征索引, value=约束方向。
	//  1  = 单调递增（预测值随特征值增大而增大）
	//  -1 = 单调递减（预测值随特征值增大而减小）
	//  0  = 无约束（默认）
	// 未在此 map 中的特征视为无约束。
	// 对应 XGBoost 的 monotone_constraints 参数。
	MonotoneConstraints map[int]int

	// InteractionConstraints 交互约束。定义哪些特征组可以相互交互。
	// 例如 [][]int{{0, 1}, {2, 3}} 表示 f0/f1 可以交互, f2/f3 可以交互,
	// 但 f0 和 f2 不能出现在同一分支中。
	// 对应 XGBoost 的 interaction_constraints 参数。
	InteractionConstraints [][]int

	// CustomObjective 自定义目标函数。当设置后，会覆盖 Objective 字段。
	// 函数签名见 CustomObjFunc。
	CustomObjective CustomObjFunc

	// CustomEval 自定义评估函数。当设置后，会追加到 metrics 列表中。
	// 函数签名见 CustomEvalFunc。
	CustomEval CustomEvalFunc

	// Callbacks 是训练回调函数列表，每轮迭代后调用。
	// 如果任意回调返回 true，训练将提前停止。
	Callbacks []CallbackFunc

	// LogWriter 是训练日志的输出目标。nil 表示输出到 stdout。
	LogWriter io.Writer

	// ValidateParams 控制是否检查未知配置参数（默认 false）。
	// 对应 XGBoost 的 validate_parameters 参数。
	ValidateParams bool

	// DisableDefaultEvalMetric 控制是否禁用默认评估指标（默认 false）。
	// 对应 XGBoost 的 disable_default_eval_metric 参数。
	DisableDefaultEvalMetric bool

	// TweedieVariancePower 是 Tweedie 回归的方差幂参数（1~2），默认 1.5。
	TweedieVariancePower float64

	// QuantileAlpha 是分位数回归的分位数参数（0~1），默认 0.5。
	// 对应 XGBoost 的 quantile_alpha 参数。
	QuantileAlpha float64

	// HuberSlope 是 Pseudo Huber 损失的斜率参数 delta，默认 1.0。
	// 对应 XGBoost 的 huber_slope 参数。
	HuberSlope float64

	// RateDrop 是 DART boosting 的树丢弃概率，默认 0.0。
	RateDrop float64

	// OneDrop 控制 DART 是否每轮最多丢弃一棵树，默认 false。
	OneDrop bool

	// SkipDrop 是 DART 跳过丢弃的概率，默认 0.0。
	SkipDrop float64

	// SampleType 是 DART 的采样类型："uniform" 或 "weighted"。
	// 对应 XGBoost 的 sample_type 参数。默认 "uniform"。
	SampleType string

	// NormalizeType 是 DART 的归一化类型："tree" 或 "forest"。
	// 对应 XGBoost 的 normalize_type 参数。默认 "tree"。
	NormalizeType string

	// ── 排序（Learning to Rank）详细参数 ──────────────────────

	// LambdaRankPairMethod 是 LambdaRank 的 pair 生成方法："topk" 或 "mean"。
	LambdaRankPairMethod string

	// LambdaRankNumPairPerSample 是每个样本的 pair 数量上限（0 = 全部）。
	LambdaRankNumPairPerSample int

	// LambdaRankNormalization 控制是否对 LambdaRank 梯度进行归一化。
	LambdaRankNormalization bool

	// NDCGExpGain 控制是否使用指数增益（2^rel - 1），false 使用线性增益。
	NDCGExpGain bool

	// ── 类别特征参数 ──────────────────────────────────────────

	// MaxCatToOneHot 是触发 one-hot 编码的最大类别数。
	MaxCatToOneHot int

	// MaxCatThreshold 是类别分裂的最大类别阈值。
	MaxCatThreshold int
}

// DefaultConfig 返回一个包含 XGBoost 默认值的 Config。
func DefaultConfig() *Config {
	return &Config{
		TreeMethod:                 "hist",
		BoostType:                  BoostGBTree,
		NumTrees:                   100,
		LearningRate:               0.3,
		MaxDepth:                   6,
		Gamma:                      0.0,
		Lambda:                     1.0,
		Alpha:                      0.0,
		MinChildWeight:             1.0,
		MaxDeltaStep:               0.0,
		BoostFromAverage:           true,
		Subsample:                  1.0,
		SamplingMethod:             "uniform",
		ColSampleByTree:            1.0,
		ColSampleByLevel:           1.0,
		ColSampleByNode:            1.0,
		BaseScore:                  0.5, // 概率空间，回归直接使用，分类自动转换为 log-odds
		NumClass:                   1,
		Objective:                  ObjRegSquareError,
		Seed:                       0,
		Verbosity:                  1,
		MaxBin:                     256,
		ScalePosWeight:             1.0,
		TweedieVariancePower:       1.5,
		QuantileAlpha:              0.5,
		HuberSlope:                 1.0,
		SampleType:                 "uniform",
		NormalizeType:              "tree",
		ProcessType:                "default",
		LambdaRankPairMethod:       "topk",
		LambdaRankNumPairPerSample: 0,
		LambdaRankNormalization:    true,
		NDCGExpGain:                true,
		MaxCatToOneHot:             4,
		MaxCatThreshold:            64,
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
	if c.MaxDeltaStep < 0 {
		return errors.New("xgb: MaxDeltaStep must be non-negative")
	}
	if c.SamplingMethod != "" && c.SamplingMethod != "uniform" && c.SamplingMethod != "gradient_based" {
		return errors.New("xgb: SamplingMethod must be 'uniform' or 'gradient_based'")
	}
	if c.TweedieVariancePower < 1 || c.TweedieVariancePower > 2 {
		return errors.New("xgb: TweedieVariancePower must be in [1, 2]")
	}
	if c.QuantileAlpha < 0 || c.QuantileAlpha > 1 {
		return errors.New("xgb: QuantileAlpha must be in [0, 1]")
	}
	if c.HuberSlope <= 0 {
		return errors.New("xgb: HuberSlope must be positive")
	}
	if c.SampleType != "" && c.SampleType != "uniform" && c.SampleType != "weighted" {
		return errors.New("xgb: SampleType must be 'uniform' or 'weighted'")
	}
	if c.NormalizeType != "" && c.NormalizeType != "tree" && c.NormalizeType != "forest" {
		return errors.New("xgb: NormalizeType must be 'tree' or 'forest'")
	}
	if c.ProcessType != "" && c.ProcessType != "default" && c.ProcessType != "update" {
		return errors.New("xgb: ProcessType must be 'default' or 'update'")
	}
	if c.LambdaRankPairMethod != "" && c.LambdaRankPairMethod != "topk" && c.LambdaRankPairMethod != "mean" {
		return errors.New("xgb: LambdaRankPairMethod must be 'topk' or 'mean'")
	}
	return nil
}
