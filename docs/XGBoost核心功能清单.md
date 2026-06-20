# XGBoost 核心功能清单

> 本文档作为 xgb 库开发实现的对照参考和验证依据。
> 标注说明：✅ = 已实现 | ⚠️ = 部分实现 | ❌ = 未实现 | ➖ = 不在实现范围

---

## 1. 提升器类型（Booster）

| 功能 | XGBoost 原生 | 当前实现 | 状态 | 备注 |
|------|-------------|---------|------|------|
| gbtree（树提升器） | 支持 | GBTree | ✅ | 核心实现 |
| dart（Dropouts meet Multiple Additive Regression Trees） | 支持 | GBTree + selectDroppedTrees | ✅ | 完整实现：rate_drop / one_drop / skip_drop |
| gblinear（线性提升器） | 支持 | GBLinear | ✅ | 坐标下降法 |

---

## 2. 通用参数（General Parameters）

| 参数 | XGBoost 默认值 | 当前实现 | 状态 | 备注 |
|------|---------------|---------|------|------|
| booster | gbtree | BoostType | ✅ | gbtree / dart 可用 |
| verbosity | 1 | Verbosity (0-3) | ✅ | |
| nthread | 自动 | — | ➖ | Go 原生并发，非 OpenMP |
| validate_parameters | false | Config.ValidateParams | ✅ | 未知参数校验 |
| disable_default_eval_metric | false | Config.DisableDefaultEvalMetric | ✅ | |

---

## 3. 树提升器参数（Tree Booster Parameters）

### 3.1 核心训练参数

| 参数 | XGBoost 默认值 | Config 字段 | 状态 | 备注 |
|------|---------------|------------|------|------|
| eta / learning_rate | 0.3 | LearningRate | ✅ | |
| n_estimators / num_boost_round | 100 | NumTrees | ✅ | |
| max_depth | 6 | MaxDepth | ✅ | |
| min_child_weight | 1 | MinChildWeight | ✅ | |
| gamma / min_split_loss | 0 | Gamma | ✅ | |
| lambda / reg_lambda | 1 | Lambda | ✅ | L2 正则化 |
| alpha / reg_alpha | 0 | Alpha | ✅ | L1 正则化 |
| base_score | 0.5（原生自动估计） | BaseScore / BoostFromAverage | ✅ | 支持自动估计（boost_from_average） |
| max_delta_step | 0 | Config.MaxDeltaStep | ✅ | 叶节点输出约束 |

### 3.2 子采样参数

| 参数 | XGBoost 默认值 | Config 字段 | 状态 | 备注 |
|------|---------------|------------|------|------|
| subsample | 1 | Subsample | ✅ | 行采样 |
| sampling_method | uniform | Config.SamplingMethod | ✅ | uniform / gradient_based 均支持 |
| colsample_bytree | 1 | ColSampleByTree | ✅ | 树级列采样 |
| colsample_bylevel | 1 | ColSampleByLevel | ✅ | 每层级列采样 |
| colsample_bynode | 1 | ColSampleByNode | ✅ | 每节点列采样 |
| feature_weights | — | Config.FeatureWeights | ✅ | DMatrix 特征权重 |

### 3.3 树构建算法

| 参数 | XGBoost 选项 | 当前实现 | 状态 | 备注 |
|------|-------------|---------|------|------|
| tree_method | auto/exact/approx/hist | HistBuilder + ExactBuilder | ✅ | hist 和 exact 两种均完整实现 |
| grow_policy | depthwise/lossguide | depthwise + lossguide | ✅ | MaxLeaves > 0 时自动启用 lossguide（基于堆的优先队列） |
| max_leaves | 0 | Config.MaxLeaves | ✅ | 与 XGBoost 行为一致：> 0 启用 lossguide 增长 |
| max_bin | 256 | Config.MaxBin | ✅ | 可配置，默认 256 |
| num_parallel_tree | 1 | Config.NumParallelTree | ✅ | 随机森林模式：bootstrap 采样（mt19937）+ 每棵树独立列采样 |

### 3.4 高级树参数

| 参数 | 说明 | 状态 | 备注 |
|------|------|------|------|
| monotone_constraints | 单调性约束 | ✅ | Config.MonotoneConstraints |
| interaction_constraints | 特征交互约束 | ✅ | Config.InteractionConstraints |
| updater | 树更新器序列 | ➖ | 内部自动选择 |
| refresh_leaf | 叶节点刷新 | ✅ | Config.RefreshLeaf + refreshLeafValues()，构建后重算叶值 |
| process_type | default/update | ✅ | Config.ProcessType，Validate() 已检查取值范围 |
| max_cached_hist_node | 直方图缓存节点数 | ✅ | Config.MaxCachedHistNode + ExactBuilderConfig.MaxCachedHistNode |

### 3.5 类别特征参数

| 参数 | 说明 | 状态 | 备注 |
|------|------|------|------|
| max_cat_to_onehot | one-hot 编码阈值 | ✅ | Config.MaxCatToOneHot + CatEncoder.OneHotEncode |
| max_cat_threshold | 分类分裂最大类别数 | ✅ | ExactBuilder 类别梯度比分组分裂 |

---

## 4. 目标函数（Objective Functions）

### 4.1 回归目标

| 目标 | XGBoost 字符串 | 当前实现 | 状态 | 备注 |
|------|---------------|---------|------|------|
| 平方误差 | reg:squarederror | SquaredError | ✅ | |
| 平方对数误差 | reg:squaredlogerror | SquaredLogError | ✅ | |
| 逻辑回归 | reg:logistic | LogisticRegression | ✅ | |
| Pseudo Huber | reg:pseudohubererror | PseudoHuberError | ✅ | 斜率 delta 可配置 |
| 绝对误差 | reg:absoluteerror | AbsoluteError | ✅ | L1 梯度近似 |
| 分位数回归 | reg:quantileerror | QuantileError | ✅ | alpha 分位数参数可配置 |
| Gamma 回归 | reg:gamma | GammaRegression | ✅ | log 链接 |
| Tweedie 回归 | reg:tweedie | TweedieRegression | ✅ | 方差幂参数可配置 |

### 4.2 分类目标

| 目标 | XGBoost 字符串 | 当前实现 | 状态 | 备注 |
|------|---------------|---------|------|------|
| 二分类逻辑回归 | binary:logistic | LogisticRegression | ✅ | |
| 二分类原始分数 | binary:logitraw | LogisticRaw | ✅ | |
| 二分类 hinge | binary:hinge | BinaryHinge | ✅ | |
| 多分类 softmax | multi:softmax | MulticlassSoftmax | ✅ | |
| 多分类 softprob | multi:softprob | MulticlassSoftmax | ✅ | 共用实现 |

### 4.3 排序目标

| 目标 | XGBoost 字符串 | 状态 | 备注 |
|------|---------------|------|------|
| LambdaMART NDCG | rank:ndcg | ✅ | 需设置 Group 信息 |
| LambdaMART MAP | rank:map | ✅ | 需设置 Group 信息 |
| LambdaRank pairwise | rank:pairwise | ✅ | 支持分组与无分组模式 |

### 4.4 计数与生存分析目标

| 目标 | XGBoost 字符串 | 状态 | 备注 |
|------|---------------|------|------|
| Poisson 回归 | count:poisson | ✅ | log 链接 |
| Cox 回归 | survival:cox | SurvivalCox | ✅ | Efron 近似偏似然 |
| AFT 生存分析 | survival:aft | SurvivalAFT | ✅ | 参数化 AFT 模型（对数正态/逻辑/极值分布） |

### 4.5 目标函数附属参数

| 参数 | 说明 | 状态 | 备注 |
|------|------|------|------|
| scale_pos_weight | 正样本权重比 | ✅ | 二分类类别不平衡 |
| tweedie_variance_power | Tweedie 方差幂 | ✅ | Config.TweedieVariancePower（默认 1.5） |
| huber_slope | Pseudo Huber 斜率 | ✅ | Config.HuberSlope（默认 1.0） |
| quantile_alpha | 分位数 alpha | ✅ | Config.QuantileAlpha（默认 0.5） |
| aft_loss_distribution | AFT 分布类型 | ✅ | SurvivalAFT.Distribution（normal/logistic/extreme） |

---

## 5. 评估指标（Evaluation Metrics）

| 指标 | XGBoost 字符串 | 当前实现 | 状态 | 备注 |
|------|---------------|---------|------|------|
| 均方根误差 | rmse | RMSEMetric | ✅ | |
| 均方根对数误差 | rmsle | RMSLEMetric | ✅ | |
| 平均绝对误差 | mae | MAEMetric | ✅ | |
| 平均绝对百分比误差 | mape | MAPEMetric | ✅ | |
| 平均 Pseudo Huber 误差 | mphe | MPHMetric | ✅ | delta 可配置 |
| 对数损失 | logloss | LogLossMetric | ✅ | |
| 分类错误率 | error | ErrorMetric | ✅ | |
| 自定义阈值错误率 | error@t | ErrorAtMetric | ✅ | 阈值可配置 |
| 多分类错误率 | merror | MErrorMetric | ✅ | |
| 多分类对数损失 | mlogloss | MLogLossMetric | ✅ | |
| ROC 曲线下面积 | auc | AUCMetric | ✅ | |
| PR 曲线下面积 | aucpr | AUCPRMetric | ✅ | 插值平均精度 |
| NDCG | ndcg | NDCGMetric | ✅ | NDCG@k（k 可配置） |
| MAP | map | MAPMetric | ✅ | Mean Average Precision@k |
| Precision@k | pre | PrecisionMetric | ✅ | Precision@k |
| Poisson 负对数似然 | poisson-nloglik | PoissonNLogLikMetric | ✅ | |
| Gamma 负对数似然 | gamma-nloglik | GammaNLogLikMetric | ✅ | |
| Cox 负偏对数似然 | cox-nloglik | CoxNLogLikMetric | ✅ | |
| Gamma 偏差 | gamma-deviance | GammaDevianceMetric | ✅ | |
| Tweedie 负对数似然 | tweedie-nloglik | TweedieNLogLikMetric | ✅ | |

---

## 6. 数据管理（DMatrix）

| 功能 | 说明 | 状态 | 备注 |
|------|------|------|------|
| 稠密矩阵创建 | NewDMatrix | ✅ | |
| CSR 稀疏矩阵创建 | NewDMatrixFromCSR | ✅ | 压缩稀疏行格式 |
| 样本权重 | SetWeights | ✅ | |
| 基础边距 | SetBaseMargin | ✅ | |
| 缺失值处理（NaN） | HasMissing | ✅ | 树预测时默认方向 |
| 分组信息（排序用） | SetGroup | ✅ | rank:ndcg 等排序目标使用 |
| 从文件加载 | LoadDMatrixFromCSV / LoadDMatrixFromLibSVM | ✅ | CSV 和 LibSVM 格式 |
| 特征名称/类型 | SetFeatureNames / SetFeatureTypes | ✅ | |
| 外部内存模式 | DMatrixChunked + ExternalTrain() | ✅ | 分块加载（CSV/LibSVM），跨块梯度累积 |

---

## 7. 训练功能

### 7.1 基础训练

| 功能 | 说明 | 状态 | 备注 |
|------|------|------|------|
| 完整训练 | Train() | ✅ | |
| 继续训练 | ContinueTrain() | ✅ | 保留已有树 |
| 梯度计算 | GetGradient() | ✅ | 一阶+二阶 |
| 预测更新 | 逐轮更新 preds | ✅ | |
| 多分类支持 | 按类别轮建树 | ✅ | |

### 7.2 交叉验证

| 功能 | 说明 | 状态 | 备注 |
|------|------|------|------|
| K 折交叉验证 | CV() / CVAdvanced() | ✅ | 支持 shuffle 和自定义折数 |
| 分层 K 折 | StratifiedKFold | ✅ | 按类别比例分配各 fold（CVAdvanced 参数） |
| 交叉验证结果返回 | CVResult + SummarizeCV | ✅ | 每折指标 + 均值/标准差 |

### 7.3 早停（Early Stopping）

| 功能 | 说明 | 状态 | 备注 |
|------|------|------|------|
| 验证集监控 | Config.EvalData | ✅ | |
| 早停轮数 | Config.EarlyStoppingRounds | ✅ | |
| 最佳迭代保存 | bestIteration | ✅ | |
| 最佳分数记录 | bestEvalScore | ✅ | |

### 7.4 训练监控与回调

| 功能 | 说明 | 状态 | 备注 |
|------|------|------|------|
| 定期打印指标 | 每 10 轮 | ✅ | Verbosity >= 2 |
| 训练历史记录 | Train() 返回 []EvalResult | ✅ | 含每轮所有指标的结构化历史 |
| 自定义回调 | Config.Callbacks []CallbackFunc | ✅ | 每轮后调用，可提前停止 |
| 训练日志重定向 | Config.LogWriter io.Writer | ✅ | nil = stdout |

---

## 8. 推理/预测（Prediction）

| 功能 | 说明 | 状态 | 备注 |
|------|------|------|------|
| 单样本预测 | Predict() | ✅ | 原始分数 |
| 概率预测 | PredictProb() | ✅ | sigmoid 变换 |
| 批量预测 | PredictBatch() | ✅ | |
| 叶节点索引 | GetLeafIndex() | ✅ | |
| PredTransform | 目标函数变换 | ✅ | 通过 Objective 接口 |
| iteration_range | 指定树范围预测 | ✅ | Predict / PredictBatch / GetLeafIndex 可选参数 |
| 多分类概率输出 | softprob 输出 | ✅ | PredTransform 支持 |
| 边际贡献 | pred_contribs / SHAP 值 | ✅ | GBTree.SHAP() — TreeSHAP 算法 |
| 近似边际贡献 | approx_contribs / ApproxSHAP | ✅ | GBTree.ApproxSHAP() — 路径平均算法，与 XGBoost approx_contribs 行为一致 |
| 特征交互 | pred_interactions / SHAP 交互值 | ✅ | GBTree.SHAPInteraction() — 交互值矩阵 |

---

## 9. 特征重要性分析（Feature Importance）

| 功能 | 说明 | 状态 | 备注 |
|------|------|------|------|
| 分裂频率（weight） | 特征被用于分裂的次数 | ✅ | GetFScore() / GetScore(ImportanceWeight) |
| 增益（gain） | 特征分裂的总增益 | ✅ | GetScore(ImportanceGain) |
| 覆盖度（cover） | 特征分裂的总覆盖样本数 | ✅ | GetScore(ImportanceCover) |
| SHAP 值 | SHapley Additive exPlanations | ✅ | GBTree.SHAP() — TreeSHAP（coef 算法） |
| SHAP 交互值 | 特征间 Shapley 交互 | ✅ | GBTree.SHAPInteraction() |
| 特征重要性排序数据 | ImportanceRanking() | ✅ | 返回排序后的 []FeatureImportance（可直接绘图） |
| 偏依赖数据 | PartialDependence() | ✅ | 返回 []PDPPoint 含置信区间（可直接绘图） |

---

## 10. 模型序列化与持久化

### 10.1 保存格式

| 功能 | 说明 | 状态 | 备注 |
|------|------|------|------|
| JSON 原生格式 | SaveModel() | ✅ | XGBoost 3.2 兼容 |
| 旧格式（legacy） | 自定义 JSON | ✅ | 向后兼容 |
| UBJSON 二进制 | SaveModelUBJSON / LoadModelUBJSON | ✅ | XGBoost 原生 UBJSON 字段布局 |
| Pickle（Python） | — | ➖ | Python 专属 |
| 旧二进制格式 | — | ➖ | XGBoost 1.x 已弃用 |

### 10.2 加载功能

| 功能 | 说明 | 状态 | 备注 |
|------|------|------|------|
| 加载原生 JSON | LoadModel() | ✅ | |
| 加载旧格式 | loadLegacy() | ✅ | |
| 自动格式检测 | — | ✅ | |
| 学习率缩放还原 | 撤销保存时的 LR 缩放 | ✅ | |
| base_score 还原 | logit 空间转换 | ✅ | |

### 10.3 模型导出

| 功能 | 说明 | 状态 | 备注 |
|------|------|------|------|
| 树结构文本 dump | DumpModel() 文本格式 | ✅ | 见第 18 节 |
| 树结构可视化 DOT | DumpModelDOT() Graphviz 格式 | ✅ | 见第 18 节 |
| 导出 ONNX | — | ❌ | 需外部 protobuf 库 |
| 导出 PMML | — | ❌ | 需外部 XML schema 库 |

---

## 11. 超参数调优

| 功能 | 说明 | 状态 | 备注 |
|------|------|------|------|
| 配置验证 | Config.Validate() | ✅ | 范围检查 |
| 默认配置 | DefaultConfig() | ✅ | |
| 网格搜索 | GridSearch() | ✅ | 暴力搜索参数组合 + CV 评估 |
| 随机搜索 | RandomSearch() | ✅ | 随机采样参数空间 + CV 评估 |
| 贝叶斯优化 | — | ❌ | 需高斯过程库，非 Go 生态范围 |

---

## 12. DART 提升器参数

| 参数 | XGBoost 默认值 | 状态 | 备注 |
|------|---------------|------|------|
| sample_type | uniform | ✅ | Config.SampleType：uniform / weighted |
| normalize_type | tree | ✅ | Config.NormalizeType：tree / forest |
| rate_drop | 0.0 | ✅ | Config.RateDrop |
| one_drop | 0 | ✅ | Config.OneDrop（布尔） |
| skip_drop | 0.0 | ✅ | Config.SkipDrop |

---

## 13. 线性提升器参数（gblinear）

| 参数 | XGBoost 默认值 | 状态 | 备注 |
|------|---------------|------|------|
| updater | shotgun / coord_descent | ✅ | Config 未暴露，代码内部自动选择（shotgun 随机顺序 / coord_descent 固定顺序） |
| feature_selector | cyclic | ✅ | 默认 cyclic（固定顺序），shotgun 模式下为 randomized |
| top_k | 0 | ➖ | 仅 greedy/thrifty 模式，与本实现无关 |
| lambda | 0.0 | ✅ | Config.Lambda |
| alpha | 0.0 | ✅ | Config.Alpha |
| eta | 0.5 | ✅ | Config.LearningRate |

---

## 14. 排序任务参数（Learning to Rank）

| 参数 | 说明 | 状态 | 备注 |
|------|------|------|------|
| lambdarank_pair_method | topk / mean | ✅ | RankNDCG.PairMethod，Config.LambdaRankPairMethod |
| lambdarank_num_pair_per_sample | 采样对数 | ✅ | RankNDCG.NumPairPerSample，Config.LambdaRankNumPairPerSample（0 = 全部） |
| lambdarank_normalization | true | ✅ | RankNDCG.Normalization，Config.LambdaRankNormalization |
| lambdarank_score_normalization | true | ❌ | 保留，需验证 C++ 实现细节 |
| lambdarank_unbiased | false | ❌ | 特定于点击数据，不可移植 |
| lambdarank_bias_norm | 2.0 | ❌ | 同上 |
| ndcg_exp_gain | true | ✅ | RankNDCG.NDCGExpGain，Config.NDCGExpGain |

---

## 15. GPU 加速

| 功能 | 说明 | 状态 | 备注 |
|------|------|------|------|
| CUDA GPU 训练 | device=cuda | ➖ | 纯 Go CPU 实现 |
| GPU 直方图算法 | grow_gpu_hist | ➖ | |
| GPU 近似算法 | grow_gpu_approx | ➖ | |
| 多 GPU 支持 | — | ➖ | |

---

## 16. 分布式训练

| 功能 | 说明 | 状态 | 备注 |
|------|------|------|------|
| 分布式提升 | Rabit | ➖ | |
| 联邦学习 | — | ➖ | |
| Spark/Flink 集成 | — | ➖ | |

---

## 17. 高级功能

| 功能 | 说明 | 状态 | 备注 |
|------|------|------|------|
| 自定义目标函数 | 用户传入梯度/Hessian | ✅ | CustomObjFunc + CustomObj 包装 Objective 接口 |
| 自定义评估指标 | 用户传入评估函数 | ✅ | CustomEvalFunc + CustomMetric 包装 Metric 接口 |
| 增量学习 | 从已有模型继续 | ✅ | ContinueTrain |
| K 折交叉验证 | CV() | ✅ | 支持 shuffle 和自定义折数 |
| 网格搜索 | GridSearch() | ✅ | 暴力搜索参数组合 + CV 评估 |
| 随机搜索 | RandomSearch() | ✅ | 随机采样参数空间 + CV 评估 |
| 线性提升器 | GBLinear | ✅ | 坐标下降法，支持 L1/L2 正则化 |
| 类别特征 | CatEncoder | ✅ | 自动检测 + one-hot 编码 |
| 模型融合 | 多模型合并 | ❌ | |
| 多目标/多输出 | MultiTargetModel | ✅ | 每个目标独立 GBTree，支持 PredictBatch 返回矩阵 |
| 外部内存训练 | ExternalTrain() + DMatrixChunked | ✅ | 分块文件加载 + 跨块梯度累积 |
| 位置去偏 | — | ❌ | 点击数据 |

---

## 18. 辅助工具与可视化

| 功能 | 说明 | 状态 | 备注 |
|------|------|------|------|
| 树结构 dump | DumpModel() 文本格式 | ✅ | 含 gain/cover，支持特征名 |
| 树可视化 DOT | DumpModelDOT() Graphviz 格式 | ✅ | 可直接用 graphviz 工具渲染 |
| 学习曲线 | PlotLearningCurve() ASCII 图表 + LearningCurve() 数据 | ✅ | 终端图表，数据取自 []EvalResult |
| 特征重要性图 | PlotImportance() ASCII 条形图 + ImportanceRanking() 数据 | ✅ | 终端图表，支持特征名标注 |
| 部分依赖图 | PDP | ❌ | PartialDependence() 提供数据，绘图需用户处理 |
| 超参数重要性 | — | ❌ | |
| UBJSON 序列化 | SaveModelUBJSON / LoadModelUBJSON | ✅ | XGBoost 原生字段布局 |
| CSV/LibSVM 加载 | LoadDMatrixFromCSV / LoadDMatrixFromLibSVM | ✅ | 文件 I/O |

---

## 19. 当前实现统计摘要

| 类别 | 已实现 | 部分实现 | 未实现 | 不在范围 |
|------|--------|---------|--------|---------|
| 提升器类型 | 3 | 0 | 0 | 0 |
| 通用参数 | 4 | 0 | 0 | 1 |
| 树参数（核心） | 9 | 0 | 0 | 0 |
| 树参数（子采样） | 6 | 0 | 0 | 0 |
| 树参数（构建算法） | 5 | 0 | 0 | 0 |
| 树参数（高级） | 5 | 0 | 0 | 1 |
| 树参数（类别） | 2 | 0 | 0 | 0 |
| 目标函数 | 24 | 0 | 0 | 0 |
| 评估指标 | 20 | 0 | 0 | 0 |
| 数据管理 | 9 | 0 | 0 | 0 |
| 训练功能 | 16 | 0 | 0 | 0 |
| 推理功能 | 10 | 0 | 0 | 0 |
| 特征重要性 | 7 | 0 | 0 | 0 |
| 模型序列化 | 9 | 0 | 2 | 2 |
| 超参数调优 | 4 | 0 | 1 | 0 |
| DART 参数 | 5 | 0 | 0 | 0 |
| 线性提升器参数 | 5 | 0 | 0 | 1 |
| 排序参数 | 4 | 0 | 3 | 0 |
| 高级功能 | 10 | 0 | 2 | 0 |
| 辅助工具 | 6 | 0 | 2 | 0 |
| **合计** | **163** | **0** | **10** | **5** |

---

## 20. 明确不可移植功能及原因

> 所有可移植功能（共 **163 项 ✅**）均已实现且与 C++ XGBoost 逻辑行为对齐。
> 以下 **10 项 ❌ + 5 项 ➖** 因技术原因明确不可移植。

### 依赖外部运行时 / 硬件（4 项 ➖）

| 功能 | 类别 | 原因 |
|------|------|------|
| GPU/CUDA 加速 | 硬件依赖 | 纯 Go CPU 库，无 CUDA 运行时 |
| 分布式训练（Rabit） | 运行时依赖 | Go 无 MPI/RDMA 绑定 |
| ONNX / PMML 导出 | 外部库依赖 | 需 protobuf / XML schema 库 |
| Python Pickle | 语言专属 | Python 对象序列化 |

### 仅 C++ 实现的功能（4 项 ❌）

| 功能 | 类别 | 原因 |
|------|------|------|
| **旧二进制格式** | 已弃用 | XGBoost 1.x 格式，官方推荐 JSON/UBJSON |
| **贝叶斯超参数优化** | 算法库依赖 | 需要高斯过程库，Go 生态缺失。替代方案：GridSearch / RandomSearch |
| **lambdarank_unbiased** | 应用特定 | Web 搜索点击去偏算法，非通用梯度提升功能 |
| **lambdarank_bias_norm** | 应用特定 | 同上 |

### 超出梯度提升库范畴（5 项 ❌ + 1 项 ➖）

| 功能 | 类别 | 原因 |
|------|------|------|
| **模型融合（Model Ensemble）** | 超范围 | Stacking/加权融合是上层策略，非核心库职能 |
| **位置去偏（Position Debias）** | 应用特定 | 非通用梯度提升功能 |
| **lambdarank_score_normalization** | 需验证 | 保留待后续与 C++ 比对细节 |
| **模型可视化图像** | 需图形库 | 数据已由 `DumpModelDOT()` / `ImportanceRanking()` / `PlotLearningCurve()` 返回 |
| **部分依赖图（PDP）** | 需图形库 | 数据已由 `PartialDependence()` 返回 |
| **超参数重要性** | 需多次训练 | 可从 `GridSearch` 结果推导 |
