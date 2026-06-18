# xgb — Pure Go XGBoost-compatible Gradient Boosting

`xgb` 是 [XGBoost](https://github.com/dmlc/xgboost) 梯度提升决策树引擎的纯 Go 实现，与 C++ XGBoost 逻辑行为一致。

## 特性

### 核心算法
- **树方法**：hist（直方图，默认）和 exact（精确贪婪）
- **增长策略**：depthwise（层级遍历）和 lossguide（损失引导优先队列）
- **目标函数**：20 种（详见下方列表）
- **评估指标**：20 种（详见下方列表）
- **提升器**：gbtree（树提升）和 dart（带 Dropout 的树提升）

### 精度
- 全 `float64` 精度（梯度、直方图、叶值计算全部在 float64 中完成）
- 高于 XGBoost（内部用 float32）的数值精度

### 训练控制
- 学习率 (eta) · 最大深度 · 最小分裂增益 (gamma)
- L1/L2 正则化 (alpha / lambda) · 最小叶节点权重
- 行子采样 (subsample) · 列子采样 (colsample_bytree/level/node)
- 正类权重 (scale_pos_weight) · 单调约束 · 交互约束
- 早停 (early_stopping_rounds + validation set)
- 自定义目标函数 · 自定义评估函数

### 模型管理
- JSON 序列化（XGBoost 3.2 原生格式兼容）
- UBJSON 二进制序列化（XGBoost 字段布局）
- 继续训练 (ContinueTrain)
- 树结构文本导出 (DumpModel)
- Graphviz DOT 格式导出 (DumpModelDOT)

### 解释性
- TreeSHAP 值（完整 Shapley 值）
- SHAP 交互值（特征间交互归因）
- 近似 SHAP (ApproxSHAP，5x 加速)
- 特征重要性（weight / gain / cover）
- 偏依赖分析 (PartialDependence)

### 高级功能
- 交叉验证 (K-Fold + StratifiedKFold)
- 超参数搜索（网格搜索 GridSearch / 随机搜索 RandomSearch）
- 多目标/多输出 (MultiTargetModel，共享树结构)
- 外部内存训练 (ExternalTrain，逐块直方图累加)
- 类别特征自动编码 (CatEncoder)

---

## 快速开始

```go
package main

import (
    "fmt"
    "github.com/bambuo/xgb"
)

func main() {
    // 准备数据
    data := [][]float64{
        {1.0, 2.0},
        {2.0, 3.0},
        {3.0, 4.0},
        {4.0, 5.0},
    }
    labels := []float64{3.0, 5.0, 7.0, 9.0}

    dm, err := xgb.NewDMatrix(data, labels)
    if err != nil {
        panic(err)
    }

    // 配置模型
    cfg := xgb.DefaultConfig()
    cfg.NumTrees = 100
    cfg.MaxDepth = 6
    cfg.LearningRate = 0.3
    cfg.Objective = xgb.ObjRegSquareError

    // 训练
    gbt := xgb.NewGBTree(cfg)
    history, err := gbt.Train(dm, []xgb.Metric{&xgb.RMSEMetric{}})
    if err != nil {
        panic(err)
    }

    // 预测
    pred := gbt.Predict([]float64{5.0, 6.0})
    fmt.Printf("prediction: %.4f\n", pred)
    fmt.Printf("training RMSE: %.4f\n", history[len(history)-1].Metrics["rmse"])

    // SHAP 解释
    shap := gbt.SHAP([]float64{5.0, 6.0})
    fmt.Printf("SHAP base: %.4f\n", shap[2])

    // 保存模型
    if err := gbt.SaveModel("/tmp/model.json"); err != nil {
        panic(err)
    }
}
```

---

## 目标函数 (20 种)

| 目标 | 名称 | 域 |
|------|------|-----|
| `reg:squarederror` | SquaredError | 回归 |
| `reg:logistic` | LogisticRegression | 回归概率 |
| `binary:logistic` | LogisticRegression | 二分类 |
| `binary:logitraw` | LogisticRaw | 二分类原始输出 |
| `reg:gamma` | GammaRegression | Gamma 回归 |
| `reg:tweedie` | TweedieRegression | Tweedie 回归 |
| `reg:squaredlogerror` | SquaredLogError | 对数误差回归 |
| `reg:absoluteerror` | AbsoluteError | L1 回归 |
| `reg:pseudohubererror` | PseudoHuberError | 平滑 L1 回归 |
| `reg:quantileerror` | QuantileError | 分位数回归 |
| `binary:hinge` | BinaryHinge | SVM hinge 损失 |
| `count:poisson` | PoissonRegression | Poisson 回归 |
| `multi:softmax` | MulticlassSoftmax | 多分类 |
| `multi:softprob` | MulticlassSoftmax | 多分类概率 |
| `rank:ndcg` | RankNDCG | LambdaRank NDCG |
| `rank:map` | RankMAP | LambdaRank MAP |
| `rank:pairwise` | RankPairwise | 成对排序 |
| `survival:cox` | SurvivalCox | Cox 比例风险 |
| `survival:aft` | SurvivalAFT | 加速失效时间 |

---

## 评估指标 (20 种)

回归：`rmse` · `mae` · `rmsle` · `mape` · `mphe`
分类：`logloss` · `error` · `error@t` · `merror` · `mlogloss`
排序：`ndcg` · `map` · `pre`
统计：`poisson-nloglik` · `gamma-nloglik` · `gamma-deviance` · `tweedie-nloglik` · `cox-nloglik`
排序：`auc` · `aucpr`

---

## 交叉验证

```go
results, err := xgb.CV(cfg, dm, []xgb.Metric{&xgb.RMSEMetric{}}, 5, true, 42)
summary := xgb.SummarizeCV(results, "rmse")
fmt.Printf("CV RMSE: %.4f ± %.4f\n", summary.TestMean, summary.TestStd)

// 分层 K 折（用于分类）
// results, err = xgb.CVAdvanced(cfg, dm, metrics, 5, true, 42, true)
```

---

## 超参数搜索

```go
gc := xgb.GridSearchConfig{
    Params: []xgb.GridSearchParam{
        {Name: "learning_rate", Values: []interface{}{0.01, 0.1, 0.3}},
        {Name: "max_depth", Values: []interface{}{3, 6, 9}},
    },
    Metrics: []xgb.Metric{&xgb.RMSEMetric{}},
    NFolds:   3,
}
results, err := xgb.GridSearch(cfg, dm, gc)
// results[0] = 最优参数组合
```

---

## 外部内存训练

```go
dmc, _ := xgb.NewDMatrixChunked([]string{"chunk1.csv", "chunk2.csv"}, "csv", 10, 0)
gbt := xgb.NewGBTree(cfg)
history, err := gbt.ExternalTrain(dmc, nil)
```

---

## 多目标/多输出

```go
labels := [][]float64{
    {1.0, 2.0},
    {3.0, 4.0},
}
mt := xgb.NewMultiTargetModel(cfg, 2)
history, err := mt.Train(data, labels, nil)

preds := mt.Predict([]float64{5.0, 6.0}) // []float64{target0, target1}
```

---

## 序列化

```go
// JSON (XGBoost 原生格式)
gbt.SaveModel("/tmp/model.json")
loaded := xgb.LoadModel("/tmp/model.json")

// UBJSON (二进制)
gbt.SaveModelUBJSON("/tmp/model.ubj")
loaded, _ = xgb.LoadModelUBJSON("/tmp/model.ubj")
```

---

## 可视化

```go
// 树结构文本
dumps := gbt.DumpModel()
fmt.Println(dumps[0])

// Graphviz DOT 格式
dot := gbt.DumpModelDOT()
_ = dot

// ASCII 学习曲线
xgb.PrintLearningCurve(history, "rmse")

// ASCII 特征重要性
ranking := gbt.ImportanceRanking(xgb.ImportanceGain)
xgb.PrintImportance(ranking, 10, nil)
```

---

## 功能覆盖

```
163 ✅ 已实现 + C++ 行为一致
 10 ❌ 明确不可移植（GPU、分布式、Pickle、ONNX 等）
  5 ➖ 不在范围
```

### 不可移植清单

| 功能 | 原因 |
|------|------|
| GPU/CUDA | 纯 Go CPU 库 |
| 分布式训练 | Go 无 MPI/RDMA 绑定 |
| Python Pickle | Python 格式 |
| ONNX/PMML 导出 | 需外部 proto/XML 库 |
| 旧二进制格式 | XGBoost 1.x 已弃用 |
| 贝叶斯优化 | 需高斯过程库 |
| 位置去偏 | Web 搜索特定 |
| 模型可视化图像 | 需图形库 |
| 部分依赖图 | 需图形库 |
| 超参数重要性 | 可推导 |

---

## 性能

| 场景 | 500 样本 | 5000 样本 |
|------|----------|-----------|
| 训练 (50 trees, depth=6) | ~40 ms | ~190 ms |
| 预测 (50 trees) | ~1.7 μs/样本 | ~1.3 μs/样本 |
| SHAP (50 trees) | ~102 μs/样本 |
| ApproxSHAP (100 trees) | ~19 μs/样本（5x 加速） |

> 注：未来优化方向 — goroutine 并行化直方图构建（预期 5-20 倍提升）

---

## 与 XGBoost 的关系

- **算法逻辑一致**：核心算法（层级遍历、bin-based 分区、缺失值处理、TreeSHAP、DART）经 C++ 源码逐行比对
- **精度更高**：float64 vs XGBoost 的 float32
- **格式兼容**：JSON 序列化与 XGBoost 3.2 互读写
- **性能不同**：纯 Go 单线程，XGBoost 使用 OpenMP + GPU

---

## 许可证

Apache 2.0
