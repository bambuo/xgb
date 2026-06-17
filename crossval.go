package xgb

import (
	"fmt"
	"math"
	"math/rand"
)

// KFold 实现 K 折交叉验证的分割器。
// 与 XGBoost C++ 的 KFold 行为一致：先 shuffle（可选），然后按顺序切分成连续块。
type KFold struct {
	NSplits int
	Shuffle bool
	Seed    int64
}

// Split 返回训练/验证索引对，每对对应一折。
func (kf *KFold) Split(n int) []struct{ Train, Test []int } {
	if kf.NSplits < 2 {
		kf.NSplits = 5
	}
	if kf.NSplits > n {
		kf.NSplits = n
	}

	indices := make([]int, n)
	for i := range indices {
		indices[i] = i
	}

	if kf.Shuffle {
		rng := rand.New(rand.NewSource(kf.Seed))
		rng.Shuffle(n, func(i, j int) {
			indices[i], indices[j] = indices[j], indices[i]
		})
	}

	foldSize := n / kf.NSplits
	remainder := n % kf.NSplits

	splits := make([]struct{ Train, Test []int }, kf.NSplits)
	start := 0
	for i := 0; i < kf.NSplits; i++ {
		size := foldSize
		if i < remainder {
			size++
		}
		testIdx := make([]int, size)
		copy(testIdx, indices[start:start+size])

		trainIdx := make([]int, 0, n-size)
		trainIdx = append(trainIdx, indices[:start]...)
		trainIdx = append(trainIdx, indices[start+size:]...)

		splits[i] = struct{ Train, Test []int }{Train: trainIdx, Test: testIdx}
		start += size
	}
	return splits
}

// StratifiedKFold 实现分层 K 折交叉验证。
// 保持每个 fold 中类别比例与原始数据集一致。
// 对应 XGBoost C++ 的 stratified k-fold。
type StratifiedKFold struct {
	NSplits int
	Shuffle bool
	Seed    int64
}

// Split 返回分层后的训练/验证索引对。
func (skf *StratifiedKFold) Split(labels []float64) []struct{ Train, Test []int } {
	if skf.NSplits < 2 {
		skf.NSplits = 5
	}

	// 将样本按标签分组
	classIndices := make(map[float64][]int)
	for i, l := range labels {
		classIndices[l] = append(classIndices[l], i)
	}

	// 为每个类别独立创建 splits，然后合并
	type foldSplit struct{ train, test []int }
	perClassSplits := make([][]foldSplit, len(classIndices))

	classIdx := 0
	for _, indices := range classIndices {
		m := len(indices)
		// 打乱类别内的索引
		if skf.Shuffle {
			rng := rand.New(rand.NewSource(skf.Seed + int64(classIdx)*1000))
			rng.Shuffle(m, func(i, j int) {
				indices[i], indices[j] = indices[j], indices[i]
			})
		}

		// 按顺序切分成连续块
		foldSize := m / skf.NSplits
		remainder := m % skf.NSplits
		start := 0
		splits := make([]foldSplit, skf.NSplits)
		for i := 0; i < skf.NSplits; i++ {
			size := foldSize
			if i < remainder {
				size++
			}
			testIdx := make([]int, size)
			copy(testIdx, indices[start:start+size])
			trainIdx := make([]int, 0, m-size)
			trainIdx = append(trainIdx, indices[:start]...)
			trainIdx = append(trainIdx, indices[start+size:]...)
			splits[i] = foldSplit{train: trainIdx, test: testIdx}
			start += size
		}
		perClassSplits[classIdx] = splits
		classIdx++
	}

	// 合并所有类别的 splits
	merged := make([]struct{ Train, Test []int }, skf.NSplits)
	for i := 0; i < skf.NSplits; i++ {
		var trainAll, testAll []int
		for _, cs := range perClassSplits {
			trainAll = append(trainAll, cs[i].train...)
			testAll = append(testAll, cs[i].test...)
		}
		merged[i] = struct{ Train, Test []int }{Train: trainAll, Test: testAll}
	}
	return merged
}

// CVResult 保存单折交叉验证的结果。
type CVResult struct {
	Fold      int
	TrainEval map[string]float64 // 训练集最终指标
	TestEval  map[string]float64 // 验证集最终指标
}

// CV 运行 K 折交叉验证。
//
// 参数：
//   - cfg：模型配置（会被复制，不会修改原配置）
//   - dm：完整数据集
//   - metrics：评估指标列表
//   - nFolds：折数
//   - shuffle：是否在分割前打乱数据
//   - seed：随机种子
//   - stratified：是否使用分层抽样（保持类别比例）
//
// 返回每折的评估结果。
func CV(cfg *Config, dm *DMatrix, metrics []Metric, nFolds int, shuffle bool, seed int64) ([]CVResult, error) {
	return CVAdvanced(cfg, dm, metrics, nFolds, shuffle, seed, false)
}

// CVAdvanced 运行 K 折交叉验证（支持分层）。
func CVAdvanced(cfg *Config, dm *DMatrix, metrics []Metric, nFolds int, shuffle bool, seed int64, stratified bool) ([]CVResult, error) {
	var splits []struct{ Train, Test []int }
	if stratified {
		skf := &StratifiedKFold{NSplits: nFolds, Shuffle: shuffle, Seed: seed}
		splits = skf.Split(dm.Labels)
	} else {
		kf := &KFold{NSplits: nFolds, Shuffle: shuffle, Seed: seed}
		splits = kf.Split(dm.NumRows)
	}

	results := make([]CVResult, nFolds)
	for foldIdx, split := range splits {
		// 创建训练集和验证集的 DMatrix
		trainData := make([][]float64, len(split.Train))
		trainLabels := make([]float64, len(split.Train))
		trainWeights := make([]float64, len(split.Train))
		for i, idx := range split.Train {
			trainData[i] = dm.Data[idx]
			trainLabels[i] = dm.Labels[idx]
			if dm.Weights != nil {
				trainWeights[i] = dm.Weights[idx]
			} else {
				trainWeights[i] = 1.0
			}
		}
		trainDM, _ := NewDMatrix(trainData, trainLabels)
		if dm.Weights != nil {
			trainDM.SetWeights(trainWeights)
		}

		testData := make([][]float64, len(split.Test))
		testLabels := make([]float64, len(split.Test))
		for i, idx := range split.Test {
			testData[i] = dm.Data[idx]
			testLabels[i] = dm.Labels[idx]
		}
		testDM, _ := NewDMatrix(testData, testLabels)

		// 复制配置并训练
		foldCfg := *cfg
		foldCfg.Verbosity = 0 // 静默模式
		foldCfg.EvalData = nil
		foldCfg.EarlyStoppingRounds = 0

		gbt := NewGBTree(&foldCfg)
		history, err := gbt.Train(trainDM, metrics)
		if err != nil {
			return results, fmt.Errorf("fold %d training: %w", foldIdx, err)
		}

		// 评估
		trainPreds := gbt.PredictBatch(trainData)
		testPreds := gbt.PredictBatch(testData)

		result := CVResult{
			Fold:      foldIdx + 1,
			TrainEval: make(map[string]float64),
			TestEval:  make(map[string]float64),
		}

		// 使用历史最后一条记录作为训练指标
		if len(history) > 0 {
			last := history[len(history)-1]
			for name, val := range last.Metrics {
				result.TrainEval[name] = val
			}
		} else {
			// 回退：直接计算
			for _, m := range metrics {
				result.TrainEval[m.Name()] = m.Evaluate(trainPreds, trainDM)
			}
		}

		for _, m := range metrics {
			result.TestEval[m.Name()] = m.Evaluate(testPreds, testDM)
		}

		results[foldIdx] = result
	}

	return results, nil
}

// CVSummary 计算交叉验证结果的汇总统计。
type CVSummary struct {
	Metric       string
	TrainMean    float64
	TrainStd     float64
	TestMean     float64
	TestStd      float64
	PerFoldTrain []float64
	PerFoldTest  []float64
}

// Summarize 计算所有折的指标汇总。
func SummarizeCV(results []CVResult, metricName string) CVSummary {
	s := CVSummary{Metric: metricName}
	for _, r := range results {
		if v, ok := r.TrainEval[metricName]; ok {
			s.PerFoldTrain = append(s.PerFoldTrain, v)
			s.TrainMean += v
		}
		if v, ok := r.TestEval[metricName]; ok {
			s.PerFoldTest = append(s.PerFoldTest, v)
			s.TestMean += v
		}
	}
	n := float64(len(results))
	if n > 0 {
		s.TrainMean /= n
		s.TestMean /= n
	}
	// 标准差
	for _, v := range s.PerFoldTrain {
		d := v - s.TrainMean
		s.TrainStd += d * d
	}
	for _, v := range s.PerFoldTest {
		d := v - s.TestMean
		s.TestStd += d * d
	}
	if n > 0 {
		s.TrainStd = math.Sqrt(s.TrainStd / n)
		s.TestStd = math.Sqrt(s.TestStd / n)
	}
	return s
}
