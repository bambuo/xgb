package xgb

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
)

// makeDataset 生成回归测试数据集。
func makeDataset(n, nFeat int) ([][]float64, []float64) {
	data := make([][]float64, n)
	labels := make([]float64, n)
	rng := rand.New(rand.NewSource(42))
	for i := 0; i < n; i++ {
		data[i] = make([]float64, nFeat)
		var sum float64
		for j := 0; j < nFeat; j++ {
			data[i][j] = rng.Float64()
			sum += data[i][j] * float64(j+1)
		}
		labels[i] = sum
	}
	return data, labels
}

// BenchmarkTreeBuilding 测试树构建性能。
func BenchmarkTreeBuilding(b *testing.B) {
	for _, n := range []int{500, 1000} {
		for _, nFeat := range []int{5, 10} {
			b.Run(fmt.Sprintf("n=%d_feat=%d", n, nFeat), func(b *testing.B) {
				data, labels := makeDataset(n, nFeat)
				dm, _ := NewDMatrix(data, labels)
				grads := make([]float64, n)
				hess := make([]float64, n)
				for i := range grads {
					grads[i] = 0.5 - labels[i]
					hess[i] = 1.0
				}

				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					builder := NewExactBuilder(&ExactBuilderConfig{
						MaxDepth: 6, Gamma: 0, Lambda: 1.0,
						Alpha: 0, MinChildWeight: 1.0,
						NumFeatures: nFeat,
					})
					if err := builder.Build(dm, grads, hess, nil, nil); err != nil {
						b.Fatal(err)
					}
				}
			})
		}
	}
}

// BenchmarkTraining 测试端到端训练性能。
func BenchmarkTraining(b *testing.B) {
	for _, n := range []int{500, 5000} {
		b.Run(fmt.Sprintf("n=%d", n), func(b *testing.B) {
			data, labels := makeDataset(n, 10)
			dm, _ := NewDMatrix(data, labels)

			cfg := DefaultConfig()
			cfg.NumTrees = 50
			cfg.MaxDepth = 6
			cfg.LearningRate = 0.3
			cfg.Objective = ObjRegSquareError
			cfg.Verbosity = 0

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				gbt := NewGBTree(cfg)
				_, err := gbt.Train(dm, nil)
				if err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

// BenchmarkPrediction 测试预测性能。
func BenchmarkPrediction(b *testing.B) {
	for _, nTrees := range []int{50, 200} {
		for _, nSamples := range []int{100, 10000} {
			b.Run(fmt.Sprintf("trees=%d_samples=%d", nTrees, nSamples), func(b *testing.B) {
				// 训练模型
				data, labels := makeDataset(500, 10)
				dm, _ := NewDMatrix(data, labels)

				cfg := DefaultConfig()
				cfg.NumTrees = nTrees
				cfg.MaxDepth = 6
				cfg.LearningRate = 0.3
				cfg.Verbosity = 0

				gbt := NewGBTree(cfg)
				_, _ = gbt.Train(dm, nil)

				// 测试数据
				testData, _ := makeDataset(nSamples, 10)

				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_ = gbt.PredictBatch(testData)
				}
			})
		}
	}
}

// BenchmarkGBLinear 测试线性模型性能。
func BenchmarkGBLinear(b *testing.B) {
	data, labels := makeDataset(1000, 20)
	dm, _ := NewDMatrix(data, labels)

	cfg := DefaultConfig()
	cfg.NumTrees = 100
	cfg.LearningRate = 0.1
	cfg.Verbosity = 0

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		gl := NewGBLinear(cfg)
		_, err := gl.Train(dm, nil)
		if err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkSHAP 测试 SHAP 值计算性能。
func BenchmarkSHAP(b *testing.B) {
	data, labels := makeDataset(500, 10)
	dm, _ := NewDMatrix(data, labels)

	cfg := DefaultConfig()
	cfg.NumTrees = 50
	cfg.MaxDepth = 6
	cfg.Verbosity = 0

	gbt := NewGBTree(cfg)
	_, _ = gbt.Train(dm, nil)

	sample := make([]float64, 10)
	for i := range sample {
		sample[i] = float64(i) / 10.0
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = gbt.SHAP(sample)
	}
}

// BenchmarkApproxSHAP 测试近似 SHAP 性能。
func BenchmarkApproxSHAP(b *testing.B) {
	data, labels := makeDataset(500, 10)
	dm, _ := NewDMatrix(data, labels)

	cfg := DefaultConfig()
	cfg.NumTrees = 100
	cfg.MaxDepth = 6
	cfg.Verbosity = 0

	gbt := NewGBTree(cfg)
	_, _ = gbt.Train(dm, nil)

	sample := make([]float64, 10)
	for i := range sample {
		sample[i] = float64(i) / 10.0
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = gbt.ApproxSHAP(sample)
	}
}

// BenchmarkMemoryAllocation 测试内存分配。
func BenchmarkMemoryAllocation(b *testing.B) {
	// 模拟每轮迭代的分配模式
	n := 10000
	nFeat := 50

	data, labels := makeDataset(n, nFeat)
	dm, _ := NewDMatrix(data, labels)

	cfg := DefaultConfig()
	cfg.NumTrees = 5
	cfg.MaxDepth = 4
	cfg.Verbosity = 0

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		gbt := NewGBTree(cfg)
		_, _ = gbt.Train(dm, nil)
		_ = gbt.PredictBatch(data)
	}
	b.ReportMetric(float64(b.N)*float64(cfg.NumTrees), "trees_total")
}

// BenchmarkCrossValidation 测试交叉验证性能。
func BenchmarkCrossValidation(b *testing.B) {
	data, labels := makeDataset(200, 5)
	dm, _ := NewDMatrix(data, labels)

	cfg := DefaultConfig()
	cfg.NumTrees = 20
	cfg.MaxDepth = 4
	cfg.Verbosity = 0

	for i := 0; i < b.N; i++ {
		_, err := CV(cfg, dm, []Metric{&RMSEMetric{}}, 3, false, 42)
		if err != nil {
			b.Fatal(err)
		}
	}
}

// TestLargeDataset 集成测试：大量数据的训练和预测。
func TestLargeDataset(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping large dataset test in short mode")
	}

	n, nFeat := 50000, 20
	data, labels := makeDataset(n, nFeat)
	dm, _ := NewDMatrix(data, labels)

	cfg := DefaultConfig()
	cfg.NumTrees = 100
	cfg.MaxDepth = 8
	cfg.Subsample = 0.8
	cfg.ColSampleByTree = 0.8
	cfg.Objective = ObjRegSquareError
	cfg.Verbosity = 0

	gbt := NewGBTree(cfg)
	history, err := gbt.Train(dm, []Metric{&RMSEMetric{}})
	if err != nil {
		t.Fatalf("Train error: %v", err)
	}

	if len(history) != cfg.NumTrees {
		t.Errorf("expected %d history, got %d", cfg.NumTrees, len(history))
	}

	// 检查 RMSE 收敛
	firstRMSE := history[0].Metrics["rmse"]
	lastRMSE := history[len(history)-1].Metrics["rmse"]
	if lastRMSE >= firstRMSE {
		t.Errorf("RMSE did not converge: first=%f last=%f", firstRMSE, lastRMSE)
	}

	// 批量预测
	preds := gbt.PredictBatch(data)
	if len(preds) != n {
		t.Errorf("expected %d preds, got %d", n, len(preds))
	}

	// 验证预测值非 NaN
	for i, p := range preds {
		if math.IsNaN(p) {
			t.Errorf("NaN prediction at %d", i)
			break
		}
	}
}
