package xgb

import (
	"math"
	"os"
	"testing"
)

func TestGBTree_TrainRegression(t *testing.T) {
	// 简单回归：y = 2*x1 + 3*x2
	n := 200
	data := make([][]float64, n)
	labels := make([]float64, n)
	for i := 0; i < n; i++ {
		data[i] = make([]float64, 2)
		data[i][0] = float64(i%10) / 10.0
		data[i][1] = float64((i*7)%10) / 10.0
		labels[i] = 2*data[i][0] + 3*data[i][1]
	}

	dm, err := NewDMatrix(data, labels)
	if err != nil {
		t.Fatalf("NewDMatrix: %v", err)
	}

	cfg := DefaultConfig()
	cfg.NumTrees = 20
	cfg.MaxDepth = 4
	cfg.LearningRate = 0.3
	cfg.Objective = ObjRegSquareError
	cfg.Verbosity = 0

	gbt := NewGBTree(cfg)
	err = gbt.Train(dm, nil)
	if err != nil {
		t.Fatalf("Train: %v", err)
	}

	if len(gbt.Trees) != 20 {
		t.Errorf("expected 20 trees, got %d", len(gbt.Trees))
	}

	// 测试预测质量
	preds := gbt.PredictBatch(data)
	rmse := (&RMSEMetric{}).Evaluate(preds, dm)
	if rmse > 0.5 {
		t.Errorf("RMSE too high: %f", rmse)
	}
}

func TestGBTree_TrainBinary(t *testing.T) {
	// 二分类：可分离数据
	n := 200
	data := make([][]float64, n)
	labels := make([]float64, n)
	for i := 0; i < n; i++ {
		data[i] = make([]float64, 2)
		data[i][0] = float64(i%20) / 10.0
		data[i][1] = float64((i*11)%20) / 10.0
		// label = 1 if x0 + x1 > 1.0 else 0
		if data[i][0]+data[i][1] > 1.0 {
			labels[i] = 1
		} else {
			labels[i] = 0
		}
	}

	dm, err := NewDMatrix(data, labels)
	if err != nil {
		t.Fatalf("NewDMatrix: %v", err)
	}

	cfg := DefaultConfig()
	cfg.NumTrees = 20
	cfg.MaxDepth = 4
	cfg.LearningRate = 0.3
	cfg.Objective = ObjBinaryLogistic
	cfg.Verbosity = 0

	gbt := NewGBTree(cfg)
	err = gbt.Train(dm, nil)
	if err != nil {
		t.Fatalf("Train: %v", err)
	}

	// 检查预测值为概率
	for _, row := range data {
		p := gbt.PredictProb(row)
		if p < 0 || p > 1 {
			t.Errorf("predict prob out of range: %f", p)
		}
	}
}

func TestGBTree_PredictProb(t *testing.T) {
	cfg := DefaultConfig()
	cfg.NumTrees = 5
	cfg.Objective = ObjBinaryLogistic
	gbt := NewGBTree(cfg)

	// 空树：predict = base_score (=0), sigmoid(0) = 0.5
	p := gbt.PredictProb([]float64{1.0, 2.0})
	if math.Abs(p-0.5) > 1e-10 {
		t.Errorf("expected 0.5, got %f", p)
	}
}

func TestGBTree_SaveLoad(t *testing.T) {
	// 训练一个小模型
	n := 100
	data := make([][]float64, n)
	labels := make([]float64, n)
	for i := 0; i < n; i++ {
		data[i] = make([]float64, 2)
		data[i][0] = float64(i) / 100.0
		data[i][1] = float64((i*3)%100) / 100.0
		labels[i] = data[i][0] + 0.5*data[i][1]
	}

	dm, _ := NewDMatrix(data, labels)

	cfg := DefaultConfig()
	cfg.NumTrees = 5
	cfg.MaxDepth = 3
	cfg.Objective = ObjRegSquareError
	cfg.Verbosity = 0

	gbt := NewGBTree(cfg)
	gbt.Train(dm, nil)

	// 保存
	path := "/tmp/test_xgb_model.json"
	defer os.Remove(path)
	err := gbt.SaveModel(path)
	if err != nil {
		t.Fatalf("SaveModel: %v", err)
	}

	// 加载
	loaded, err := LoadModel(path)
	if err != nil {
		t.Fatalf("LoadModel: %v", err)
	}

	// 比较预测值
	origPreds := gbt.PredictBatch(data)
	loadedPreds := loaded.PredictBatch(data)

	for i := range origPreds {
		if math.Abs(origPreds[i]-loadedPreds[i]) > 1e-10 {
			t.Errorf("pred mismatch at %d: orig=%f, loaded=%f", i, origPreds[i], loadedPreds[i])
		}
	}
}

func TestMetric_RMSE(t *testing.T) {
	dm, _ := NewDMatrix([][]float64{{1}, {2}, {3}}, []float64{1, 2, 3})
	m := &RMSEMetric{}
	// 完美预测
	v := m.Evaluate([]float64{1, 2, 3}, dm)
	if v != 0 {
		t.Errorf("expected 0 RMSE for perfect preds, got %f", v)
	}
	// 有误差的预测
	v = m.Evaluate([]float64{2, 3, 4}, dm)
	if math.Abs(v-1.0) > 1e-10 {
		t.Errorf("expected 1.0 RMSE, got %f", v)
	}
}

func TestMetric_AUC(t *testing.T) {
	dm, _ := NewDMatrix([][]float64{{1}, {2}, {3}, {4}}, []float64{0, 0, 1, 1})
	m := &AUCMetric{}
	// 完美排序
	v := m.Evaluate([]float64{0.1, 0.2, 0.8, 0.9}, dm)
	if v < 0.99 {
		t.Errorf("expected AUC ≈ 1.0, got %f", v)
	}
	// 随机排序（更多样本以获得稳定的 AUC）
	nRand := 100
	randData := make([][]float64, nRand)
	randLabels := make([]float64, nRand)
	randPreds := make([]float64, nRand)
	for i := 0; i < nRand; i++ {
		randData[i] = []float64{float64(i)}
		randLabels[i] = float64(i % 2)
		randPreds[i] = float64((i*137+53)%100) / 100.0 // pseudorandom
	}
	randDm, _ := NewDMatrix(randData, randLabels)
	v = m.Evaluate(randPreds, randDm)
	if v < 0.3 || v > 0.7 {
		t.Errorf("expected AUC ≈ 0.5 for random-ish preds, got %f", v)
	}
}

func TestParseMetric(t *testing.T) {
	for _, name := range []string{"rmse", "mae", "logloss", "error", "auc"} {
		m, ok := ParseMetric(name)
		if !ok {
			t.Errorf("ParseMetric(%q) should succeed", name)
		}
		if m.Name() != name {
			t.Errorf("expected name %q, got %q", name, m.Name())
		}
	}
	_, ok := ParseMetric("unknown")
	if ok {
		t.Error("ParseMetric(unknown) should fail")
	}
}

func TestRowSample(t *testing.T) {
	n := 100
	ratio := 0.5
	indices := rowSample(n, ratio, 42)
	if len(indices) == 0 {
		t.Error("should have some indices")
	}
	// 检查无重复
	seen := make(map[int]bool)
	for _, idx := range indices {
		if seen[idx] {
			t.Errorf("duplicate index: %d", idx)
		}
		seen[idx] = true
	}
}

func TestColSample(t *testing.T) {
	n := 20
	ratio := 0.5
	features := colSample(n, ratio, 42)
	if len(features) == 0 {
		t.Error("should have some features")
	}
}
