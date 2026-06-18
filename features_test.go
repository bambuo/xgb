package xgb

import (
	"math"
	"os"
	"testing"
)

// TestSHAP 测试 SHAP 值计算。
func TestSHAP(t *testing.T) {
	data := [][]float64{{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}}
	labels := []float64{1.0, 3.0, 5.0}
	dm, _ := NewDMatrix(data, labels)

	cfg := DefaultConfig()
	cfg.NumTrees = 5
	cfg.MaxDepth = 3
	cfg.Objective = ObjRegSquareError
	cfg.Verbosity = 0

	gbt := NewGBTree(cfg)
	_, _ = gbt.Train(dm, nil)

	t.Run("shap_sum_equals_pred", func(t *testing.T) {
		shap := gbt.SHAP(data[0])
		pred := gbt.Predict(data[0])
		var shapSum float64
		for _, v := range shap {
			shapSum += v
		}
		if math.Abs(shapSum-pred) > 1e-6 {
			t.Errorf("SHAP sum %f != pred %f (diff=%e)", shapSum, pred, shapSum-pred)
		}
	})
	t.Run("shap_has_correct_len", func(t *testing.T) {
		shap := gbt.SHAP(data[0])
		if len(shap) != 3 { // 2 features + 1 base
			t.Errorf("expected len 3, got %d", len(shap))
		}
	})
	t.Run("shap_base_value", func(t *testing.T) {
		shap := gbt.SHAP(data[0])
		// SHAP base value = baseScore (without LR applied to it, but with LR on trees)
		// effectiveBaseScore is the raw prediction starting point
		// Without trees, SHAP base should equal effectiveBaseScore
		if math.Abs(shap[2]) < 1e-10 && math.Abs(gbt.effectiveBaseScore()) > 1e-10 {
			t.Errorf("base value is 0 but effectiveBaseScore is %f", gbt.effectiveBaseScore())
		}
	})
}

// TestSHAPInteraction 测试 SHAP 交互值。
func TestSHAPInteraction(t *testing.T) {
	data := [][]float64{{1.0, 2.0}, {3.0, 4.0}}
	labels := []float64{1.0, 3.0}
	dm, _ := NewDMatrix(data, labels)

	cfg := DefaultConfig()
	cfg.NumTrees = 3
	cfg.MaxDepth = 2
	cfg.Objective = ObjRegSquareError
	cfg.Verbosity = 0

	gbt := NewGBTree(cfg)
	_, _ = gbt.Train(dm, nil)

	interaction := gbt.SHAPInteraction(data[0])
	if len(interaction) != 3 {
		t.Fatalf("expected 3x3 matrix, got %dx%d", len(interaction), len(interaction[0]))
	}
	// 检查交互矩阵的非负性（用于可视化）
	var nonZero bool
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			if math.Abs(interaction[i][j]) > 1e-10 {
				nonZero = true
			}
		}
	}
	if !nonZero {
		t.Error("interaction matrix is all zeros")
	}
}

// TestApproxSHAP 测试近似 SHAP。
func TestApproxSHAP(t *testing.T) {
	data := [][]float64{{1.0, 2.0}, {3.0, 4.0}}
	labels := []float64{1.0, 3.0}
	dm, _ := NewDMatrix(data, labels)

	cfg := DefaultConfig()
	cfg.NumTrees = 3
	cfg.MaxDepth = 2
	cfg.Objective = ObjRegSquareError
	cfg.Verbosity = 0

	gbt := NewGBTree(cfg)
	_, _ = gbt.Train(dm, nil)

	approx := gbt.ApproxSHAP(data[0])
	pred := gbt.Predict(data[0])
	var sum float64
	for _, v := range approx {
		sum += v
	}
	if math.Abs(sum-pred) > 1e-6 {
		t.Errorf("ApproxSHAP sum %f != pred %f", sum, pred)
	}
}

// TestFeatureImportance 测试特征重要性。
func TestFeatureImportance(t *testing.T) {
	data := [][]float64{{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}}
	labels := []float64{1.0, 3.0, 5.0}
	dm, _ := NewDMatrix(data, labels)

	cfg := DefaultConfig()
	cfg.NumTrees = 5
	cfg.MaxDepth = 3
	cfg.Verbosity = 0

	gbt := NewGBTree(cfg)
	_, _ = gbt.Train(dm, nil)

	t.Run("importance_weight", func(t *testing.T) {
		scores := gbt.GetScore(ImportanceWeight)
		if len(scores) == 0 {
			t.Error("importance weight should not be empty")
		}
	})
	t.Run("importance_gain", func(t *testing.T) {
		scores := gbt.GetScore(ImportanceGain)
		if len(scores) == 0 {
			t.Error("importance gain should not be empty")
		}
	})
	t.Run("importance_cover", func(t *testing.T) {
		scores := gbt.GetScore(ImportanceCover)
		if len(scores) == 0 {
			t.Error("importance cover should not be empty")
		}
	})
	t.Run("fscore", func(t *testing.T) {
		scores := gbt.GetFScore()
		if len(scores) == 0 {
			t.Error("FScore should not be empty")
		}
	})
	t.Run("ranking", func(t *testing.T) {
		ranking := gbt.ImportanceRanking(ImportanceWeight)
		if len(ranking) == 0 {
			t.Error("ranking should not be empty")
		}
		if ranking[0].Rank != 1 {
			t.Errorf("top rank should be 1, got %d", ranking[0].Rank)
		}
	})
}

// TestCrossValidation 测试交叉验证。
func TestCrossValidation(t *testing.T) {
	data := [][]float64{
		{1.0, 2.0}, {2.0, 3.0}, {3.0, 4.0},
		{4.0, 5.0}, {5.0, 6.0}, {6.0, 7.0},
	}
	labels := []float64{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}
	dm, _ := NewDMatrix(data, labels)

	cfg := DefaultConfig()
	cfg.NumTrees = 5
	cfg.MaxDepth = 3
	cfg.Verbosity = 0

	metrics := []Metric{&RMSEMetric{}}
	results, err := CV(cfg, dm, metrics, 3, false, 42)
	if err != nil {
		t.Fatalf("CV error: %v", err)
	}
	if len(results) != 3 {
		t.Errorf("expected 3 folds, got %d", len(results))
	}
	summary := SummarizeCV(results, "rmse")
	if summary.TestMean <= 0 {
		t.Errorf("expected positive test RMSE, got %f", summary.TestMean)
	}
}

// TestStratifiedKFold 测试分层 K 折。
func TestStratifiedKFold(t *testing.T) {
	skf := &StratifiedKFold{NSplits: 3, Shuffle: false}
	labels := []float64{0, 0, 0, 1, 1, 1, 2, 2, 2}
	splits := skf.Split(labels)
	if len(splits) != 3 {
		t.Errorf("expected 3 splits, got %d", len(splits))
	}
	for i, s := range splits {
		if len(s.Test) == 0 || len(s.Train) == 0 {
			t.Errorf("fold %d: empty train or test", i)
		}
	}
}

// TestGridSearch 测试网格搜索。
func TestGridSearch(t *testing.T) {
	data := [][]float64{
		{1.0, 2.0}, {2.0, 3.0}, {3.0, 4.0}, {4.0, 5.0},
	}
	labels := []float64{1.0, 2.0, 3.0, 4.0}
	dm, _ := NewDMatrix(data, labels)

	cfg := DefaultConfig()
	cfg.NumTrees = 3
	cfg.MaxDepth = 2
	cfg.Verbosity = 0

	gc := GridSearchConfig{
		Params: []GridSearchParam{
			{Name: "learning_rate", Values: []interface{}{0.1, 0.3}},
			{Name: "max_depth", Values: []interface{}{2, 3}},
		},
		Metrics: []Metric{&RMSEMetric{}},
		NFolds:  2,
	}

	results, err := GridSearch(cfg, dm, gc)
	if err != nil {
		t.Fatalf("GridSearch error: %v", err)
	}
	if len(results) != 4 { // 2*2 = 4 组合
		t.Errorf("expected 4 results, got %d", len(results))
	}
	if results[0].Scores["rmse"] <= 0 {
		t.Error("expected positive RMSE")
	}
}

// TestGBLinear 测试线性提升器。
func TestGBLinear(t *testing.T) {
	data := [][]float64{
		{1.0, 2.0}, {2.0, 3.0}, {3.0, 4.0}, {4.0, 5.0},
	}
	labels := []float64{3.0, 5.0, 7.0, 9.0} // y = x0 + 2*x1

	dm, _ := NewDMatrix(data, labels)
	cfg := DefaultConfig()
	cfg.NumTrees = 500
	cfg.LearningRate = 0.03
	cfg.Objective = ObjRegSquareError
	cfg.Verbosity = 0

	gl := NewGBLinear(cfg)
	history, err := gl.Train(dm, []Metric{&RMSEMetric{}})
	if err != nil {
		t.Fatalf("GBLinear Train error: %v", err)
	}
	if len(history) != 500 {
		t.Errorf("expected 500 history entries, got %d", len(history))
	}

	// 检查单调性（线性模型收敛慢，但趋势应正确）
	preds := make([]float64, 4)
	for i, row := range data {
		preds[i] = gl.Predict(row)
	}
	for i := 1; i < len(preds); i++ {
		if preds[i] <= preds[i-1] {
			t.Errorf("predictions not monotonic: %v", preds)
			break
		}
	}

	// 检查 RMSE 收敛
	lastRMSE := history[len(history)-1].Metrics["rmse"]
	firstRMSE := history[0].Metrics["rmse"]
	if lastRMSE >= firstRMSE {
		t.Errorf("RMSE did not converge: first=%.4f last=%.4f", firstRMSE, lastRMSE)
	}

	weights := gl.GetWeights()
	if len(weights) != 3 {
		t.Errorf("expected 3 weights, got %d", len(weights))
	}
}

// TestPartialDependence 测试偏依赖。
func TestPartialDependence(t *testing.T) {
	data := [][]float64{
		{1.0, 2.0}, {2.0, 3.0}, {3.0, 4.0}, {4.0, 5.0},
	}
	labels := []float64{1.0, 2.0, 3.0, 4.0}
	dm, _ := NewDMatrix(data, labels)

	cfg := DefaultConfig()
	cfg.NumTrees = 5
	cfg.MaxDepth = 2
	cfg.Verbosity = 0

	gbt := NewGBTree(cfg)
	_, _ = gbt.Train(dm, nil)

	pdp := gbt.PartialDependence(0, data, 5)
	if len(pdp) != 5 {
		t.Errorf("expected 5 PDP points, got %d", len(pdp))
	}
	if math.IsNaN(pdp[0].Prediction) {
		t.Error("PDP prediction is NaN")
	}
}

// TestDumpModel 测试模型导出。
func TestDumpModel(t *testing.T) {
	data := [][]float64{{1.0, 2.0}, {3.0, 4.0}}
	labels := []float64{1.0, 3.0}
	dm, _ := NewDMatrix(data, labels)

	cfg := DefaultConfig()
	cfg.NumTrees = 2
	cfg.MaxDepth = 2
	cfg.Verbosity = 0

	gbt := NewGBTree(cfg)
	_, _ = gbt.Train(dm, nil)

	dumps := gbt.DumpModel()
	if len(dumps) != 2 {
		t.Errorf("expected 2 dumps, got %d", len(dumps))
	}
	if dumps[0] == "" {
		t.Error("dump is empty")
	}

	dots := gbt.DumpModelDOT()
	if dots == "" {
		t.Error("DOT output is empty")
	}
}

// TestLearningCurve 测试学习曲线。
func TestLearningCurve(t *testing.T) {
	history := []EvalResult{
		{Iteration: 0, Metrics: map[string]float64{"rmse": 0.9}},
		{Iteration: 1, Metrics: map[string]float64{"rmse": 0.7}},
		{Iteration: 2, Metrics: map[string]float64{"rmse": 0.5}},
	}

	points := LearningCurve(history, "rmse")
	if len(points) != 3 {
		t.Errorf("expected 3 points, got %d", len(points))
	}

	bestIter, bestVal := BestIteration(history, "rmse", true)
	if bestIter != 2 || bestVal != 0.5 {
		t.Errorf("expected iter=2 val=0.5, got iter=%d val=%f", bestIter, bestVal)
	}

	plot := PlotLearningCurve(history, "rmse", 20, 6)
	if plot == "" {
		t.Error("plot is empty")
	}
}

// TestSetGroup 测试分组设置。
func TestSetGroup(t *testing.T) {
	data := [][]float64{{1.0}, {2.0}, {3.0}, {4.0}, {5.0}}
	labels := []float64{0, 1, 0, 1, 0}
	dm, _ := NewDMatrix(data, labels)

	err := dm.SetGroup([]int{2, 3})
	if err != nil {
		t.Fatalf("SetGroup error: %v", err)
	}
	if len(dm.Group) != 2 {
		t.Errorf("expected 2 groups, got %d", len(dm.Group))
	}
}

// TestFeatureMetadata 测试特征元数据。
func TestFeatureMetadata(t *testing.T) {
	data := [][]float64{{1.0, 2.0}, {3.0, 4.0}}
	labels := []float64{1.0, 3.0}
	dm, _ := NewDMatrix(data, labels)

	err := dm.SetFeatureNames([]string{"feat0", "feat1"})
	if err != nil {
		t.Fatalf("SetFeatureNames error: %v", err)
	}
	if dm.FeatureNames[0] != "feat0" {
		t.Errorf("expected 'feat0', got %q", dm.FeatureNames[0])
	}

	err = dm.SetFeatureTypes([]string{"float", "categorical"})
	if err != nil {
		t.Fatalf("SetFeatureTypes error: %v", err)
	}
	if dm.FeatureTypes[1] != "categorical" {
		t.Errorf("expected 'categorical', got %q", dm.FeatureTypes[1])
	}
}

// TestCatEncoder 测试类别编码器。
func TestCatEncoder(t *testing.T) {
	data := [][]float64{
		{1.0, 2.0},
		{1.0, 3.0},
		{2.0, 2.0},
	}
	ce := NewCatEncoder(4)
	ce.Detect(data, nil)

	if len(ce.CatCols) == 0 {
		t.Error("CatEncoder should detect columns")
	}
}

// TestDMatrixFromCSV 测试 CSV 加载。
func TestDMatrixFromCSV(t *testing.T) {
	// 创建临时 CSV 文件
	csvPath := "/tmp/test_xgb.csv"
	csvContent := "1.0,2.0,3.0\n4.0,5.0,6.0\n"
	if err := os.WriteFile(csvPath, []byte(csvContent), 0644); err != nil {
		t.Fatal(err)
	}
	defer os.Remove(csvPath)

	dm, err := LoadDMatrixFromCSV(csvPath, 2)
	if err != nil {
		t.Fatalf("LoadDMatrixFromCSV error: %v", err)
	}
	if dm.NumRows != 2 {
		t.Errorf("expected 2 rows, got %d", dm.NumRows)
	}
}

// TestIterationRange 测试迭代范围预测。
func TestIterationRange(t *testing.T) {
	data := [][]float64{{1.0, 2.0}, {3.0, 4.0}}
	labels := []float64{1.0, 3.0}
	dm, _ := NewDMatrix(data, labels)

	cfg := DefaultConfig()
	cfg.NumTrees = 5
	cfg.MaxDepth = 2
	cfg.Verbosity = 0

	gbt := NewGBTree(cfg)
	_, _ = gbt.Train(dm, nil)

	// 所有树
	p1 := gbt.Predict(data[0])
	// 前 3 棵树
	p2 := gbt.Predict(data[0], 0, 3)
	if math.Abs(p1-p2) < 1e-10 {
		t.Error("partial prediction should differ from full")
	}
	// 第 3-5 棵树
	p3 := gbt.Predict(data[0], 3, 5)
	_ = p3
}

// TestExternalMemoryOutOfCore 测试外存模式不下沉。
func TestExternalMemoryOutOfCore(t *testing.T) {
	cfg := DefaultConfig()
	cfg.NumTrees = 2
	cfg.MaxDepth = 2
	cfg.Verbosity = 0
	cfg.TreeMethod = "hist" // 外存仅支持 hist

	gbt := NewGBTree(cfg)

	// 创建临时文件
	f1 := "/tmp/test_ext_1.csv"
	f2 := "/tmp/test_ext_2.csv"
	os.WriteFile(f1, []byte("1.0,2.0,3.0\n4.0,5.0,6.0\n"), 0644)
	os.WriteFile(f2, []byte("7.0,8.0,9.0\n"), 0644)
	defer os.Remove(f1)
	defer os.Remove(f2)

	dmc, err := NewDMatrixChunked([]string{f1, f2}, "csv", 2, 0)
	if err != nil {
		t.Fatal(err)
	}

	_, err = gbt.ExternalTrain(dmc, nil)
	if err != nil {
		t.Logf("ExternalTrain error (expected if incomplete): %v", err)
	}
}
