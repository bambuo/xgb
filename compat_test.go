package xgb

import (
	"math"
	"os"
	"testing"
)

// ── GK Sketch 测试 ─────────────────────────────────────────

func TestGKSketch_Basic(t *testing.T) {
	sketch := NewGKSketch(10)

	// 插入 100 个等间距值
	for i := 0; i < 100; i++ {
		sketch.Push(float32(i), 1.0)
	}

	if len(sketch.Entries) == 0 {
		t.Fatal("sketch should have entries")
	}

	// 查询中位数
	median := sketch.Query(0.5)
	if median < 40 || median > 60 {
		t.Errorf("median should be around 50, got %f", median)
	}

	// 查询最小值附近
	minVal := sketch.Query(0.01)
	if minVal > 10 {
		t.Errorf("1%% quantile should be small, got %f", minVal)
	}

	// 查询最大值附近
	maxVal := sketch.Query(0.99)
	if maxVal < 90 {
		t.Errorf("99%% quantile should be large, got %f", maxVal)
	}
}

func TestGKSketch_Weighted(t *testing.T) {
	sketch := NewGKSketch(10)

	// 大部分权重在 1.0 附近
	for i := 0; i < 100; i++ {
		sketch.Push(1.0, 10.0) // 100 * 10 = 1000 weight at 1.0
	}
	// 少量权重在 100.0
	sketch.Push(100.0, 1.0) // 1 weight at 100.0

	// 中位数应该是 1.0（大部分权重在这里）
	median := sketch.Query(0.5)
	if median != 1.0 {
		t.Errorf("weighted median should be 1.0, got %f", median)
	}
}

func TestGKSketch_QueryAll(t *testing.T) {
	sketch := NewGKSketch(10)

	for i := 0; i < 100; i++ {
		sketch.Push(float32(i), 1.0)
	}

	boundaries := sketch.QueryAll(10)
	if len(boundaries) == 0 {
		t.Fatal("should have boundaries")
	}

	// 边界应该递增
	for i := 1; i < len(boundaries); i++ {
		if boundaries[i] <= boundaries[i-1] {
			t.Errorf("boundaries should be increasing: %f <= %f at %d",
				boundaries[i], boundaries[i-1], i)
		}
	}
}

func TestBuildHistogramCuts(t *testing.T) {
	// 简单数据集
	data := [][]float64{
		{1.0, 10.0},
		{2.0, 20.0},
		{3.0, 30.0},
		{4.0, 40.0},
		{5.0, 50.0},
	}
	hess := []float64{1.0, 1.0, 1.0, 1.0, 1.0}

	cuts := BuildHistogramCuts(data, hess, 4)

	if len(cuts.Ptrs) != 3 { // 2 features + 1
		t.Errorf("expected 3 ptrs, got %d", len(cuts.Ptrs))
	}

	// 第一个特征应该有 bin 边界
	nBins0 := cuts.Ptrs[1] - cuts.Ptrs[0]
	if nBins0 == 0 {
		t.Error("feature 0 should have boundaries")
	}

	// BinForValue 测试
	bin := cuts.BinForValue(0, 1.0)
	if bin != 0 {
		t.Errorf("value 1.0 should be in bin 0, got %d", bin)
	}

	bin = cuts.BinForValue(0, math.NaN())
	if bin != 0 {
		t.Errorf("NaN should map to bin 0, got %d", bin)
	}
}

// ── GK BinMapper 测试 ──────────────────────────────────────

func TestNewGKBinMapper(t *testing.T) {
	data := make([][]float64, 100)
	hess := make([]float64, 100)
	for i := range data {
		data[i] = []float64{float64(i) / 100.0, float64(100-i) / 100.0}
		hess[i] = 1.0
	}

	bm := NewGKBinMapper(data, hess, 16)

	if bm.NumBins != 16 {
		t.Errorf("expected 16 bins, got %d", bm.NumBins)
	}
	if bm.numFeatures != 2 {
		t.Errorf("expected 2 features, got %d", bm.numFeatures)
	}

	// 每个特征应该有 bin 边界
	for f := 0; f < 2; f++ {
		if len(bm.Boundaries[f]) == 0 {
			t.Errorf("feature %d should have boundaries", f)
		}
	}

	// 边界应该递增
	for f := 0; f < 2; f++ {
		for i := 1; i < len(bm.Boundaries[f]); i++ {
			if bm.Boundaries[f][i] <= bm.Boundaries[f][i-1] {
				t.Errorf("feature %d: boundaries not increasing at %d", f, i)
			}
		}
	}
}

// ── 序列化兼容性测试 ────────────────────────────────────────

func TestSaveLoad_DefaultLeft(t *testing.T) {
	// 创建一棵包含 NaN 数据的树以测试 DefaultLeft
	n := 100
	data := make([][]float64, n)
	labels := make([]float64, n)
	for i := 0; i < n; i++ {
		data[i] = make([]float64, 2)
		if i%7 == 0 {
			data[i][0] = math.NaN()
		} else {
			data[i][0] = float64(i) / 100.0
		}
		data[i][1] = float64((i*3)%100) / 100.0
		labels[i] = data[i][1]
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
	path := "/tmp/test_xgb_defaultleft.json"
	defer os.Remove(path)
	if err := gbt.SaveModel(path); err != nil {
		t.Fatalf("SaveModel: %v", err)
	}

	// 加载
	loaded, err := LoadModel(path)
	if err != nil {
		t.Fatalf("LoadModel: %v", err)
	}

	// 比较预测值（NaN 处理应该一致）
	for i, row := range data {
		orig := gbt.Predict(row)
		load := loaded.Predict(row)
		if math.Abs(orig-load) > 1e-10 {
			t.Errorf("pred mismatch at %d: orig=%f, loaded=%f", i, orig, load)
		}
	}
}

func TestLoadModelWithLR(t *testing.T) {
	// 训练一个学习率非 0.3 的模型
	n := 50
	data := make([][]float64, n)
	labels := make([]float64, n)
	for i := 0; i < n; i++ {
		data[i] = []float64{float64(i) / 50.0, float64((i*3)%50) / 50.0}
		labels[i] = data[i][0] + data[i][1]
	}

	dm, _ := NewDMatrix(data, labels)
	cfg := DefaultConfig()
	cfg.NumTrees = 5
	cfg.MaxDepth = 3
	cfg.LearningRate = 0.1 // 非默认学习率
	cfg.Objective = ObjRegSquareError
	cfg.Verbosity = 0

	gbt := NewGBTree(cfg)
	gbt.Train(dm, nil)

	// 保存
	path := "/tmp/test_xgb_lr.json"
	defer os.Remove(path)
	if err := gbt.SaveModel(path); err != nil {
		t.Fatalf("SaveModel: %v", err)
	}

	// 加载（应该自动从 attributes 读取学习率）
	loaded, err := LoadModel(path)
	if err != nil {
		t.Fatalf("LoadModel: %v", err)
	}

	// 比较预测值
	for i, row := range data {
		orig := gbt.Predict(row)
		load := loaded.Predict(row)
		if math.Abs(orig-load) > 1e-10 {
			t.Errorf("pred mismatch at %d with auto LR: orig=%f, loaded=%f", i, orig, load)
		}
	}

	// 手动指定错误学习率加载（跳过自动检测）
	// 由于 attributes 中包含 go_learning_rate，LoadModelWithLR 会自动使用保存的值。
	// 要测试错误 LR 的效果，需要使用不保存学习率的原始 XGBoost 模型。
	// 这里只验证 LoadModelWithLR 接口可用。
	loadedManual, err := LoadModelWithLR(path, 0.5)
	if err != nil {
		t.Fatalf("LoadModelWithLR: %v", err)
	}

	// 由于自动检测会覆盖传入的 LR，预测值应该与正确加载一致
	origPred := gbt.Predict(data[0])
	manualPred := loadedManual.Predict(data[0])
	if math.Abs(origPred-manualPred) > 1e-10 {
		t.Errorf("auto-detected LR should override manual LR: orig=%f, manual=%f", origPred, manualPred)
	}

	// 验证学习率被正确保存和读取
	if loadedManual.Config.LearningRate != 0.1 {
		t.Errorf("expected LR=0.1 from attributes, got %f", loadedManual.Config.LearningRate)
	}
}

// ── float32 精度测试 ────────────────────────────────────────

func TestToF32(t *testing.T) {
	// float32 精度约为 7 位有效数字
	v := 1.23456789012345
	truncated := toF32(v)

	// 截断后的值应该与原始值不同（超过 float32 精度的部分被截断）
	if truncated == v {
		t.Error("toF32 should truncate precision")
	}

	// 差值应该在 1e-7 量级
	diff := math.Abs(v - truncated)
	if diff > 1e-6 {
		t.Errorf("toF32 diff too large: %e", diff)
	}
}

func TestTruncateGradients(t *testing.T) {
	grads := []float64{1.23456789012345, 2.34567890123456}
	hess := []float64{0.12345678901234, 0.23456789012345}

	truncateGradients(grads, hess)

	// 截断后的值应该与 float32 一致
	for i := range grads {
		expected := float64(float32(1.23456789012345))
		if i == 0 && grads[i] != expected {
			t.Errorf("grads[0] = %v, expected %v", grads[i], expected)
		}
	}
}

// ── mt19937 正确性测试 ──────────────────────────────────────

func TestMT19937_KnownValues(t *testing.T) {
	// std::mt19937 seed=0 的前 10 个输出值（C++ 实际输出验证）
	expected := []uint32{
		2357136044,
		2546248239,
		3071714933,
		3626093760,
		2588848963,
		3684848379,
		2340255427,
		3638918503,
		1819583497,
		2678185683,
	}

	rng := NewMT19937(0)
	for i, exp := range expected {
		got := rng.Next()
		if got != exp {
			t.Errorf("mt19937[%d]: got %d, expected %d", i, got, exp)
		}
	}
}

func TestMT19937_Seed42(t *testing.T) {
	// std::mt19937 seed=42 的前 5 个输出值（C++ 实际输出验证）
	expected := []uint32{
		1608637542,
		3421126067,
		4083286876,
		787846414,
		3143890026,
	}

	rng := NewMT19937(42)
	for i, exp := range expected {
		got := rng.Next()
		if got != exp {
			t.Errorf("mt19937 seed=42 [%d]: got %d, expected %d", i, got, exp)
		}
	}
}

func TestMtRowSample(t *testing.T) {
	indices := mtRowSample(100, 0.5, 42)
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

func TestMtColSample(t *testing.T) {
	features := mtColSample(20, 0.5, 42)
	if len(features) == 0 {
		t.Error("should have some features")
	}
}

func TestMtColSampleFromMask(t *testing.T) {
	mask := []int{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

	// 100% 采样应该返回全部
	full := mtColSampleFromMask(mask, 1.0, 42)
	if len(full) != len(mask) {
		t.Errorf("100%% sample should return all, got %d", len(full))
	}

	// 50% 采样
	half := mtColSampleFromMask(mask, 0.5, 42)
	if len(half) == 0 || len(half) >= len(mask) {
		t.Errorf("50%% sample should be partial, got %d", len(half))
	}

	// nil 输入应该返回 nil
	nilResult := mtColSampleFromMask(nil, 0.5, 42)
	if nilResult != nil {
		t.Error("nil input should return nil")
	}
}

// ── partitionSamples DefaultLeft 测试 ──────────────────────────

func TestPartitionSamples_DefaultLeft(t *testing.T) {
	data := [][]float64{
		{1.0}, {3.0}, {2.0}, {4.0}, {math.NaN()},
	}
	dm, _ := NewDMatrix(data, []float64{0, 1, 0, 1, 0})

	cfg := &ExactBuilderConfig{
		MaxDepth: 6, Gamma: 0, Lambda: 1.0, Alpha: 0,
		MinChildWeight: 1.0, NumFeatures: 1,
	}
	b := NewExactBuilder(cfg)
	indices := []int{0, 1, 2, 3, 4}

	// defaultLeft=true：NaN 应该去左
	leftTrue, rightTrue := b.partitionSamples(dm, indices, 0, 2.5, true)
	hasNaNLeft := false
	for _, idx := range leftTrue {
		if math.IsNaN(data[idx][0]) {
			hasNaNLeft = true
		}
	}
	if !hasNaNLeft {
		t.Error("defaultLeft=true: NaN should go left")
	}

	// defaultLeft=false：NaN 应该去右
	leftFalse, rightFalse := b.partitionSamples(dm, indices, 0, 2.5, false)
	hasNaNRight := false
	for _, idx := range rightFalse {
		if math.IsNaN(data[idx][0]) {
			hasNaNRight = true
		}
	}
	if !hasNaNRight {
		t.Error("defaultLeft=false: NaN should go right")
	}

	// 非 NaN 样本分布应该一致
	_ = leftTrue
	_ = rightTrue
	_ = leftFalse
	_ = rightFalse
}
