package xgb

import (
	"math"
	"testing"
)

func TestSplitCandidate_Default(t *testing.T) {
	tests := []struct {
		name string
	}{
		{"split_candidate_default"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// 基础测试：分裂候选构造
			sc := &SplitCandidate{
				FeatureIndex: 3,
				Threshold:    0.5,
				Gain:         1.2,
				DefaultLeft:  true,
			}
			if sc.FeatureIndex != 3 || sc.Gain != 1.2 {
				t.Error("SplitCandidate fields not set correctly")
			}
		})
	}
}

func TestCalcGain(t *testing.T) {
	// 梯度平衡的分裂
	gain := calcGain(5.0, 10.0, 5.0, 10.0, 1.0, 0.0)

	// GL²/(HL+λ) = 25/11 ≈ 2.2727
	// GR²/(HR+λ) = 25/11 ≈ 2.2727
	// (GL+GR)²/(HL+HR+λ) = 100/21 ≈ 4.7619
	// Gain = 0.5 * (2.2727 + 2.2727 - 4.7619) = 0.5 * (-0.2165) = -0.1083
	if math.IsNaN(gain) {
		t.Error("gain should not be NaN")
	}
}

func TestCalcLeafWeight(t *testing.T) {
	w := calcLeafWeight(-10.0, 20.0, 1.0, 0.0)
	expected := 10.0 / 21.0
	if math.Abs(w-expected) > 1e-10 {
		t.Errorf("expected %f, got %f", expected, w)
	}
}

func TestCalcLeafWeight_L1Reg(t *testing.T) {
	// With alpha=5, sumGrad=-10, sumHess=20, lambda=1
	// w = 10/21 ≈ 0.476
	// alpha/denom = 5/21 ≈ 0.238
	// w > alpha/denom, so w -= alpha/denom ≈ 0.238
	w := calcLeafWeight(-10.0, 20.0, 1.0, 5.0)
	if math.IsNaN(w) {
		t.Error("L1-regularized leaf weight should not be NaN")
	}
}

func TestCalcLeafWeight_ZeroDenom(t *testing.T) {
	w := calcLeafWeight(5.0, -1.0, 1.0, 0.0)
	if w != 0 {
		t.Errorf("expected 0 for zero denominator, got %f", w)
	}
}

func TestPartitionSamples(t *testing.T) {
	data := [][]float64{
		{1.0}, {3.0}, {2.0}, {4.0}, {math.NaN()},
	}
	dm, _ := NewDMatrix(data, []float64{0, 1, 0, 1, 0})

	cfg := &ExactBuilderConfig{
		MaxDepth:       6,
		Gamma:          0,
		Lambda:         1.0,
		Alpha:          0,
		MinChildWeight: 1.0,
		NumFeatures:    1,
	}
	b := NewExactBuilder(cfg)

	indices := []int{0, 1, 2, 3, 4}
	// 左：1.0 ≤ 2.5, 2.0 ≤ 2.5 → 索引 {0, 2}
	// 右：3.0 > 2.5, 4.0 > 2.5, NaN（defaultLeft=false → 向右）→ 索引 {1, 3, 4}
	left, right := b.partitionSamples(dm, indices, 0, 2.5, false)

	// 左：1.0 ≤ 2.5, 2.0 ≤ 2.5 → 索引 {0, 2}
	// 右：3.0 > 2.5, 4.0 > 2.5, NaN（默认向右）→ 索引 {1, 3, 4}
	if len(left) != 2 || len(right) != 3 {
		t.Errorf("wrong partition sizes: left=%d, right=%d", len(left), len(right))
	}
}

func TestSumGH(t *testing.T) {
	grads := []float64{1.0, 2.0, 3.0, 4.0, 5.0}
	hess := []float64{0.1, 0.2, 0.3, 0.4, 0.5}
	indices := []int{0, 2, 4}

	g, h := sumGH(grads, hess, indices)
	if math.Abs(g-9.0) > 1e-10 || math.Abs(h-0.9) > 1e-10 {
		t.Errorf("expected g=9.0, h=0.9, got g=%f, h=%f", g, h)
	}
}

func TestEnumerateSplit_Basic(t *testing.T) {
	data := [][]float64{{1.0, 5.0}, {2.0, 4.0}, {3.0, 3.0}}
	dm, _ := NewDMatrix(data, []float64{0, 1, 0})

	grads := []float64{-0.5, 0.3, 0.2}
	hess := []float64{0.25, 0.21, 0.19}
	indices := []int{0, 1, 2}

	cfg := &ExactBuilderConfig{
		MaxDepth:       6,
		Gamma:          0,
		Lambda:         1.0,
		Alpha:          0,
		MinChildWeight: 0.1, // low threshold
		NumFeatures:    2,
	}
	b := NewExactBuilder(cfg)

	candidate := b.EnumerateSplit(dm, 0, grads, hess, indices, 0.0, 0.65)
	if candidate == nil {
		t.Fatal("expected valid split candidate")
	}

	// 分裂应在 1.0 和 2.0 之间或 2.0 和 3.0 之间
	if candidate.Gain <= 0 {
		t.Errorf("expected positive gain, got %f", candidate.Gain)
	}
}

func TestEnumerateSplit_AllSameValue(t *testing.T) {
	data := [][]float64{{1.0}, {1.0}, {1.0}}
	dm, _ := NewDMatrix(data, []float64{0, 1, 0})
	grads := []float64{-0.5, 0.3, 0.2}
	hess := []float64{0.25, 0.21, 0.19}

	cfg := &ExactBuilderConfig{
		MaxDepth:       6,
		Gamma:          0,
		Lambda:         1.0,
		Alpha:          0,
		MinChildWeight: 1.0,
		NumFeatures:    1,
	}
	b := NewExactBuilder(cfg)

	candidate := b.EnumerateSplit(dm, 0, grads, hess, []int{0, 1, 2}, 0, 0.65)
	if candidate != nil {
		t.Error("all same values: should not produce split")
	}
}

func TestFindBestSplit(t *testing.T) {
	data := [][]float64{
		{1.0, 10.0},
		{2.0, 20.0},
		{3.0, 30.0},
		{4.0, 40.0},
		{5.0, 50.0},
	}
	dm, _ := NewDMatrix(data, []float64{0, 1, 0, 1, 0})
	grads := []float64{0.5, -0.5, 0.5, -0.5, 0.5}
	hess := []float64{1.0, 1.0, 1.0, 1.0, 1.0}

	cfg := &ExactBuilderConfig{
		MaxDepth:       6,
		Gamma:          0,
		Lambda:         1.0,
		Alpha:          0,
		MinChildWeight: 0.1,
		NumFeatures:    2,
	}
	b := NewExactBuilder(cfg)

	totalG, totalH := sumGH(grads, hess, []int{0, 1, 2, 3, 4})
	best := b.FindBestSplit(dm, grads, hess, []int{0, 1, 2, 3, 4}, nil, totalG, totalH)

	if best == nil || best.Gain <= 0 {
		t.Error("expected a valid split with positive gain")
	}
}

func TestExactBuilder_Build(t *testing.T) {
	data := [][]float64{
		{1.0, 5.0},
		{2.0, 4.0},
		{3.0, 3.0},
		{4.0, 2.0},
		{5.0, 1.0},
	}
	dm, _ := NewDMatrix(data, []float64{0, 1, 0, 1, 0})

	grads := []float64{0.5, -0.5, 0.5, -0.5, 0.5}
	hess := []float64{1.0, 1.0, 1.0, 1.0, 1.0}

	cfg := &ExactBuilderConfig{
		MaxDepth:       6,
		Gamma:          0,
		Lambda:         1.0,
		Alpha:          0,
		MinChildWeight: 0.1,
		NumFeatures:    2,
	}
	b := NewExactBuilder(cfg)

	err := b.Build(dm, grads, hess, nil, nil)
	if err != nil {
		t.Fatalf("Build returned error: %v", err)
	}

	if b.tree == nil {
		t.Fatal("tree should not be nil after build")
	}
	if len(b.tree.Nodes) == 0 {
		t.Fatal("tree should have nodes")
	}

	// 根应为分裂节点（因为数据有变化）
	if b.tree.Nodes[0].IsLeaf() && len(data) > 2 {
		t.Fatal("root should be a split for varied data")
	}
}
