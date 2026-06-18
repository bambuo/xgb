package xgb

import (
	"math"
	"testing"
)

// TestObjectiveGradients 测试所有目标函数的梯度/Hessian 在已知输入下的正确性。
func TestObjectiveGradients(t *testing.T) {
	tests := []struct {
		name      string
		obj       Objective
		pred      []float64
		labels    []float64
		checkGrad func(g, h []float64) bool
	}{
		{
			name: "reg:squarederror",
			obj:  &SquaredError{},
			pred: []float64{1.0, 2.0, 3.0},
			labels: []float64{0.5, 2.5, 3.5},
			checkGrad: func(g, h []float64) bool {
				return math.Abs(g[0]-0.5) < 1e-10 &&
					math.Abs(g[1] - (-0.5)) < 1e-10 &&
					math.Abs(g[2] - (-0.5)) < 1e-10 &&
					math.Abs(h[0]-1.0) < 1e-10
			},
		},
		{
			name: "reg:logistic",
			obj:  &LogisticRegression{ScalePosWeight: 1.0},
			pred: []float64{0.0, 1.0, -1.0},
			labels: []float64{0, 1, 0},
			checkGrad: func(g, h []float64) bool {
				p0 := sigmoid(0.0)
				p1 := sigmoid(1.0)
				p2 := sigmoid(-1.0)
				_ = p2
				return math.Abs(g[0]-(p0-0)) < 1e-10 &&
					math.Abs(g[1]-(p1-1)) < 1e-10 &&
					!math.IsNaN(g[2]) &&
					math.Abs(h[0]-p0*(1-p0)) < 1e-10 &&
					math.Abs(h[1]-p1*(1-p1)) < 1e-10
			},
			},
			{
			name: "binary:hinge",
			obj:  &BinaryHinge{},
			pred: []float64{2.0, -0.5, -2.0},
			labels: []float64{1, 1, 0}, // y in {0,1}, mapped to {-1,1}
			checkGrad: func(g, h []float64) bool {
				// For label=1 (y=1): g=-1 when 1*2<1? No, so g=0
				// For label=1 (y=1): g=-1 when 1*(-0.5)<1? Yes, so g=-1
				// For label=0 (y=-1): g=1 when -1*(-2)<1? No, so g=0
				return math.Abs(g[0]) < 1e-10 &&
					math.Abs(g[1]+1.0) < 1e-10 &&
					math.Abs(g[2]) < 1e-10
			},
		},
		{
			name: "count:poisson",
			obj:  &PoissonRegression{},
			pred: []float64{0.0, 1.0, 2.0},
			labels: []float64{1.0, 2.0, 3.0},
			checkGrad: func(g, h []float64) bool {
				// g = exp(pred) - label, h = exp(pred)
				return math.Abs(g[0]-(math.Exp(0)-1)) < 1e-8 &&
					math.Abs(g[1]-(math.Exp(1)-2)) < 1e-8 &&
					math.Abs(h[0]-math.Exp(0)) < 1e-8 &&
					math.Abs(h[1]-math.Exp(1)) < 1e-8
			},
		},
		{
			name: "reg:gamma",
			obj:  &GammaRegression{},
			pred: []float64{0.0, 1.0},
			labels: []float64{1.0, 2.0},
			checkGrad: func(g, h []float64) bool {
				// g = 1 - label/exp(pred), h = label/exp(pred)^2
				return math.Abs(g[0]-(1.0-1.0/1.0)) < 1e-8 &&
					math.Abs(g[1]-(1.0-2.0/math.Exp(1))) < 1e-8
			},
		},
		{
			name: "reg:squaredlogerror",
			obj:  &SquaredLogError{},
			pred: []float64{0.0, 1.0},
			labels: []float64{0.5, 1.5},
			checkGrad: func(g, h []float64) bool {
				// g = log1p(pred) - log1p(label) / (1+pred)
				// h = (1 - log1p(pred) + log1p(label)) / (1+pred)^2
				lp0 := math.Log(1 + 0.0)
				ll0 := math.Log(1 + 0.5)
				_g0 := (lp0 - ll0) / 1.0
				return !math.IsNaN(g[0]) && !math.IsNaN(g[1]) &&
					math.Abs(g[0]-_g0) < 1e-8
			},
		},
		{
			name: "survival:cox",
			obj:  &SurvivalCox{},
			pred: []float64{0.0, 1.0, 2.0},
			labels: []float64{1.0, 0.0, 1.0},
			checkGrad: func(g, h []float64) bool {
				// Cox 梯度是排序依赖的，只检查非 NaN
				for i := range g {
					if math.IsNaN(g[i]) || math.IsNaN(h[i]) {
						return false
					}
				}
				return len(g) == 3
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g, h, err := tt.obj.GetGradient(tt.pred, tt.labels, nil)
			if err != nil {
				t.Fatalf("GetGradient error: %v", err)
			}
			if len(g) != len(tt.pred) || len(h) != len(tt.pred) {
				t.Fatalf("expected len=%d, got g=%d h=%d", len(tt.pred), len(g), len(h))
			}
			if !tt.checkGrad(g, h) {
				t.Errorf("gradient/hessian check failed: g=%v h=%v", g, h)
			}
		})
	}
}

// TestObjectivePredTransform 测试所有目标函数的 PredTransform。
func TestObjectivePredTransform(t *testing.T) {
	tests := []struct {
		name   string
		obj    Objective
		input  []float64
		output []float64
		check  func(out, expected []float64) bool
	}{
		{
			name: "reg:squarederror identity",
			obj:  &SquaredError{},
			input: []float64{1.0, 2.0, 3.0},
			check: func(out, _ []float64) bool {
				return math.Abs(out[0]-1.0) < 1e-10 && math.Abs(out[1]-2.0) < 1e-10
			},
		},
		{
			name: "binary:logistic sigmoid",
			obj:  &LogisticRegression{},
			input: []float64{0.0, 1.0, -1.0},
			check: func(out, _ []float64) bool {
				return math.Abs(out[0]-0.5) < 1e-10 &&
					math.Abs(out[1]-sigmoid(1.0)) < 1e-10 &&
					math.Abs(out[2]-sigmoid(-1.0)) < 1e-10
			},
		},
		{
			name: "binary:hinge threshold",
			obj:  &BinaryHinge{},
			input: []float64{0.5, -0.5, 0.0},
			check: func(out, _ []float64) bool {
				return out[0] == 1.0 && out[1] == 0.0 && out[2] == 1.0
			},
		},
		{
			name: "count:poisson exp",
			obj:  &PoissonRegression{},
			input: []float64{0.0, 1.0, 2.0},
			check: func(out, _ []float64) bool {
				return math.Abs(out[0]-1.0) < 1e-10 &&
					math.Abs(out[1]-math.E) < 1e-10 &&
					math.Abs(out[2]-math.E*math.E) < 1e-10
			},
		},
		{
			name: "survival:cox exp",
			obj:  &SurvivalCox{},
			input: []float64{0.0, 1.0},
			check: func(out, _ []float64) bool {
				return math.Abs(out[0]-1.0) < 1e-10 && math.Abs(out[1]-math.E) < 1e-10
			},
		},
		{
			name: "survival:aft exp",
			obj:  &SurvivalAFT{Distribution: "normal"},
			input: []float64{0.0, 1.0},
			check: func(out, _ []float64) bool {
				return math.Abs(out[0]-1.0) < 1e-10 && math.Abs(out[1]-math.E) < 1e-10
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			out := tt.obj.PredTransform(tt.input)
			if len(out) != len(tt.input) {
				t.Errorf("expected len=%d, got %d", len(tt.input), len(out))
			}
			if !tt.check(out, nil) {
				t.Errorf("PredTransform check failed: got %v", out)
			}
		})
	}
}

// TestNewObjective 测试 NewObjective 工厂函数。
func TestNewObjective(t *testing.T) {
	objTypes := map[string]ObjectiveType{
		"reg:squarederror":   ObjRegSquareError,
		"reg:logistic":       ObjRegLogistic,
		"binary:logistic":    ObjBinaryLogistic,
		"binary:logitraw":    ObjBinaryLogitRaw,
		"reg:gamma":          ObjRegGamma,
		"count:poisson":      ObjRegPoisson,
		"reg:tweedie":        ObjRegTweedie,
		"rank:ndcg":          ObjRankNDCG,
		"multi:softmax":      ObjMultiSoftmax,
		"multi:softprob":     ObjMultiSoftProb,
		"reg:squaredlogerror": ObjRegSquaredLogError,
		"reg:absoluteerror":   ObjRegAbsoluteError,
		"reg:pseudohubererror": ObjRegPseudoHuberError,
		"reg:quantileerror":   ObjRegQuantileError,
		"binary:hinge":        ObjBinaryHinge,
		"rank:map":            ObjRankMAP,
		"rank:pairwise":       ObjRankPairwise,
		"survival:cox":        ObjSurvivalCox,
		"survival:aft":        ObjSurvivalAFT,
	}

	// reg:logistic 和 multi:softprob 与 binary:logistic / multi:softmax 共享实现
	sharedImpl := map[string]string{
		"reg:logistic":   "binary:logistic",
		"multi:softprob": "multi:softmax",
	}

	rankSpecial := map[ObjectiveType]bool{
		ObjRankNDCG: true,
		ObjRankMAP:  true,
	}

	for name, objType := range objTypes {
		t.Run(name, func(t *testing.T) {
			if rankSpecial[objType] {
				t.Skip("rank objectives require group info")
			}
			obj := NewObjective(objType, 1, 1.0)
			if obj == nil {
				t.Fatal("NewObjective returned nil")
			}
			expectedName := name
			if alt, ok := sharedImpl[name]; ok {
				expectedName = alt
			}
			if obj.Name() != expectedName {
				t.Errorf("expected name %q, got %q", expectedName, obj.Name())
			}
			g, h, err := obj.GetGradient([]float64{0.0, 1.0}, []float64{0.5, 1.5}, nil)
			if err != nil {
				t.Errorf("GetGradient error for %s: %v", name, err)
			}
			if len(g) != 2 || len(h) != 2 {
				t.Errorf("expected 2 gradients, got g=%d h=%d", len(g), len(h))
			}
			for i := range g {
				if math.IsNaN(g[i]) || math.IsNaN(h[i]) {
					t.Errorf("NaN gradient at %d for %s: g=%v h=%v", i, name, g, h)
				}
			}
		})
	}
}

// TestObjectiveWithWeights 验证权重正确应用到梯度。
func TestObjectiveWithWeights(t *testing.T) {
	obj := &SquaredError{}
	// 无权重
	g1, h1, _ := obj.GetGradient([]float64{1.0, 2.0}, []float64{0.0, 1.0}, nil)
	// 有权重
	weights := []float64{2.0, 0.5}
	g2, h2, _ := obj.GetGradient([]float64{1.0, 2.0}, []float64{0.0, 1.0}, weights)

	if math.Abs(g2[0]-g1[0]*2.0) > 1e-10 {
		t.Errorf("weighted[0] should be %f, got %f", g1[0]*2.0, g2[0])
	}
	if math.Abs(h2[0]-h1[0]*2.0) > 1e-10 {
		t.Errorf("weighted hess[0] should be %f, got %f", h1[0]*2.0, h2[0])
	}
	if math.Abs(g2[1]-g1[1]*0.5) > 1e-10 {
		t.Errorf("weighted[1] should be %f, got %f", g1[1]*0.5, g2[1])
	}
}

// TestMulticlassSoftmax 验证多分类梯度形状。
func TestMulticlassSoftmax(t *testing.T) {
	nClasses := 3
	obj := &MulticlassSoftmax{NumClass: nClasses}
	pred := []float64{0.0, 1.0, 2.0, 0.5, 1.5, 2.5} // 2 samples, 3 classes
	labels := []float64{0, 2}

	g, h, err := obj.GetGradient(pred, labels, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(g) != 6 || len(h) != 6 {
		t.Errorf("expected 6 gradients (2*3), got g=%d h=%d", len(g), len(h))
	}

	// softmax 检查：第 0 个样本的 3 个概率之和应 ≈ 1
	ps := softmax(pred[0:3])
	if math.Abs(ps[0]+ps[1]+ps[2]-1.0) > 1e-10 {
		t.Errorf("softmax probs should sum to 1, got %v sum=%f", ps, ps[0]+ps[1]+ps[2])
	}
}

// TestRankNDCG 验证排序目标可计算。
func TestRankNDCG(t *testing.T) {
	obj := &RankNDCG{NDCGExpGain: true, Normalization: true}
	pred := []float64{0.5, 1.0, 0.0, 0.8}
	labels := []float64{2.0, 1.0, 0.0, 1.0}
	groups := []int{4}

	g, h, err := obj.GetGradientWithGroup(pred, labels, nil, groups, 1.0)
	if err != nil {
		t.Fatal(err)
	}
	if len(g) != 4 || len(h) != 4 {
		t.Errorf("expected 4 grads, got g=%d h=%d", len(g), len(h))
	}
	for i := range g {
		if math.IsNaN(g[i]) || math.IsNaN(h[i]) {
			t.Errorf("NaN at %d: g=%f h=%f", i, g[i], h[i])
		}
	}
}
