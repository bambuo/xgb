package xgb

import (
	"testing"
)

func TestDefaultConfig(t *testing.T) {
	cfg := DefaultConfig()

	if cfg.NumTrees != 100 {
		t.Errorf("expected NumTrees=100, got %d", cfg.NumTrees)
	}
	if cfg.LearningRate != 0.3 {
		t.Errorf("expected LearningRate=0.3, got %f", cfg.LearningRate)
	}
	if cfg.MaxDepth != 6 {
		t.Errorf("expected MaxDepth=6, got %d", cfg.MaxDepth)
	}
	if cfg.Lambda != 1.0 {
		t.Errorf("expected Lambda=1.0, got %f", cfg.Lambda)
	}
	if cfg.BaseScore != 0.5 {
		t.Errorf("expected BaseScore=0.5, got %f", cfg.BaseScore)
	}
}

func TestConfig_Validate(t *testing.T) {
	cfg := DefaultConfig()
	if err := cfg.Validate(); err != nil {
		t.Errorf("default config should be valid: %v", err)
	}
}

func TestConfig_Validate_NegativeNumTrees(t *testing.T) {
	cfg := DefaultConfig()
	cfg.NumTrees = 0
	if err := cfg.Validate(); err == nil {
		t.Error("expected error for NumTrees=0")
	}
}

func TestConfig_Validate_NegativeLearningRate(t *testing.T) {
	cfg := DefaultConfig()
	cfg.LearningRate = -0.1
	if err := cfg.Validate(); err == nil {
		t.Error("expected error for negative LearningRate")
	}
}

func TestConfig_Validate_Subsample(t *testing.T) {
	cfg := DefaultConfig()
	cfg.Subsample = 1.5
	if err := cfg.Validate(); err == nil {
		t.Error("expected error for Subsample > 1")
	}
	cfg.Subsample = 0
	if err := cfg.Validate(); err == nil {
		t.Error("expected error for Subsample = 0")
	}
}

func TestParseObjectiveType(t *testing.T) {
	tests := []struct {
		input    string
		expected ObjectiveType
	}{
		{"reg:squarederror", ObjRegSquareError},
		{"reg:logistic", ObjRegLogistic},
		{"binary:logistic", ObjBinaryLogistic},
		{"binary:logitraw", ObjBinaryLogitRaw},
		{"reg:gamma", ObjRegGamma},
		{"multi:softmax", ObjMultiSoftmax},
		{"multi:softprob", ObjMultiSoftProb},
	}

	for _, tt := range tests {
		got, err := ParseObjectiveType(tt.input)
		if err != nil {
			t.Errorf("ParseObjectiveType(%q) returned error: %v", tt.input, err)
		}
		if got != tt.expected {
			t.Errorf("ParseObjectiveType(%q): expected %d, got %d", tt.input, tt.expected, got)
		}
	}
}

func TestParseObjectiveType_Unknown(t *testing.T) {
	_, err := ParseObjectiveType("unknown:objective")
	if err == nil {
		t.Error("expected error for unknown objective")
	}
}

func TestObjectiveType_String(t *testing.T) {
	if ObjRegSquareError.String() != "reg:squarederror" {
		t.Errorf("unexpected string: %s", ObjRegSquareError.String())
	}
}
