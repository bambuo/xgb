package xgb

import (
	"math"
	"testing"
)

// TestMetricValues 测试所有评估指标的正确值。
func TestMetricValues(t *testing.T) {
	dm, _ := NewDMatrix([][]float64{{1}, {2}, {3}, {4}}, []float64{1, 2, 3, 4})
	preds := []float64{1.0, 2.0, 3.0, 4.0} // 完美预测

	t.Run("rmse_perfect", func(t *testing.T) {
		v := (&RMSEMetric{}).Evaluate(preds, dm)
		if v != 0 {
			t.Errorf("expected 0, got %f", v)
		}
	})
	t.Run("rmse_off", func(t *testing.T) {
		v := (&RMSEMetric{}).Evaluate([]float64{2, 3, 4, 5}, dm)
		expected := math.Sqrt(4.0 / 4.0) // (1+1+1+1)/4
		if math.Abs(v-expected) > 1e-10 {
			t.Errorf("expected %f, got %f", expected, v)
		}
	})
	t.Run("mae_perfect", func(t *testing.T) {
		v := (&MAEMetric{}).Evaluate(preds, dm)
		if v != 0 {
			t.Errorf("expected 0, got %f", v)
		}
	})
	t.Run("mae_off", func(t *testing.T) {
		v := (&MAEMetric{}).Evaluate([]float64{2, 3, 4, 5}, dm)
		if math.Abs(v-1.0) > 1e-10 {
			t.Errorf("expected 1.0, got %f", v)
		}
	})
	t.Run("rmsle_perfect", func(t *testing.T) {
		v := (&RMSLEMetric{}).Evaluate(preds, dm)
		if v != 0 {
			t.Errorf("expected 0, got %f", v)
		}
	})
	t.Run("logloss_perfect", func(t *testing.T) {
		dm2, _ := NewDMatrix([][]float64{{1}, {2}}, []float64{0, 1})
		v := (&LogLossMetric{}).Evaluate([]float64{0.001, 0.999}, dm2)
		if v > 0.01 {
			t.Errorf("expected near 0, got %f", v)
		}
	})
	t.Run("logloss_worst", func(t *testing.T) {
		dm2, _ := NewDMatrix([][]float64{{1}, {2}}, []float64{0, 1})
		v := (&LogLossMetric{}).Evaluate([]float64{0.999, 0.001}, dm2)
		if v < 5.0 {
			t.Errorf("expected high logloss for wrong preds, got %f", v)
		}
	})
	t.Run("error_perfect", func(t *testing.T) {
		dm2, _ := NewDMatrix([][]float64{{1}, {2}}, []float64{0, 1})
		v := (&ErrorMetric{}).Evaluate([]float64{0.1, 0.9}, dm2)
		if v != 0 {
			t.Errorf("expected 0, got %f", v)
		}
	})
	t.Run("error_50pct", func(t *testing.T) {
		dm2, _ := NewDMatrix([][]float64{{1}, {2}}, []float64{0, 0})
		v := (&ErrorMetric{}).Evaluate([]float64{0.9, 0.1}, dm2)
		if math.Abs(v-0.5) > 1e-10 {
			t.Errorf("expected 0.5, got %f", v)
		}
	})
	t.Run("auc_perfect", func(t *testing.T) {
		dm3, _ := NewDMatrix([][]float64{{1}, {2}, {3}, {4}}, []float64{0, 0, 1, 1})
		v := (&AUCMetric{}).Evaluate([]float64{0.1, 0.2, 0.9, 0.8}, dm3)
		if v < 0.99 {
			t.Errorf("expected ~1.0, got %f", v)
		}
	})
	t.Run("aucpr_perfect", func(t *testing.T) {
		dm3, _ := NewDMatrix([][]float64{{1}, {2}, {3}, {4}}, []float64{0, 0, 1, 1})
		v := (&AUCPRMetric{}).Evaluate([]float64{0.1, 0.2, 0.9, 0.8}, dm3)
		if v < 0.99 {
			t.Errorf("expected ~1.0, got %f", v)
		}
	})
}

// TestParseMetricAll 测试 ParseMetric 能解析所有指标。
func TestParseMetricAll(t *testing.T) {
	metricNames := []string{
		"rmse", "mae", "rmsle", "mape", "mphe",
		"logloss", "error", "error@t", "merror", "mlogloss",
		"auc", "aucpr", "ndcg", "map", "pre",
		"poisson-nloglik", "gamma-nloglik", "gamma-deviance",
		"tweedie-nloglik", "cox-nloglik",
	}
	for _, name := range metricNames {
		t.Run(name, func(t *testing.T) {
			m, ok := ParseMetric(name)
			if !ok {
				t.Errorf("ParseMetric(%q) failed", name)
			}
			if m.Name() != name {
				t.Errorf("expected name %q, got %q", name, m.Name())
			}
			// 验证 Evaluate 可调用
			dm, _ := NewDMatrix([][]float64{{1}, {2}}, []float64{0.5, 1.5})
			v := m.Evaluate([]float64{0.6, 1.4}, dm)
			if math.IsNaN(v) {
				t.Errorf("Evaluate returned NaN for %s", name)
			}
		})
	}
}

// TestMultiClassMetrics 测试多分类指标。
func TestMultiClassMetrics(t *testing.T) {
	t.Run("merror", func(t *testing.T) {
		dm, _ := NewDMatrix([][]float64{{1}, {2}, {3}}, []float64{0, 1, 2})
		// 3 samples, 3 classes: preds flattened as [sample][class]
		preds := []float64{
			0.9, 0.05, 0.05, // correct -> class 0
			0.1, 0.8, 0.1, // correct -> class 1
			0.3, 0.3, 0.4, // wrong -> class 2 predicted, but class 0 or 1 could be wrong
		}
		v := (&MErrorMetric{}).Evaluate(preds, dm)
		// 第 3 个样本 label=2, preds[2]=0.3,0.3,0.4, max at class 2 → correct
		// All correct
		if v != 0 {
			t.Errorf("expected 0, got %f", v)
		}
	})
	t.Run("mlogloss", func(t *testing.T) {
		dm, _ := NewDMatrix([][]float64{{1}, {2}}, []float64{0, 1})
		preds := []float64{
			0.9, 0.1, // correct
			0.2, 0.8, // correct
		}
		v := (&MLogLossMetric{}).Evaluate(preds, dm)
		if math.IsNaN(v) || v < 0 {
			t.Errorf("expected positive logloss, got %f", v)
		}
	})
}

// TestRankingMetrics 测试排序指标。
func TestRankingMetrics(t *testing.T) {
	dm, _ := NewDMatrix([][]float64{{1}, {2}, {3}}, []float64{2, 1, 0})
	preds := []float64{0.9, 0.5, 0.1} // perfect ranking

	t.Run("ndcg_perfect", func(t *testing.T) {
		v := (&NDCGMetric{K: 3}).Evaluate(preds, dm)
		if math.Abs(v-1.0) > 1e-10 {
			t.Errorf("expected 1.0, got %f", v)
		}
	})
	t.Run("ndcg_wrong", func(t *testing.T) {
		v := (&NDCGMetric{K: 3}).Evaluate([]float64{0.1, 0.5, 0.9}, dm)
		if v > 1.0 || v < 0 {
			t.Errorf("expected [0,1], got %f", v)
		}
	})
	t.Run("precision", func(t *testing.T) {
		dm2, _ := NewDMatrix([][]float64{{1}, {2}, {3}, {4}}, []float64{1, 0, 1, 0})
		v := (&PrecisionMetric{K: 2}).Evaluate([]float64{0.9, 0.8, 0.1, 0.0}, dm2)
		// top 2 preds: both relevant (0.9, 0.8 correspond to labels 1,0)
		if v < 0.49 || v > 0.51 {
			t.Errorf("expected ~0.5, got %f", v)
		}
	})
}

// TestStatisticalMetrics 测试统计指标。
func TestStatisticalMetrics(t *testing.T) {
	dm, _ := NewDMatrix([][]float64{{1}, {2}, {3}}, []float64{1, 2, 3})
	preds := []float64{1.1, 2.1, 3.1}

	t.Run("mape", func(t *testing.T) {
		v := (&MAPEMetric{}).Evaluate(preds, dm)
		if v <= 0 {
			t.Errorf("expected positive MAPE, got %f", v)
		}
	})
	t.Run("poisson-nloglik", func(t *testing.T) {
		v := (&PoissonNLogLikMetric{}).Evaluate(preds, dm)
		if math.IsNaN(v) {
			t.Errorf("expected finite, got NaN")
		}
	})
	t.Run("gamma-nloglik", func(t *testing.T) {
		v := (&GammaNLogLikMetric{}).Evaluate(preds, dm)
		if math.IsNaN(v) {
			t.Errorf("expected finite, got NaN")
		}
	})
	t.Run("gamma-deviance", func(t *testing.T) {
		v := (&GammaDevianceMetric{}).Evaluate(preds, dm)
		if math.IsNaN(v) {
			t.Errorf("expected finite, got NaN")
		}
	})
	t.Run("cox-nloglik", func(t *testing.T) {
		dm2, _ := NewDMatrix([][]float64{{1}, {2}, {3}}, []float64{1, 0, 1})
		v := (&CoxNLogLikMetric{}).Evaluate([]float64{0.5, 1.0, 1.5}, dm2)
		if math.IsNaN(v) {
			t.Errorf("expected finite, got NaN")
		}
	})
	t.Run("tweedie-nloglik", func(t *testing.T) {
		v := (&TweedieNLogLikMetric{VariancePower: 1.5}).Evaluate(preds, dm)
		if math.IsNaN(v) {
			t.Errorf("expected finite, got NaN")
		}
	})
	t.Run("mphe", func(t *testing.T) {
		v := (&MPHMetric{Delta: 1.0}).Evaluate(preds, dm)
		if v <= 0 {
			t.Errorf("expected positive MPHE, got %f", v)
		}
	})
}
