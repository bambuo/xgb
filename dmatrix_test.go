package xgb

import (
	"math"
	"testing"
)

func TestNewDMatrix(t *testing.T) {
	data := [][]float64{
		{1.0, 2.0, 3.0},
		{4.0, 5.0, 6.0},
		{7.0, 8.0, 9.0},
	}
	labels := []float64{0, 1, 0}

	dm, err := NewDMatrix(data, labels)
	if err != nil {
		t.Fatalf("NewDMatrix failed: %v", err)
	}

	if dm.NumRows != 3 {
		t.Errorf("expected 3 rows, got %d", dm.NumRows)
	}
	if dm.NumCols != 3 {
		t.Errorf("expected 3 cols, got %d", dm.NumCols)
	}
}

func TestNewDMatrix_NilData(t *testing.T) {
	_, err := NewDMatrix(nil, []float64{1})
	if err == nil {
		t.Error("expected error for nil data")
	}
}

func TestNewDMatrix_EmptyData(t *testing.T) {
	_, err := NewDMatrix([][]float64{}, []float64{1})
	if err == nil {
		t.Error("expected error for empty data")
	}
}

func TestNewDMatrix_LabelMismatch(t *testing.T) {
	data := [][]float64{{1.0, 2.0}}
	labels := []float64{0, 1}
	_, err := NewDMatrix(data, labels)
	if err == nil {
		t.Error("expected error for label/row mismatch")
	}
}

func TestNewDMatrix_InconsistentColumns(t *testing.T) {
	data := [][]float64{
		{1.0, 2.0},
		{3.0}, // only 1 column
	}
	labels := []float64{0, 1}
	_, err := NewDMatrix(data, labels)
	if err == nil {
		t.Error("expected error for inconsistent columns")
	}
}

func TestDMatrix_GetColValues(t *testing.T) {
	data := [][]float64{
		{1.0, 10.0},
		{2.0, 20.0},
		{3.0, 30.0},
	}
	dm, _ := NewDMatrix(data, []float64{0, 0, 0})

	values, indices := dm.GetColValues(0)
	if len(values) != 3 {
		t.Errorf("expected 3 values, got %d", len(values))
	}
	for i := range values {
		if indices[i] != i {
			t.Errorf("expected index %d, got %d", i, indices[i])
		}
	}
}

func TestDMatrix_HasMissing(t *testing.T) {
	data := [][]float64{
		{1.0, math.NaN()},
		{2.0, 20.0},
	}
	dm, _ := NewDMatrix(data, []float64{0, 0})

	if dm.HasMissing(0) {
		t.Error("column 0 should not have missing")
	}
	if !dm.HasMissing(1) {
		t.Error("column 1 should have missing")
	}
}
