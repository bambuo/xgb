package xgb

import (
	"math"
)

// DMatrix 以稠密格式保存训练数据。
//
// 对应 XGBoost 的 SimpleDMatrix（src/data/simple_dmatrix.cc）。
type DMatrix struct {
	// Data 保存特征矩阵，格式为 [numRows][numCols] float64。
	Data [][]float64

	// Labels 保存目标值。
	Labels []float64

	// Weights 保存可选样本权重（nil 表示均匀权重 = 1）。
	Weights []float64

	// BaseMargin 保存可选的基础边距值（nil 表示使用 base_score）。
	BaseMargin []float64

	// NumRows 返回训练样本数。
	NumRows int

	// NumCols 返回特征数。
	NumCols int
}

// NewDMatrix 从特征数据和标签创建新的 DMatrix。
// data 格式为 [numRows][numCols]，labels 长度必须等于 numRows。
func NewDMatrix(data [][]float64, labels []float64) (*DMatrix, error) {
	if data == nil || len(data) == 0 {
		return nil, ErrEmptyData
	}
	if labels == nil {
		return nil, ErrNilLabels
	}

	numRows := len(data)
	if len(labels) != numRows {
		return nil, ErrLabelRowsMismatch
	}

	numCols := 0
	if numRows > 0 {
		numCols = len(data[0])
		for i := 1; i < numRows; i++ {
			if len(data[i]) != numCols {
				return nil, ErrInconsistentColumns
			}
		}
	}

	return &DMatrix{
		Data:    data,
		Labels:  labels,
		NumRows: numRows,
		NumCols: numCols,
	}, nil
}

// SetWeights 为 DMatrix 设置可选的每个样本权重。
func (dm *DMatrix) SetWeights(weights []float64) error {
	if len(weights) != dm.NumRows {
		return ErrWeightRowsMismatch
	}
	dm.Weights = weights
	return nil
}

// SetBaseMargin 为 DMatrix 设置基础边距。
func (dm *DMatrix) SetBaseMargin(margin []float64) error {
	if len(margin) != dm.NumRows {
		return ErrMarginRowsMismatch
	}
	dm.BaseMargin = margin
	return nil
}

// GetColValues 提取指定特征列的所有值。
// 用于排序分裂搜索，返回特征值及其对应的梯度/Hessian 值，按特征值排序。
func (dm *DMatrix) GetColValues(featureIdx int) ([]float64, []int) {
	values := make([]float64, dm.NumRows)
	indices := make([]int, dm.NumRows)
	for i := 0; i < dm.NumRows; i++ {
		values[i] = dm.Data[i][featureIdx]
		indices[i] = i
	}
	return values, indices
}

// HasMissing 检查给定的特征列是否包含 NaN 值。
func (dm *DMatrix) HasMissing(featureIdx int) bool {
	for i := 0; i < dm.NumRows; i++ {
		if math.IsNaN(dm.Data[i][featureIdx]) {
			return true
		}
	}
	return false
}
