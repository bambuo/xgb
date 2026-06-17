package xgb

import (
	"math"
)

// DMatrix 以稠密格式保存训练数据。
//
// 对应 XGBoost 的 SimpleDMatrix（src/data/simple_dmatrix.cc）。
type DMatrix struct {
	// Data 保存特征矩阵，格式为 [numRows][numCols] float64。
	// 当使用 CSR 格式创建时，此字段由 CSR 转换填充。
	Data [][]float64

	// Labels 保存目标值。
	Labels []float64

	// Weights 保存可选样本权重（nil 表示均匀权重 = 1）。
	Weights []float64

	// BaseMargin 保存可选的基础边距值（nil 表示使用 base_score）。
	BaseMargin []float64

	// Group 保存排序任务的查询组边界（每个元素存储每组的样本数）。
	// 例：Group=[3,2] 表示前 3 个样本属于组 0，后 2 个属于组 1。
	// 仅排序目标（rank:ndcg 等）使用。
	Group []int

	// NumRows 返回训练样本数。
	NumRows int

	// NumCols 返回特征数。
	NumCols int

	// FeatureNames 保存可选的每个特征名称（长度 = NumCols）。
	FeatureNames []string

	// FeatureTypes 保存可选的每个特征类型（长度 = NumCols），如 "int", "float", "categorical"。
	FeatureTypes []string
}

// SetGroup 设置排序任务的分组信息。
// groups 是每个查询组的样本数，其和必须等于 NumRows。
func (dm *DMatrix) SetGroup(groups []int) error {
	total := 0
	for _, g := range groups {
		total += g
	}
	if total != dm.NumRows {
		return ErrGroupMismatch
	}
	dm.Group = groups
	return nil
}

// CSRMatrix 是压缩稀疏行格式的矩阵。
type CSRMatrix struct {
	RowPtr []int     // 行索引，长度 numRows+1
	ColIdx []int     // 列索引
	Data   []float64 // 非零值
	Shape  [2]int    // [numRows, numCols]
}

// NewDMatrix 从稠密特征数据和标签创建新的 DMatrix。
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

	f64Data := make([][]float64, numRows)
	for i := 0; i < numRows; i++ {
		row := make([]float64, numCols)
		copy(row, data[i])
		f64Data[i] = row
	}

	f64Labels := make([]float64, numRows)
	copy(f64Labels, labels)

	return &DMatrix{
		Data:    f64Data,
		Labels:  f64Labels,
		NumRows: numRows,
		NumCols: numCols,
	}, nil
}

// NewDMatrixFromCSR 从 CSR 稀疏格式创建 DMatrix。
// CSR 格式中的缺失值视为 0，NaN 视为缺失。
func NewDMatrixFromCSR(csr *CSRMatrix, labels []float64) (*DMatrix, error) {
	if csr == nil {
		return nil, ErrEmptyData
	}
	numRows := csr.Shape[0]
	numCols := csr.Shape[1]
	if numRows == 0 || numCols == 0 {
		return nil, ErrEmptyData
	}
	if len(labels) != numRows {
		return nil, ErrLabelRowsMismatch
	}

	// 转换为稠密格式
	data := make([][]float64, numRows)
	for i := 0; i < numRows; i++ {
		data[i] = make([]float64, numCols)
		for j := csr.RowPtr[i]; j < csr.RowPtr[i+1]; j++ {
			data[i][csr.ColIdx[j]] = csr.Data[j]
		}
	}

	f64Labels := make([]float64, numRows)
	copy(f64Labels, labels)

	return &DMatrix{
		Data:    data,
		Labels:  f64Labels,
		NumRows: numRows,
		NumCols: numCols,
	}, nil
}

// SetFeatureNames 设置特征名称。
// names 长度必须等于 NumCols。
func (dm *DMatrix) SetFeatureNames(names []string) error {
	if len(names) != dm.NumCols {
		return ErrInconsistentColumns
	}
	dm.FeatureNames = make([]string, dm.NumCols)
	copy(dm.FeatureNames, names)
	return nil
}

// SetFeatureTypes 设置特征类型（如 "int", "float", "categorical"）。
// types 长度必须等于 NumCols。
func (dm *DMatrix) SetFeatureTypes(types []string) error {
	if len(types) != dm.NumCols {
		return ErrInconsistentColumns
	}
	dm.FeatureTypes = make([]string, dm.NumCols)
	copy(dm.FeatureTypes, types)
	return nil
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
