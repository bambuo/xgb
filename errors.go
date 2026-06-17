package xgb

import "errors"

// xgb 包的哨兵错误。
var (
	ErrEmptyData           = errors.New("xgb: data is nil or empty")
	ErrNilLabels           = errors.New("xgb: labels is nil")
	ErrLabelRowsMismatch   = errors.New("xgb: labels length must equal number of rows")
	ErrInconsistentColumns = errors.New("xgb: all rows must have the same number of columns")
	ErrWeightRowsMismatch  = errors.New("xgb: weights length must equal number of rows")
	ErrMarginRowsMismatch  = errors.New("xgb: base margin length must equal number of rows")
)
