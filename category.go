package xgb

import (
	"fmt"
	"math"
	"sort"
)

// CatEncoder 处理类别特征的编码。
type CatEncoder struct {
	// MaxCatToOnehot 是触发 one-hot 编码的最大类别数（默认 4）。
	MaxCatToOnehot int
	// KnownCats 保存每个特征的已识别类别值，key = featureIdx。
	KnownCats map[int]map[float64]int
	// CatCols 保存被识别为类别特征的特征索引。
	CatCols []int
}

// NewCatEncoder 创建类别特征编码器。
func NewCatEncoder(maxCatToOnehot int) *CatEncoder {
	return &CatEncoder{
		MaxCatToOnehot: maxCatToOnehot,
		KnownCats:      make(map[int]map[float64]int),
	}
}

// Detect 从数据中自动检测类别特征。
// 将整数类型且唯一值数量在 [2, MaxCatToOnehot] 范围内的特征视为类别。
func (ce *CatEncoder) Detect(data [][]float64, featureTypes []string) {
	if len(data) == 0 {
		return
	}
	nCols := len(data[0])

	for f := 0; f < nCols; f++ {
		// 如果指定了特征类型且不是 "categorical"，跳过
		if f < len(featureTypes) && featureTypes[f] != "" && featureTypes[f] != "categorical" {
			continue
		}

		uniq := make(map[float64]bool)
		isInt := true
		for _, row := range data {
			v := row[f]
			if math.IsNaN(v) {
				continue
			}
			uniq[v] = true
			if v != math.Trunc(v) {
				isInt = false
			}
		}

		nUniq := len(uniq)
		if nUniq >= 2 && nUniq <= ce.MaxCatToOnehot && isInt {
			catMap := make(map[float64]int)
			// 排序以确保确定性
			sortedVals := make([]float64, 0, nUniq)
			for v := range uniq {
				sortedVals = append(sortedVals, v)
			}
			sort.Float64s(sortedVals)
			for i, v := range sortedVals {
				catMap[v] = i
			}
			ce.KnownCats[f] = catMap
			ce.CatCols = append(ce.CatCols, f)
		}
	}
}

// OneHotEncode 将检测到的类别特征 one-hot 编码。
// 返回扩展后的数据矩阵和编码映射信息。
func (ce *CatEncoder) OneHotEncode(data [][]float64) ([][]float64, map[int]string) {
	if len(ce.CatCols) == 0 {
		return data, nil
	}
	nRows := len(data)
	nCols := len(data[0])

	// 计算 one-hot 扩展后的总列数
	extraCols := 0
	catMapping := make(map[int]string)
	for _, f := range ce.CatCols {
		nCats := len(ce.KnownCats[f])
		extraCols += nCats - 1 // 移除原始列，添加 nCats 个 one-hot 列（用 nCats-1 个 dummy 变量）
		catMapping[f] = fmt.Sprintf("cat_%d_%d_vals", f, nCats)
	}

	newNCols := nCols + extraCols
	newData := make([][]float64, nRows)
	for i := 0; i < nRows; i++ {
		newData[i] = make([]float64, newNCols)
	}

	// 填充数据：非类别特征保持原样，类别特征展开为 one-hot
	outCol := 0
	for f := 0; f < nCols; f++ {
		if catMap, ok := ce.KnownCats[f]; ok {
			nCats := len(catMap)
			for i := 0; i < nRows; i++ {
				v := data[i][f]
				if math.IsNaN(v) {
					// 缺失值：所有 one-hot 列为 0
					continue
				}
				if catIdx, ok := catMap[v]; ok && catIdx > 0 {
					newData[i][outCol+catIdx-1] = 1.0
				}
			}
			outCol += nCats - 1
		} else {
			for i := 0; i < nRows; i++ {
				newData[i][outCol] = data[i][f]
			}
			outCol++
		}
	}
	return newData, nil
}
