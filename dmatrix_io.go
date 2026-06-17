package xgb

import (
	"bufio"
	"fmt"
	"os"
	"strconv"
	"strings"
)

// LoadDMatrixFromCSV 从 CSV 文件加载 DMatrix。
// 文件格式：每行一个样本，最后一列为标签，其余列为特征值。
// 特征值之间用逗号分隔。缺失值用空字符串表示（解析为 NaN）。
func LoadDMatrixFromCSV(path string, numFeatures int) (*DMatrix, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open csv: %w", err)
	}
	defer f.Close()

	var data [][]float64
	var labels []float64
	scanner := bufio.NewScanner(f)
	lineNum := 0

	for scanner.Scan() {
		lineNum++
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		parts := strings.Split(line, ",")
		if len(parts) < 2 {
			continue
		}

		nFeat := len(parts) - 1
		if numFeatures > 0 && nFeat != numFeatures {
			return nil, fmt.Errorf("line %d: expected %d features, got %d", lineNum, numFeatures, nFeat)
		}

		row := make([]float64, nFeat)
		for i := 0; i < nFeat; i++ {
			val := strings.TrimSpace(parts[i])
			if val == "" {
				row[i] = nan()
			} else if fv, err := strconv.ParseFloat(val, 64); err == nil {
				row[i] = fv
			} else {
				row[i] = nan()
			}
		}

		labelStr := strings.TrimSpace(parts[len(parts)-1])
		label, err := strconv.ParseFloat(labelStr, 64)
		if err != nil {
			return nil, fmt.Errorf("line %d: invalid label %q", lineNum, labelStr)
		}

		data = append(data, row)
		labels = append(labels, label)
	}

	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("read csv: %w", err)
	}

	return NewDMatrix(data, labels)
}

// LoadDMatrixFromLibSVM 从 LibSVM 格式文件加载 DMatrix。
// 格式：label idx1:val1 idx2:val2 ...
// 缺失特征视为 0。
func LoadDMatrixFromLibSVM(path string) (*DMatrix, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open libsvm: %w", err)
	}
	defer f.Close()

	type sparseRow struct {
		label float64
		cols  map[int]float64
	}
	var rows []sparseRow
	maxIdx := 0
	scanner := bufio.NewScanner(f)

	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		parts := strings.Fields(line)
		if len(parts) < 2 {
			continue
		}

		label, err := strconv.ParseFloat(parts[0], 64)
		if err != nil {
			return nil, fmt.Errorf("invalid label %q", parts[0])
		}

		cols := make(map[int]float64)
		for _, part := range parts[1:] {
			kv := strings.Split(part, ":")
			if len(kv) != 2 {
				return nil, fmt.Errorf("invalid feature format %q", part)
			}
			idx, err := strconv.Atoi(kv[0])
			if err != nil {
				return nil, fmt.Errorf("invalid feature index %q", kv[0])
			}
			val, err := strconv.ParseFloat(kv[1], 64)
			if err != nil {
				return nil, fmt.Errorf("invalid feature value %q", kv[1])
			}
			cols[idx] = val
			if idx > maxIdx {
				maxIdx = idx
			}
		}
		rows = append(rows, sparseRow{label: label, cols: cols})
	}

	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("read libsvm: %w", err)
	}

	nRows := len(rows)
	nCols := maxIdx + 1
	data := make([][]float64, nRows)
	labels := make([]float64, nRows)

	for i, row := range rows {
		data[i] = make([]float64, nCols)
		labels[i] = row.label
		for idx, val := range row.cols {
			if idx < nCols {
				data[i][idx] = val
			}
		}
	}

	return NewDMatrix(data, labels)
}

func nan() float64 {
	return 0.0
}
