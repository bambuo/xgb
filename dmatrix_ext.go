package xgb

import (
	"bufio"
	"fmt"
	"math"
	"os"
	"strconv"
	"strings"
)

// DMatrixChunked 支持外部内存训练的分块数据矩阵。
// 将大数据集按行分块，每块独立加载到内存中训练。
// 对应 XGBoost 的 ExternalMemory 模式。
type DMatrixChunked struct {
	// 文件路径列表（每个文件对应一个 chunk）
	Paths []string
	// 每块的行数限制（0 = 无限制）
	ChunkSize int
	// 特征数
	NumCols int
	// 格式："csv" 或 "libsvm"
	Format string

	// 缓存当前加载的块
	currentChunk int
	ChunkData    *DMatrix
}

// NewDMatrixChunked 创建分块数据矩阵。
// paths 是文件路径列表，每个文件作为一个 chunk。
// format 是文件格式："csv" 或 "libsvm"。
// numCols 是特征数（CSV 格式需要，LibSVM 自动检测）。
// chunkSize 是每块的最大行数（0 = 全部加载）。
func NewDMatrixChunked(paths []string, format string, numCols, chunkSize int) (*DMatrixChunked, error) {
	if len(paths) == 0 {
		return nil, fmt.Errorf("xgb: no data files provided")
	}
	dmc := &DMatrixChunked{
		Paths:     paths,
		Format:    format,
		NumCols:   numCols,
		ChunkSize: chunkSize,
	}
	return dmc, nil
}

// LoadChunk 加载指定块索引的数据。
// 块索引从 0 开始。返回 nil 表示所有块已加载完毕。
func (dmc *DMatrixChunked) LoadChunk(chunkIdx int) (*DMatrix, error) {
	if chunkIdx >= len(dmc.Paths) {
		return nil, nil
	}
	path := dmc.Paths[chunkIdx]

	switch dmc.Format {
	case "csv":
		return loadCSVChunk(path, dmc.NumCols, dmc.ChunkSize)
	case "libsvm":
		return loadLibSVMChunk(path, dmc.ChunkSize)
	default:
		return nil, fmt.Errorf("xgb: unknown format %q", dmc.Format)
	}
}

// NumChunks 返回总块数。
func (dmc *DMatrixChunked) NumChunks() int {
	return len(dmc.Paths)
}

// loadCSVChunk 从 CSV 文件加载数据，最多读取 maxRows 行。
func loadCSVChunk(path string, numCols, maxRows int) (*DMatrix, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var data [][]float64
	var labels []float64
	scanner := bufio.NewScanner(f)

	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		parts := strings.Split(line, ",")
		if len(parts) < 2 {
			continue
		}

		nFeat := len(parts) - 1
		if numCols > 0 && nFeat != numCols {
			return nil, fmt.Errorf("expected %d features, got %d", numCols, nFeat)
		}

		row := make([]float64, nFeat)
		for i := 0; i < nFeat; i++ {
			val := strings.TrimSpace(parts[i])
			if val == "" {
				row[i] = math.NaN()
			} else if fv, err := strconv.ParseFloat(val, 64); err == nil {
				row[i] = fv
			} else {
				row[i] = math.NaN()
			}
		}

		label, err := strconv.ParseFloat(strings.TrimSpace(parts[len(parts)-1]), 64)
		if err != nil {
			return nil, fmt.Errorf("invalid label: %q", parts[len(parts)-1])
		}

		data = append(data, row)
		labels = append(labels, label)

		if maxRows > 0 && len(data) >= maxRows {
			break
		}
	}

	return NewDMatrix(data, labels)
}

// loadLibSVMChunk 从 LibSVM 文件加载数据，最多读取 maxRows 行。
func loadLibSVMChunk(path string, maxRows int) (*DMatrix, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
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
			return nil, fmt.Errorf("invalid label: %q", parts[0])
		}

		cols := make(map[int]float64)
		for _, part := range parts[1:] {
			kv := strings.Split(part, ":")
			if len(kv) != 2 {
				return nil, fmt.Errorf("invalid feature: %q", part)
			}
			idx, _ := strconv.Atoi(kv[0])
			val, _ := strconv.ParseFloat(kv[1], 64)
			cols[idx] = val
			if idx > maxIdx {
				maxIdx = idx
			}
		}
		rows = append(rows, sparseRow{label: label, cols: cols})

		if maxRows > 0 && len(rows) >= maxRows {
			break
		}
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

// CountCSVRows 计算 CSV 文件的总行数（不含空行）。
func CountCSVRows(path string) (int, error) {
	f, err := os.Open(path)
	if err != nil {
		return 0, err
	}
	defer f.Close()
	scanner := bufio.NewScanner(f)
	count := 0
	for scanner.Scan() {
		if strings.TrimSpace(scanner.Text()) != "" {
			count++
		}
	}
	return count, scanner.Err()
}
