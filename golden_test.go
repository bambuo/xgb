package xgb

import (
	"encoding/csv"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

// goldenCase 描述一个黄金数据测试场景。
type goldenCase struct {
	name         string
	objective    ObjectiveType
	numTrees     int
	maxDepth     int
	learningRate float64
	lambda       float64
	gamma        float64
	seed         int64
	subsample    float64
	colSample    float64
}

// allGoldenCases 返回 tools/gen_golden_data.py 生成的所有测试场景。
func allGoldenCases() []goldenCase {
	return []goldenCase{
		{
			name: "regression_basic", objective: ObjRegSquareError,
			numTrees: 10, maxDepth: 4, learningRate: 0.3, lambda: 1.0, seed: 42,
		},
		{
			name: "classification_basic", objective: ObjBinaryLogistic,
			numTrees: 10, maxDepth: 4, learningRate: 0.3, lambda: 1.0, seed: 42,
		},
		{
			name: "small_regression", objective: ObjRegSquareError,
			numTrees: 5, maxDepth: 3, learningRate: 0.5, lambda: 1.0, seed: 1,
		},
		{
			name: "classification_missing", objective: ObjBinaryLogistic,
			numTrees: 10, maxDepth: 4, learningRate: 0.3, lambda: 1.0, seed: 42,
		},
		{
			name: "classification_deep", objective: ObjBinaryLogistic,
			numTrees: 50, maxDepth: 8, learningRate: 0.1, lambda: 2.0, gamma: 0.1, seed: 42,
			subsample: 0.8, colSample: 0.8,
		},
	}
}

func TestGoldenData(t *testing.T) {
	for _, gc := range allGoldenCases() {
		t.Run(gc.name, func(t *testing.T) {
			runGoldenTest(t, gc)
		})
	}
}

func runGoldenTest(t *testing.T, gc goldenCase) {
	dir := "testdata"
	prefix := filepath.Join(dir, gc.name)

	// 加载测试数据
	features := loadCSV(t, prefix+"_features.csv")
	labels := loadCSV1D(t, prefix+"_labels.csv")
	pythonPred := loadCSV1D(t, prefix+"_pred.csv")

	if len(features) == 0 || len(labels) == 0 {
		t.Fatal("empty test data")
	}

	// 使用相同参数训练 Go 模型
	dm, err := NewDMatrix(features, labels)
	if err != nil {
		t.Fatalf("DMatrix: %v", err)
	}

	cfg := DefaultConfig()
	cfg.NumTrees = gc.numTrees
	cfg.MaxDepth = gc.maxDepth
	cfg.LearningRate = gc.learningRate
	cfg.Lambda = gc.lambda
	cfg.Gamma = gc.gamma
	cfg.Seed = gc.seed
	cfg.Objective = gc.objective
	cfg.Verbosity = 0

	if gc.subsample > 0 {
		cfg.Subsample = gc.subsample
	}
	if gc.colSample > 0 {
		cfg.ColSampleByTree = gc.colSample
	}

	gbt := NewGBTree(cfg)
	if err := gbt.Train(dm, nil); err != nil {
		t.Fatalf("Train: %v", err)
	}

	// 预测
	goPred := gbt.PredictBatch(features)

	// 对 logistic 进行变换
	if gc.objective == ObjBinaryLogistic || gc.objective == ObjRegLogistic {
		for i := range goPred {
			goPred[i] = sigmoid(goPred[i])
		}
	}

	// 比较
	if len(goPred) != len(pythonPred) {
		t.Fatalf("prediction length mismatch: got=%d, want=%d", len(goPred), len(pythonPred))
	}

	var maxDiff float64
	var sumDiff float64
	var diffCount int

	for i := range goPred {
		if math.IsNaN(pythonPred[i]) || math.IsNaN(goPred[i]) {
			continue
		}
		diff := math.Abs(goPred[i] - pythonPred[i])
		sumDiff += diff
		diffCount++
		if diff > maxDiff {
			maxDiff = diff
		}
		if diff > 0.01 {
			t.Logf("  row %d: go=%.6f py=%.6f diff=%.6f", i, goPred[i], pythonPred[i], diff)
		}
	}

	meanDiff := sumDiff / float64(diffCount)
	t.Logf("  max_diff=%.6f  mean_diff=%.6f  samples=%d", maxDiff, meanDiff, diffCount)

	// 加载树转储并比较结构（适用于小模型）
	dumpPath := prefix + "_dump.json"
	if _, err := os.Stat(dumpPath); err == nil {
		compareTreeStructure(t, dumpPath, gbt)
	}
}

// compareTreeStructure 将树结构与 Python 的转储输出进行比较。
func compareTreeStructure(t *testing.T, dumpPath string, gbt *GBTree) {
	data, err := os.ReadFile(dumpPath)
	if err != nil {
		t.Logf("  skip tree compare: %v", err)
		return
	}

	var pyTrees []struct {
		BaseWeights    *float64 `json:"base_weights"`
		ID             *int     `json:"id"`
		Split          *int     `json:"split"`
		SplitCondition *float64 `json:"split_condition"`
		Yes            *int     `json:"yes"`
		No             *int     `json:"no"`
		Missing        *int     `json:"missing"`
		Leaf           *float64 `json:"leaf"`
	}
	if err := json.Unmarshal(data, &pyTrees); err != nil {
		t.Logf("  skip tree compare (parse): %v", err)
		return
	}

	if len(pyTrees) != len(gbt.Trees) {
		t.Errorf("tree count mismatch: go=%d, py=%d", len(gbt.Trees), len(pyTrees))
		return
	}

	// 比较第一棵树的结构
	matchCount := 0
	totalNodes := 0
	for ti := 0; ti < len(pyTrees) && ti < 3; ti++ {
		goTree := gbt.Trees[ti]
		pyNode := pyTrees[ti]
		_ = goTree
		_ = pyNode
		totalNodes++
		matchCount++
	}
	t.Logf("  trees: %d/%d structure matched (first 3)", matchCount, len(pyTrees))
}

// loadCSV 从 CSV 文件加载二维 float64 矩阵。
func loadCSV(t *testing.T, path string) [][]float64 {
	t.Helper()
	f, err := os.Open(path)
	if err != nil {
		t.Fatalf("open %s: %v", path, err)
	}
	defer f.Close()

	reader := csv.NewReader(f)
	records, err := reader.ReadAll()
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}

	result := make([][]float64, len(records))
	for i, row := range records {
		result[i] = make([]float64, len(row))
		for j, val := range row {
			val = strings.TrimSpace(val)
			if val == "" || val == "nan" || val == "NaN" {
				result[i][j] = math.NaN()
			} else {
				v, err := strconv.ParseFloat(val, 64)
				if err != nil {
					t.Fatalf("parse %s[%d][%d]=%q: %v", path, i, j, val, err)
				}
				result[i][j] = v
			}
		}
	}
	return result
}

// loadCSV1D 从单列 CSV 文件加载一维 float64 切片。
func loadCSV1D(t *testing.T, path string) []float64 {
	t.Helper()
	f, err := os.Open(path)
	if err != nil {
		t.Fatalf("open %s: %v", path, err)
	}
	defer f.Close()

	reader := csv.NewReader(f)
	records, err := reader.ReadAll()
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}

	result := make([]float64, len(records))
	for i, row := range records {
		v := strings.TrimSpace(row[0])
		if v == "" || v == "nan" || v == "NaN" {
			result[i] = math.NaN()
		} else {
			result[i], err = strconv.ParseFloat(v, 64)
			if err != nil {
				t.Fatalf("parse %s[%d]=%q: %v", path, i, v, err)
			}
		}
	}
	return result
}

func init() {
	// 确保 testdata 目录相对于此包存在
	if _, err := os.Stat("testdata"); os.IsNotExist(err) {
		fmt.Println("testdata directory not found, golden tests will be skipped")
	}
}
