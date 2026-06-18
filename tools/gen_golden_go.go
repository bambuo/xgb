//go:build ignore

package main

import (
	"encoding/csv"
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"strconv"

	xgb "github.com/bambuo/xgb"
)

// ── 配置 ───────────────────────────────────────────────────────

type goldenConfig struct {
	Name         string
	N, M         int
	NumTrees     int
	MaxDepth     int
	LearningRate float64
	Lambda       float64
	Gamma        float64
	Subsample    float64
	ColSample    float64
	Seed         int64
	Objective    xgb.ObjectiveType
	DataGen      func(n, m int, rng *rand.Rand) ([][]float64, []float64)
}

func main() {
	genAll()
}

func genAll() {
	// 相对于 tools/ 目录，testdata 在上一级
	outputDir := filepath.Join("..", "testdata")
	if _, err := os.Stat(outputDir); os.IsNotExist(err) {
		outputDir = filepath.Join("testdata")
	}
	os.MkdirAll(outputDir, 0755)

	configs := []goldenConfig{
		{
			Name: "regression_basic", N: 200, M: 5,
			NumTrees: 10, MaxDepth: 4, LearningRate: 0.3,
			Lambda: 1.0, Seed: 42,
			Objective: xgb.ObjRegSquareError,
			DataGen:   genLinearRegression,
		},
		{
			Name: "classification_basic", N: 300, M: 5,
			NumTrees: 10, MaxDepth: 4, LearningRate: 0.3,
			Lambda: 1.0, Seed: 42,
			Objective: xgb.ObjBinaryLogistic,
			DataGen:   genBinaryClassification,
		},
		{
			Name: "small_regression", N: 20, M: 3,
			NumTrees: 5, MaxDepth: 3, LearningRate: 0.5,
			Lambda: 1.0, Seed: 1,
			Objective: xgb.ObjRegSquareError,
			DataGen:   genLinearRegression,
		},
		{
			Name: "classification_missing", N: 200, M: 5,
			NumTrees: 10, MaxDepth: 4, LearningRate: 0.3,
			Lambda: 1.0, Seed: 42,
			Objective: xgb.ObjBinaryLogistic,
			DataGen:   genClassificationMissing,
		},
		{
			Name: "classification_deep", N: 500, M: 10,
			NumTrees: 50, MaxDepth: 8, LearningRate: 0.1,
			Lambda: 2.0, Gamma: 0.1, Subsample: 0.8, ColSample: 0.8,
			Seed: 42,
			Objective: xgb.ObjBinaryLogistic,
			DataGen:   genClassificationDeep,
		},
	}

	for _, cfg := range configs {
		generate(cfg, outputDir)
	}
	fmt.Println("✓ All golden data generated successfully")
}

// ── 数据生成 ────────────────────────────────────────────────────

func genLinearRegression(n, m int, rng *rand.Rand) ([][]float64, []float64) {
	features := make([][]float64, n)
	labels := make([]float64, n)
	for i := 0; i < n; i++ {
		features[i] = make([]float64, m)
		for j := 0; j < m; j++ {
			features[i][j] = rng.NormFloat64()
		}
		// y = 2*x0 - x1 + 0.5*x2 + noise*0.1
		labels[i] = 2*features[i][0] - features[i][1] + 0.5*features[i][2] + rng.NormFloat64()*0.1
	}
	return features, labels
}

func genBinaryClassification(n, m int, rng *rand.Rand) ([][]float64, []float64) {
	features := make([][]float64, n)
	labels := make([]float64, n)
	for i := 0; i < n; i++ {
		features[i] = make([]float64, m)
		for j := 0; j < m; j++ {
			features[i][j] = rng.NormFloat64()
		}
		if features[i][0]+features[i][1] > 0 {
			labels[i] = 1.0
		} else {
			labels[i] = 0.0
		}
	}
	return features, labels
}

func genClassificationMissing(n, m int, rng *rand.Rand) ([][]float64, []float64) {
	features := make([][]float64, n)
	labels := make([]float64, n)
	for i := 0; i < n; i++ {
		features[i] = make([]float64, m)
		for j := 0; j < m; j++ {
			features[i][j] = rng.NormFloat64()
		}
		// 10% 概率设为 NaN
		for j := 0; j < m; j++ {
			if rng.Float64() < 0.1 {
				features[i][j] = math.NaN()
			}
		}
		// nanmean(features[i][0:2]) > 0
		var sum, count float64
		for j := 0; j < 2 && j < m; j++ {
			if !math.IsNaN(features[i][j]) {
				sum += features[i][j]
				count++
			}
		}
		if count > 0 && sum/count > 0 {
			labels[i] = 1.0
		}
	}
	return features, labels
}

func genClassificationDeep(n, m int, rng *rand.Rand) ([][]float64, []float64) {
	features := make([][]float64, n)
	labels := make([]float64, n)
	for i := 0; i < n; i++ {
		features[i] = make([]float64, m)
		for j := 0; j < m; j++ {
			features[i][j] = rng.NormFloat64()
		}
		// y = (x0*x1 + x2 + sin(x3) + noise*0.2 > 0.5)
		noise := rng.NormFloat64() * 0.2
		val := features[i][0]*features[i][1] + features[i][2] + math.Sin(features[i][3]) + noise
		if val > 0.5 {
			labels[i] = 1.0
		}
	}
	return features, labels
}

// ── 生成单个用例 ────────────────────────────────────────────────

func generate(cfg goldenConfig, outputDir string) {
	prefix := filepath.Join(outputDir, cfg.Name)

	// 1. 生成数据
	rng := rand.New(rand.NewSource(cfg.Seed))
	features, labels := cfg.DataGen(cfg.N, cfg.M, rng)

	// 2. 训练模型
	dm, err := xgb.NewDMatrix(features, labels)
	if err != nil {
		panic(fmt.Sprintf("%s: NewDMatrix: %v", cfg.Name, err))
	}

	modelCfg := xgb.DefaultConfig()
	modelCfg.NumTrees = cfg.NumTrees
	modelCfg.MaxDepth = cfg.MaxDepth
	modelCfg.LearningRate = cfg.LearningRate
	modelCfg.Lambda = cfg.Lambda
	modelCfg.Gamma = cfg.Gamma
	modelCfg.Seed = cfg.Seed
	modelCfg.Objective = cfg.Objective
	modelCfg.Verbosity = 0
	if cfg.Subsample > 0 {
		modelCfg.Subsample = cfg.Subsample
	}
	if cfg.ColSample > 0 {
		modelCfg.ColSampleByTree = cfg.ColSample
	}

	gbt := xgb.NewGBTree(modelCfg)
	if _, err := gbt.Train(dm, nil); err != nil {
		panic(fmt.Sprintf("%s: Train: %v", cfg.Name, err))
	}

	// 3. 预测（raw margin）
	predictions := gbt.PredictBatch(features)

	// 4. 对 logistic 目标应用 sigmoid（与 Python predict() 行为一致）
	if cfg.Objective == xgb.ObjBinaryLogistic || cfg.Objective == xgb.ObjRegLogistic {
		for i := range predictions {
			predictions[i] = 1.0 / (1.0 + math.Exp(-predictions[i]))
		}
	}

	// 5. 保存黄金数据
	saveCSV(prefix+"_golden_features.csv", features)
	saveCSV1D(prefix+"_golden_labels.csv", labels)
	saveCSV1D(prefix+"_golden_pred.csv", predictions)

	// 6. 保存模型
	modelPath := prefix + "_golden_model.json"
	if err := gbt.SaveModel(modelPath); err != nil {
		panic(fmt.Sprintf("%s: SaveModel: %v", cfg.Name, err))
	}

	fmt.Printf("  %s: %ds x %df, %d trees\n", cfg.Name, cfg.N, cfg.M, cfg.NumTrees)
}

// ── CSV 读写工具 ────────────────────────────────────────────────

func saveCSV(path string, data [][]float64) {
	f, err := os.Create(path)
	if err != nil {
		panic(err)
	}
	defer f.Close()
	w := csv.NewWriter(f)
	for _, row := range data {
		rec := make([]string, len(row))
		for j, v := range row {
			if math.IsNaN(v) {
				rec[j] = "NaN"
			} else {
				rec[j] = strconv.FormatFloat(v, 'f', 18, 64)
			}
		}
		w.Write(rec)
	}
	w.Flush()
	if err := w.Error(); err != nil {
		panic(err)
	}
}

func saveCSV1D(path string, data []float64) {
	f, err := os.Create(path)
	if err != nil {
		panic(err)
	}
	defer f.Close()
	w := csv.NewWriter(f)
	for _, v := range data {
		if math.IsNaN(v) {
			w.Write([]string{"NaN"})
		} else {
			w.Write([]string{strconv.FormatFloat(v, 'f', 18, 64)})
		}
	}
	w.Flush()
	if err := w.Error(); err != nil {
		panic(err)
	}
}
