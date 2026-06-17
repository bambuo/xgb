package xgb

import (
	"fmt"
	"strings"
)

// PlotLearningCurve 绘制训练指标的学习曲线（ASCII 格式）。
// chartWidth 和 chartHeight 是字符宽度和高度。
// 返回字符串可用于终端打印或日志记录。
func PlotLearningCurve(history []EvalResult, metricName string, width, height int) string {
	if len(history) == 0 || width < 10 || height < 3 {
		return ""
	}

	points := LearningCurve(history, metricName)
	if len(points) == 0 {
		return ""
	}

	// 提取数值
	vals := make([]float64, len(points))
	for i, p := range points {
		if v, ok := p.Metrics[metricName]; ok {
			vals[i] = v
		}
	}
	if len(vals) == 0 {
		return ""
	}

	// 找到最小/最大值
	minVal, maxVal := vals[0], vals[0]
	for _, v := range vals {
		if v < minVal {
			minVal = v
		}
		if v > maxVal {
			maxVal = v
		}
	}

	// 留出边距
	if maxVal-minVal < 1e-10 {
		minVal -= 0.5
		maxVal += 0.5
	}
	rangeVal := maxVal - minVal

	// 构建图表
	plotAreaW := width - 10 // 留出 Y 轴标注空间
	if plotAreaW < 2 {
		plotAreaW = 2
	}
	plotAreaH := height - 2 // 留出 X 轴标注空间
	if plotAreaH < 1 {
		plotAreaH = 1
	}

	var b strings.Builder
	b.WriteString(fmt.Sprintf("=== Learning Curve: %s ===\n", metricName))
	b.WriteString(fmt.Sprintf("min=%.4f  max=%.4f  iter=0..%d\n", minVal, maxVal, len(vals)-1))

	// 构建栅格
	grid := make([][]byte, plotAreaH)
	for y := range grid {
		grid[y] = make([]byte, plotAreaW)
		for x := range grid[y] {
			grid[y][x] = ' '
		}
	}

	// 映射点
	for i, v := range vals {
		x := i * (plotAreaW - 1) / (len(vals) - 1)
		if x >= plotAreaW {
			x = plotAreaW - 1
		}
		y := int((v - minVal) / rangeVal * float64(plotAreaH-1))
		if y < 0 {
			y = 0
		}
		if y >= plotAreaH {
			y = plotAreaH - 1
		}
		grid[plotAreaH-1-y][x] = '*'
	}

	// 绘制
	for y := 0; y < plotAreaH; y++ {
		// Y 轴标签
		label := ""
		if y == 0 {
			label = fmt.Sprintf("%.3f ", maxVal)
		} else if y == plotAreaH-1 {
			label = fmt.Sprintf("%.3f ", minVal)
		} else {
			label = "        "
		}
		b.WriteString(label)
		for x := 0; x < plotAreaW; x++ {
			b.WriteByte(grid[y][x])
		}
		b.WriteByte('\n')
	}

	// X 轴
	b.WriteString("         ")
	for x := 0; x < plotAreaW; x++ {
		if x%(plotAreaW/5) == 0 || x == 0 {
			b.WriteRune('├')
		} else {
			b.WriteRune('─')
		}
	}
	b.WriteByte('\n')

	// X 轴标签
	tickSpacing := (len(vals) - 1) / 5
	if tickSpacing < 1 {
		tickSpacing = 1
	}
	b.WriteString("         ")
	for x := 0; x < plotAreaW; x++ {
		iter := x * (len(vals) - 1) / (plotAreaW - 1)
		if x == 0 || iter%(tickSpacing*2) == 0 {
			label := fmt.Sprintf("%d", iter)
			b.WriteString(label[:minInt(len(label), 3)])
			if len(label) < 3 {
				for i := 0; i < 3-len(label); i++ {
					b.WriteByte(' ')
				}
			}
			x += 2
		}
	}
	b.WriteByte('\n')

	return b.String()
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// PrintLearningCurve 在终端打印学习曲线。
func PrintLearningCurve(history []EvalResult, metricName string) {
	fmt.Println(PlotLearningCurve(history, metricName, 60, 12))
}

// ── 特征重要性 ASCII 条形图 ─────────────────────────────────

// PlotImportance 绘制特征重要性的 ASCII 条形图。
// ranking 是 ImportanceRanking() 的返回值。
// topN 指定显示前 N 个特征（0 = 全部）。
// 返回字符串。
func PlotImportance(ranking []FeatureImportance, topN int, featureNames []string) string {
	if len(ranking) == 0 {
		return "(no feature importance data)"
	}
	if topN <= 0 || topN > len(ranking) {
		topN = len(ranking)
	}

	// 找到最高分数用于归一化
	maxScore := ranking[0].Score
	if maxScore <= 0 {
		maxScore = 1
	}

	barMax := 40 // 最大条形宽度
	var b strings.Builder
	b.WriteString("Feature Importance Ranking:\n")

	for i := 0; i < topN; i++ {
		fi := ranking[i].FeatureIndex
		name := fmt.Sprintf("f%d", fi)
		if fi >= 0 && fi < len(featureNames) && featureNames[fi] != "" {
			name = featureNames[fi]
		}
		barLen := int(ranking[i].Score / maxScore * float64(barMax))
		if barLen < 1 && ranking[i].Score > 0 {
			barLen = 1
		}
		bar := strings.Repeat("█", barLen)
		b.WriteString(fmt.Sprintf("  #%d  %-12s %s %.4f\n", i+1, name, bar, ranking[i].Score))
	}

	return b.String()
}

// PrintImportance 在终端打印特征重要性条形图。
func PrintImportance(ranking []FeatureImportance, topN int, featureNames []string) {
	fmt.Print(PlotImportance(ranking, topN, featureNames))
}
