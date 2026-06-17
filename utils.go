package xgb

import "math"

// xgb 包的辅助函数。

// min 返回两个整数中的最小值。
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// max 返回两个整数中的最大值。
func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// absInt 返回整数的绝对值。
func absInt(x int) int {
	if x < 0 {
		return -x
	}
	return x
}

// clamp 将 v 裁剪到 [lo, hi] 范围内。
func clamp(v, lo, hi float64) float64 {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

// log1pSafe 返回 log(1 + x)，对负值进行保护。
func log1pSafe(x float64) float64 {
	if x <= -1 {
		return math.Inf(-1)
	}
	return math.Log(1.0 + x)
}

// nextFloat64 返回有序浮点数集合中的下一个可表示值。
func nextFloat64(x float64) float64 {
	if x == 0 {
		return math.SmallestNonzeroFloat64
	}
	bits := math.Float64bits(x)
	if x > 0 {
		bits++
	} else {
		bits--
	}
	return math.Float64frombits(bits)
}
