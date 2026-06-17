package xgb

// xgb 包的辅助函数。

// min 返回两个整数中的最小值。
func min(a, b int) int {
	if a < b {
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

// toF32 将 float64 截断为 float32 精度再返回 float64。
// XGBoost 内部使用 float（32位）存储梯度对（GradientPair），
// 因此每次梯度计算后需要截断以对齐舍入行为。
func toF32(v float64) float64 {
	return float64(float32(v))
}

// truncateGradients 将梯度/Hessian 数组截断为 float32 精度。
func truncateGradients(grads, hess []float64) {
	for i := range grads {
		grads[i] = toF32(grads[i])
		hess[i] = toF32(hess[i])
	}
}
