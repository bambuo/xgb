package xgb

import "math"

// ── Mersenne Twister (mt19937) PRNG ──────────────────────────
//
// 精确复刻 std::mt19937（C++ <random>），这是 GCC/Clang 上
// std::default_random_engine 的实际实现。
// XGBoost 3.2 使用 common::GlobalRandom() 返回 mt19937 引擎，
// 用于 rowSample / colSample 等子采样操作。
//
// 参考：M. Matsumoto, T. Nishimura,
// "Mersenne Twister: A 623-Dimensionally Equidistributed
// Uniform Pseudo-Random Number Generator",
// ACM Trans. on Modeling and Computer Simulation, 1998.

const (
	mtNN        = 624
	mtMM        = 397
	mtMatrixA   = 0x9908b0df
	mtUpperMask = 0x80000000
	mtLowerMask = 0x7fffffff
)

// MT19937 是 Mersenne Twister 伪随机数生成器。
type MT19937 struct {
	mt  [mtNN]uint32
	idx int
}

// NewMT19937 用给定种子创建 mt19937 生成器。
// 对应 std::mt19937::seed(seed)。
func NewMT19937(seed uint32) *MT19937 {
	r := &MT19937{}
	r.Seed(seed)
	return r
}

// Seed 重新初始化生成器状态。
func (r *MT19937) Seed(seed uint32) {
	r.mt[0] = seed
	for i := uint32(1); i < mtNN; i++ {
		r.mt[i] = 1812433253*(r.mt[i-1]^(r.mt[i-1]>>30)) + i
	}
	r.idx = mtNN
}

// Next 返回下一个 uint32 随机数。
// 对应 std::mt19937::operator()。
func (r *MT19937) Next() uint32 {
	if r.idx >= mtNN {
		r.twist()
	}

	y := r.mt[r.idx]
	r.idx++

	// Tempering
	y ^= y >> 11
	y ^= (y << 7) & 0x9d2c5680
	y ^= (y << 15) & 0xefc60000
	y ^= y >> 18

	return y
}

// Uniform 返回 [0, 1) 范围内的 float64 随机数。
func (r *MT19937) Uniform() float64 {
	return float64(r.Next()) / 4294967296.0 // 2^32
}

// twist 执行 twist 操作以生成下一批 mtNN 个随机数。
func (r *MT19937) twist() {
	for i := 0; i < mtNN; i++ {
		y := (r.mt[i] & mtUpperMask) | (r.mt[(i+1)%mtNN] & mtLowerMask)
		r.mt[i] = r.mt[(i+mtMM)%mtNN] ^ (y >> 1)
		if y&1 != 0 {
			r.mt[i] ^= mtMatrixA
		}
	}
	r.idx = 0
}

// ── 基于 mt19937 的采样函数 ─────────────────────────────────

// mtRowSample 使用 mt19937 为子采样生成行索引的随机子集。
// 算法与 XGBoost 的 bernoulli_distribution(ratio) 采样一致：
// 对每个样本生成一个 [0,1) 随机数，小于 ratio 则选中。
func mtRowSample(numRows int, ratio float64, seed uint32) []int {
	if ratio >= 1.0 {
		indices := make([]int, numRows)
		for i := range indices {
			indices[i] = i
		}
		return indices
	}

	rng := NewMT19937(seed)
	indices := make([]int, 0, int(float64(numRows)*ratio))

	for i := 0; i < numRows; i++ {
		if rng.Uniform() < ratio {
			indices = append(indices, i)
		}
	}
	return indices
}

// mtColSample 使用 mt19937 为列子采样生成特征索引的随机子集。
func mtColSample(numCols int, ratio float64, seed uint32) []int {
	if ratio >= 1.0 {
		features := make([]int, numCols)
		for i := range features {
			features[i] = i
		}
		return features
	}

	rng := NewMT19937(seed)
	features := make([]int, 0, int(float64(numCols)*ratio))

	for i := 0; i < numCols; i++ {
		if rng.Uniform() < ratio {
			features = append(features, i)
		}
	}
	return features
}

// deriveSampleSeed 从主种子派生独立子种子，用于不同迭代和采样操作。
//
// 在 XGBoost C++ 中，所有采样操作共享一个全局 mt19937 引擎
// （common::GlobalRandom()），按顺序推进。这里我们用确定性派生
// 来模拟独立的随机流，保证不同迭代（iter）和操作类型（kind）
// 产生可重现的独立随机序列。
//
// 参数：
//   - seed:  主种子（Config.Seed）
//   - iter:  当前提升轮次
//   - kind:  操作类型（0=行采样, 1=列采样, 2+=其他）
//
// 使用 SplitMix64 风格混合，确保种子空间充分扩散。
func deriveSampleSeed(seed int64, iter int, kind uint32) uint32 {
	// SplitMix64 派生：混合种子、迭代和操作类型
	x := uint64(seed) + uint64(iter)*6364136223846793005 + uint64(kind)*1442695040888963407
	x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9
	x = (x ^ (x >> 27)) * 0x94d049bb133111eb
	x = x ^ (x >> 31)
	return uint32(x)
}

// mtRowSampleGradient 使用基于梯度幅度的采样（gradient_based）。
// 采样概率与 |grad| 成正比，使高梯度样本有更高的被选中概率。
func mtRowSampleGradient(numRows int, ratio float64, seed uint32, grads []float64) []int {
	if ratio >= 1.0 || len(grads) == 0 {
		return mtRowSample(numRows, ratio, seed)
	}

	// 计算梯度绝对值作为权重
	weights := make([]float64, numRows)
	var totalWeight float64
	for i := 0; i < numRows && i < len(grads); i++ {
		w := math.Abs(grads[i])
		weights[i] = w
		totalWeight += w
	}
	if totalWeight <= 0 {
		return mtRowSample(numRows, ratio, seed)
	}

	// 归一化为概率
	for i := range weights {
		weights[i] /= totalWeight
	}

	rng := NewMT19937(seed)
	indices := make([]int, 0, int(float64(numRows)*ratio))
	targetCount := int(float64(numRows) * ratio)
	if targetCount < 1 {
		targetCount = 1
	}

	// 加权有放回采样
	for len(indices) < targetCount {
		r := rng.Uniform()
		var cum float64
		for i, w := range weights {
			cum += w
			if r < cum {
				indices = append(indices, i)
				break
			}
		}
	}

	return indices
}

// mtColSampleWeighted 使用特征权重的列采样。
func mtColSampleWeighted(numCols int, ratio float64, seed uint32, featureWeights []float64) []int {
	if ratio >= 1.0 {
		features := make([]int, numCols)
		for i := range features {
			features[i] = i
		}
		return features
	}

	if len(featureWeights) < numCols {
		return mtColSample(numCols, ratio, seed)
	}

	// 计算总权重
	var totalWeight float64
	for i := 0; i < numCols; i++ {
		if featureWeights[i] > 0 {
			totalWeight += featureWeights[i]
		}
	}
	if totalWeight <= 0 {
		return mtColSample(numCols, ratio, seed)
	}

	rng := NewMT19937(seed)
	features := make([]int, 0, int(float64(numCols)*ratio))

	for i := 0; i < numCols; i++ {
		// 每个特征的选中概率 = (ratio * featureWeight / meanWeight) 但有界
		meanWeight := totalWeight / float64(numCols)
		p := ratio * (featureWeights[i] / meanWeight)
		if p > 1.0 {
			p = 1.0
		}
		if rng.Uniform() < p {
			features = append(features, i)
		}
	}

	// 确保至少选中一个特征
	if len(features) == 0 && numCols > 0 {
		features = append(features, int(rng.Uniform()*float64(numCols)))
	}

	return features
}

// mtColSampleFromMask 使用 mt19937 从已有特征掩码中进行子采样。
func mtColSampleFromMask(mask []int, ratio float64, seed uint32) []int {
	if mask == nil || ratio >= 1.0 {
		return mask
	}

	rng := NewMT19937(seed)
	features := make([]int, 0, int(float64(len(mask))*ratio))

	for _, f := range mask {
		if rng.Uniform() < ratio {
			features = append(features, f)
		}
	}
	return features
}
