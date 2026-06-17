package xgb

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
