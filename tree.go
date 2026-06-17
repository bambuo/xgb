package xgb

import "math"

// RegTreeNode 表示回归树中的单个节点。
//
// 对应 XGBoost 的 RegTreeNode（include/xgboost/tree.h）。
// 节点以扁平数组形式存储在 RegTree 中，通过整数索引引用
// （与 C++ 实现一致），而非 Go 指针。
type RegTreeNode struct {
	// FeatureIndex 是分裂特征索引。值为 -1 表示叶节点。
	FeatureIndex int

	// Threshold 是分裂条件值。
	// 当 feature_value <= threshold 时样本进入左子节点。
	Threshold float64

	// LeftChild 是左子节点的节点索引（叶节点为 -1）。
	LeftChild int

	// RightChild 是右子节点的节点索引（叶节点为 -1）。
	RightChild int

	// LeafValue 是叶节点权重（仅对叶节点有效）。
	LeafValue float64

	// LossChange 是此分裂的损失变化（增益）。
	LossChange float64

	// SumGrad 是该节点的一阶梯度之和。
	SumGrad float64

	// SumHess 是该节点的二阶梯度（Hessian）之和。
	SumHess float64

	// DefaultLeft 表示缺失值的默认方向。
	// true = 默认进入左子节点，false = 默认进入右子节点。
	DefaultLeft bool
}

// IsLeaf 返回该节点是否为叶节点（无分裂）。
func (n *RegTreeNode) IsLeaf() bool {
	return n.FeatureIndex == -1
}

// RegTree 表示单棵回归树。
//
// 对应 XGBoost 的 RegTree（include/xgboost/tree.h）。
// 树使用扁平数组存储节点（XGBoost C++ 约定，而非基于指针的树），
// 这种方式缓存更友好且更容易序列化/反序列化。
type RegTree struct {
	// Nodes 以扁平数组保存所有树节点。索引 0 为根节点。
	Nodes []RegTreeNode

	// Param 保存树的元参数。
	Param TreeParam
}

// TreeParam 保存树级别的参数。
type TreeParam struct {
	NumNodes    int // 当前节点总数
	NumDeleted  int // 已删除（剪枝）节点数
	MaxDepth    int // 允许的最大深度
	NumFeatures int // 训练使用的特征数
}

// NewRegTree 为给定的特征数创建一棵新的空树。
func NewRegTree(numFeatures int) *RegTree {
	return &RegTree{
		Nodes: make([]RegTreeNode, 0, 64), // 预分配典型树大小
		Param: TreeParam{
			NumFeatures: numFeatures,
		},
	}
}

// AddNode 添加一个新节点并返回其索引。
func (t *RegTree) AddNode() int {
	idx := len(t.Nodes)
	t.Nodes = append(t.Nodes, RegTreeNode{
		FeatureIndex: -1,
		LeftChild:    -1,
		RightChild:   -1,
	})
	t.Param.NumNodes++
	return idx
}

// InitRoot 用总的梯度和 Hessian 和初始化根节点。
// 返回根节点索引（0）。
func (t *RegTree) InitRoot(sumGrad, sumHess float64) int {
	if len(t.Nodes) == 0 {
		t.AddNode()
	} else {
		t.Nodes = []RegTreeNode{{FeatureIndex: -1, LeftChild: -1, RightChild: -1}}
		t.Param.NumNodes = 1
	}
	t.Nodes[0].SumGrad = sumGrad
	t.Nodes[0].SumHess = sumHess
	return 0
}

// SetSplit 将一个节点设置为分裂（内部）节点。
// defaultLeft 指定缺失值（NaN）的默认方向：true=左子节点，false=右子节点。
func (t *RegTree) SetSplit(nodeIdx, featureIdx int, threshold, gain float64, leftIdx, rightIdx int, defaultLeft bool) {
	t.Nodes[nodeIdx].FeatureIndex = featureIdx
	t.Nodes[nodeIdx].Threshold = threshold
	t.Nodes[nodeIdx].LossChange = gain
	t.Nodes[nodeIdx].LeftChild = leftIdx
	t.Nodes[nodeIdx].RightChild = rightIdx
	t.Nodes[nodeIdx].DefaultLeft = defaultLeft
}

// SetLeaf 设置节点的叶节点值。
func (t *RegTree) SetLeaf(nodeIdx int, value float64) {
	t.Nodes[nodeIdx].FeatureIndex = -1
	t.Nodes[nodeIdx].LeafValue = value
}

// Predict 遍历树对单个样本进行预测，返回叶节点值。
// 对应 XGBoost C++ 的 RegTree::Predict()。
func (t *RegTree) Predict(sample []float64) float64 {
	if len(t.Nodes) == 0 {
		return 0
	}

	nid := 0 // 从根节点开始（节点索引 0）
	for !t.Nodes[nid].IsLeaf() {
		node := &t.Nodes[nid]
		fvalue := sample[node.FeatureIndex]

		if math.IsNaN(fvalue) {
			// 缺失值：按默认方向
			if node.DefaultLeft {
				nid = node.LeftChild
			} else {
				nid = node.RightChild
			}
		} else if fvalue <= node.Threshold {
			nid = node.LeftChild
		} else {
			nid = node.RightChild
		}

		// 安全保护：如果不小心到达不存在的节点，则跳出
		if nid >= len(t.Nodes) || nid < 0 {
			break
		}
	}

	// 如果由于数据损坏或边界情况到达了非叶节点
	if nid >= len(t.Nodes) || nid < 0 {
		return 0
	}
	return t.Nodes[nid].LeafValue
}

// GetLeafIndex 返回样本落入的叶节点索引。
func (t *RegTree) GetLeafIndex(sample []float64) int {
	nid := 0
	for !t.Nodes[nid].IsLeaf() {
		node := &t.Nodes[nid]
		fvalue := sample[node.FeatureIndex]

		if math.IsNaN(fvalue) {
			if node.DefaultLeft {
				nid = node.LeftChild
			} else {
				nid = node.RightChild
			}
		} else if fvalue <= node.Threshold {
			nid = node.LeftChild
		} else {
			nid = node.RightChild
		}

		if nid >= len(t.Nodes) || nid < 0 {
			break
		}
	}
	return nid
}
