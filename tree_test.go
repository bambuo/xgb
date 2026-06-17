package xgb

import (
	"math"
	"testing"
)

func TestNewRegTree(t *testing.T) {
	tree := NewRegTree(10)
	if tree.Param.NumFeatures != 10 {
		t.Errorf("expected 10 features, got %d", tree.Param.NumFeatures)
	}
	if len(tree.Nodes) != 0 {
		t.Errorf("expected empty nodes, got %d", len(tree.Nodes))
	}
}

func TestRegTree_AddNode(t *testing.T) {
	tree := NewRegTree(5)

	idx0 := tree.AddNode()
	if idx0 != 0 {
		t.Errorf("expected first node index 0, got %d", idx0)
	}
	if tree.Param.NumNodes != 1 {
		t.Errorf("expected 1 node, got %d", tree.Param.NumNodes)
	}

	idx1 := tree.AddNode()
	if idx1 != 1 {
		t.Errorf("expected second node index 1, got %d", idx1)
	}
}

func TestRegTree_InitRoot(t *testing.T) {
	tree := NewRegTree(5)
	sumGrad := -10.0
	sumHess := 20.0

	idx := tree.InitRoot(sumGrad, sumHess)
	if idx != 0 {
		t.Errorf("expected root index 0, got %d", idx)
	}
	if tree.Nodes[0].SumGrad != sumGrad {
		t.Errorf("expected SumGrad %f, got %f", sumGrad, tree.Nodes[0].SumGrad)
	}
	if tree.Nodes[0].SumHess != sumHess {
		t.Errorf("expected SumHess %f, got %f", sumHess, tree.Nodes[0].SumHess)
	}
}

func TestRegTree_SetSplitAndLeaf(t *testing.T) {
	tree := NewRegTree(5)
	tree.InitRoot(0, 0)

	// 添加子节点
	leftIdx := tree.AddNode()
	rightIdx := tree.AddNode()

	// 将根节点设置为分裂节点
	tree.SetSplit(0, 3, 0.5, 1.2, leftIdx, rightIdx, false)

	node := &tree.Nodes[0]
	if node.FeatureIndex != 3 {
		t.Errorf("expected feature 3, got %d", node.FeatureIndex)
	}
	if node.Threshold != 0.5 {
		t.Errorf("expected threshold 0.5, got %f", node.Threshold)
	}
	if node.LossChange != 1.2 {
		t.Errorf("expected gain 1.2, got %f", node.LossChange)
	}

	// 将子节点设置为叶节点
	tree.SetLeaf(leftIdx, 0.8)
	tree.SetLeaf(rightIdx, -0.3)

	if !tree.Nodes[leftIdx].IsLeaf() {
		t.Error("left child should be leaf")
	}
	if tree.Nodes[leftIdx].LeafValue != 0.8 {
		t.Errorf("expected left leaf 0.8, got %f", tree.Nodes[leftIdx].LeafValue)
	}
}

func TestRegTree_Predict(t *testing.T) {
	// 构建一棵小树：
	//        [feat 0 ≤ 0.5]
	//        /            \
	//   leaf: 0.8      feat 1 ≤ 0.3
	//                   /          \
	//              leaf: -0.3   leaf: 0.2

	tree := NewRegTree(2)
	tree.InitRoot(0, 0)

	leftIdx := tree.AddNode()
	rightIdx := tree.AddNode()
	rightLeftIdx := tree.AddNode()
	rightRightIdx := tree.AddNode()

	tree.SetSplit(0, 0, 0.5, 1.0, leftIdx, rightIdx, false)
	tree.SetLeaf(leftIdx, 0.8)
	tree.SetSplit(rightIdx, 1, 0.3, 0.5, rightLeftIdx, rightRightIdx, false)
	tree.SetLeaf(rightLeftIdx, -0.3)
	tree.SetLeaf(rightRightIdx, 0.2)

	// 测试样本：feature 0 ≤ 0.5 → 左叶节点
	sample1 := []float64{0.3, 0.9}
	pred1 := tree.Predict(sample1)
	if math.Abs(pred1-0.8) > 1e-10 {
		t.Errorf("expected 0.8 for sample [0.3, 0.9], got %f", pred1)
	}

	// 测试样本：feature 0 > 0.5, feature 1 > 0.3 → 右-右叶节点
	sample2 := []float64{0.7, 0.5}
	pred2 := tree.Predict(sample2)
	if math.Abs(pred2-0.2) > 1e-10 {
		t.Errorf("expected 0.2 for sample [0.7, 0.5], got %f", pred2)
	}

	// 测试 NaN 样本 → 按默认方向（当前 DefaultLeft=false）
	// Feature 0 为 NaN → 应向右
	sample3 := []float64{math.NaN(), 0.1}
	pred3 := tree.Predict(sample3)
	if math.Abs(pred3-(-0.3)) > 1e-10 {
		t.Errorf("expected -0.3 for sample [NaN, 0.1], got %f", pred3)
	}
}

func TestRegTree_GetLeafIndex(t *testing.T) {
	tree := NewRegTree(1)
	tree.InitRoot(0, 0)
	leftIdx := tree.AddNode()
	rightIdx := tree.AddNode()

	tree.SetSplit(0, 0, 0.5, 1.0, leftIdx, rightIdx, false)
	tree.SetLeaf(leftIdx, 0.8)
	tree.SetLeaf(rightIdx, -0.3)

	idx := tree.GetLeafIndex([]float64{0.3})
	if idx != leftIdx {
		t.Errorf("expected leaf index %d, got %d", leftIdx, idx)
	}

	idx = tree.GetLeafIndex([]float64{0.7})
	if idx != rightIdx {
		t.Errorf("expected leaf index %d, got %d", rightIdx, idx)
	}
}
