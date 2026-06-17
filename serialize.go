package xgb

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"strconv"
	"strings"
)

// SafeFloat 确保 JSON 输出始终使用十进制表示法。
// Go 的 float64(0) 会序列化为 0（JSON 整数），XGBoost C++ 不接受。
type SafeFloat float64

func (f *SafeFloat) UnmarshalJSON(b []byte) error {
	var v float64
	if err := json.Unmarshal(b, &v); err != nil {
		return err
	}
	*f = SafeFloat(v)
	return nil
}

func (f SafeFloat) MarshalJSON() ([]byte, error) {
	v := float64(f)
	if v == 0 {
		return []byte("0.0"), nil
	}
	s := strconv.FormatFloat(v, 'G', -1, 64)
	for _, c := range s {
		if c == '.' || c == 'e' || c == 'E' {
			return []byte(s), nil
		}
	}
	return []byte(s + ".0"), nil
}

// ── 原生 XGBoost JSON 类型 ─────────────────────────────────

type xgbLearner struct {
	LearnerModelParam xgbLMParam `json:"learner_model_param"`
	GradientBooster   xgbGB      `json:"gradient_booster"`
	Objective         xgbObj     `json:"objective"`
}

type xgbLMParam struct {
	BaseScore        string `json:"base_score"`
	BoostFromAverage string `json:"boost_from_average"`
	NumFeature       string `json:"num_feature"`
	NumClass         string `json:"num_class"`
	NumTarget        string `json:"num_target"`
}

type xgbGB struct {
	Name  string     `json:"name"`
	Model xgbGBModel `json:"model"`
}

type xgbGBModel struct {
	GBTreeParam     xgbTreeParam `json:"gbtree_model_param"`
	Trees           []xgbTree    `json:"trees"`
	TreeInfo        []int        `json:"tree_info"`
	IterationIndptr []int        `json:"iteration_indptr"`
	Cats            xgbCats      `json:"cats"`
}

type xgbCats struct {
	Enc             []int `json:"enc"`
	FeatureSegments []int `json:"feature_segments"`
	SortedIdx       []int `json:"sorted_idx"`
}

type xgbTreeParam struct {
	NumParallelTree string `json:"num_parallel_tree"`
	NumTrees        string `json:"num_trees"`
}

type xgbObj struct {
	Name string     `json:"name"`
	RLP  xgbRegLoss `json:"reg_loss_param"`
}

type xgbRegLoss struct {
	ScalePosWeight string `json:"scale_pos_weight"`
}

// xgbTree 字段必须与 Python xgboost 的 JSON 字段顺序完全一致。
// XGBoost 3.2 的 C++ JSON 解析器对字段位置敏感。
type xgbTree struct {
	BaseWeights        []SafeFloat `json:"base_weights"`
	Categories         []int       `json:"categories"`
	CategoriesNodes    []int       `json:"categories_nodes"`
	CategoriesSegments []int       `json:"categories_segments"`
	CategoriesSizes    []int       `json:"categories_sizes"`
	DefaultLeft        []int       `json:"default_left"`
	ID                 int         `json:"id"`
	LeftChildren       []int       `json:"left_children"`
	LossChanges        []SafeFloat `json:"loss_changes"`
	Parents            []int       `json:"parents"`
	RightChildren      []int       `json:"right_children"`
	SplitConditions    []SafeFloat `json:"split_conditions"`
	SplitIndices       []int       `json:"split_indices"`
	SplitType          []int       `json:"split_type"`
	SumHessian         []SafeFloat `json:"sum_hessian"`
	TreeParam          xgbTP       `json:"tree_param"`
}

const rootParent = math.MaxInt32

type xgbTP struct {
	NumNodes       string `json:"num_nodes"`
	NumFeature     string `json:"num_feature"`
	NumDeleted     string `json:"num_deleted"`
	SizeLeafVector string `json:"size_leaf_vector"`
}

// ── 旧格式（用于向后兼容加载）───────────────────────────────

type legacyModel struct {
	Version      int          `json:"version"`
	BaseScore    float64      `json:"base_score"`
	LearningRate float64      `json:"learning_rate"`
	NumClass     int          `json:"num_class"`
	NumTrees     int          `json:"num_trees"`
	Objective    string       `json:"objective"`
	Config       *Config      `json:"config"`
	Trees        []legacyTree `json:"trees"`
	TreeInfo     []int        `json:"tree_info"`
}

type legacyTree struct {
	Nodes []legacyNode `json:"nodes"`
}

type legacyNode struct {
	ID         int     `json:"id"`
	Feature    int     `json:"feature"`
	Threshold  float64 `json:"threshold"`
	LeftChild  int     `json:"left"`
	RightChild int     `json:"right"`
	Gain       float64 `json:"gain"`
	SumGrad    float64 `json:"sum_grad"`
	SumHess    float64 `json:"sum_hess"`
	LeafValue  float64 `json:"leaf_value,omitempty"`
}

// SaveModel 将模型序列化为 XGBoost 原生 JSON 格式。
func (gbt *GBTree) SaveModel(path string) error {
	return gbt.saveModelJSON(path)
}

func (gbt *GBTree) saveModelJSON(path string) error {
	numFeature := 0
	for _, tree := range gbt.Trees {
		if tree.Param.NumFeatures > numFeature {
			numFeature = tree.Param.NumFeatures
		}
	}
	if numFeature == 0 && len(gbt.Trees) > 0 {
		numFeature = gbt.Trees[0].Param.NumFeatures
	}

	numClass := gbt.Config.NumClass
	if numClass <= 1 {
		numClass = 0 // 原生 XGBoost：0 = 单输出（非多分类）
	}

	trees := make([]xgbTree, len(gbt.Trees))
	treeInfo := make([]int, len(gbt.Trees))

	for ti, t := range gbt.Trees {
		nn := len(t.Nodes)
		xt := xgbTree{
			TreeParam: xgbTP{
				NumNodes:       itoa(nn),
				NumFeature:     itoa(numFeature),
				NumDeleted:     "0",
				SizeLeafVector: "1",
			},
			ID:                 ti,
			SplitIndices:       make([]int, nn),
			LeftChildren:       make([]int, nn),
			RightChildren:      make([]int, nn),
			Parents:            make([]int, nn),
			DefaultLeft:        make([]int, nn),
			SplitType:          make([]int, nn),
			SplitConditions:    make([]SafeFloat, nn),
			LossChanges:        make([]SafeFloat, nn),
			SumHessian:         make([]SafeFloat, nn),
			BaseWeights:        make([]SafeFloat, nn),
			Categories:         []int{},
			CategoriesNodes:    []int{},
			CategoriesSegments: []int{},
			CategoriesSizes:    []int{},
		}

		for ni, n := range t.Nodes {
			xt.SplitIndices[ni] = maxInt(n.FeatureIndex, 0)
			xt.LeftChildren[ni] = n.LeftChild
			xt.RightChildren[ni] = n.RightChild
			// XGBoost 在保存时将叶节点值乘以学习率。
			// 参见 xgboost issue #9211：“叶节点值通过学习率缩放。”
			xt.SplitConditions[ni] = SafeFloat(n.Threshold)
			xt.LossChanges[ni] = SafeFloat(n.LossChange)
			xt.SumHessian[ni] = SafeFloat(n.SumHess)
			xt.BaseWeights[ni] = SafeFloat(n.LeafValue * gbt.Config.LearningRate)
			if n.DefaultLeft {
				xt.DefaultLeft[ni] = 1
			}
			if n.LeftChild >= 0 {
				xt.Parents[n.LeftChild] = ni
			}
			if n.RightChild >= 0 {
				xt.Parents[n.RightChild] = ni
			}
		}
		xt.Parents[0] = rootParent

		trees[ti] = xt
		treeInfo[ti] = 0
		if ti < len(gbt.TreeInfo) {
			treeInfo[ti] = gbt.TreeInfo[ti]
		}
	}

	data := map[string]interface{}{
		"version": [3]int{3, 2, 0},
		"learner": map[string]interface{}{
			"attributes": map[string]interface{}{
				// 保存学习率以便加载时还原（XGBoost 原生格式不存储学习率）
				"go_learning_rate": fmt.Sprintf("%v", gbt.Config.LearningRate),
			},
			"feature_names": []string{},
			"feature_types": []string{},
			"learner_model_param": xgbLMParam{
				BaseScore:        fmt.Sprintf("[%s]", ftoa(gbt.Config.BaseScore)),
				BoostFromAverage: "1",
				NumFeature:       itoa(numFeature),
				NumClass:         itoa(numClass),
				NumTarget:        "1",
			},
			"gradient_booster": xgbGB{
				Name: "gbtree",
				Model: xgbGBModel{
					GBTreeParam: xgbTreeParam{
						NumParallelTree: "1",
						NumTrees:        itoa(len(gbt.Trees)),
					},
					Trees:           trees,
					TreeInfo:        treeInfo,
					IterationIndptr: makeIterIndptr(len(gbt.Trees)),
					Cats: xgbCats{
						Enc:             []int{},
						FeatureSegments: []int{},
						SortedIdx:       []int{},
					},
				},
			},
			"objective": xgbObj{
				Name: gbt.Objective.Name(),
				RLP:  xgbRegLoss{ScalePosWeight: "1"},
			},
		},
	}

	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	enc := json.NewEncoder(f)
	return enc.Encode(data)
}

// ── LoadModel（原生 + 旧格式）───────────────────────────────

func LoadModel(path string) (*GBTree, error) {
	return LoadModelWithLR(path, 0.3) // 默认学习率 0.3，与 XGBoost 默认值一致
}

// LoadModelWithLR 加载模型并指定学习率。
// XGBoost JSON 格式不保存学习率，因此需要调用者提供。
// 如果模型是由 Go 保存的（attributes 中包含 go_learning_rate），
// 则会自动使用保存时的学习率。传入的 learningRate 仅在原生 XGBoost 模型时生效。
func LoadModelWithLR(path string, learningRate float64) (*GBTree, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}

	// 尝试从 attributes 读取 Go 保存的学习率
	var attrHolder struct {
		Learner struct {
			Attributes map[string]interface{} `json:"attributes"`
		} `json:"learner"`
	}
	if json.Unmarshal(raw, &attrHolder) == nil {
		if lrStr, ok := attrHolder.Learner.Attributes["go_learning_rate"]; ok {
			if lrVal, err2 := strconv.ParseFloat(fmt.Sprintf("%v", lrStr), 64); err2 == nil {
				learningRate = lrVal
			}
		}
	}

	var xgbData struct {
		Version []int      `json:"version"`
		Learner xgbLearner `json:"learner"`
	}
	if err := json.Unmarshal(raw, &xgbData); err == nil && xgbData.Learner.Objective.Name != "" {
		return loadNative(xgbData.Learner, learningRate)
	}

	var legacy legacyModel
	if err := json.Unmarshal(raw, &legacy); err != nil {
		return nil, fmt.Errorf("无法识别的模型格式：%w", err)
	}
	return loadLegacy(legacy)
}

func loadNative(lm xgbLearner, learningRate float64) (*GBTree, error) {

	numClass, _ := strconv.Atoi(lm.LearnerModelParam.NumClass)
	if numClass < 1 {
		numClass = 1
	}
	baseScoreStr := strings.Trim(lm.LearnerModelParam.BaseScore, "[]")
	baseScore, _ := strconv.ParseFloat(baseScoreStr, 64)

	cfg := DefaultConfig()
	cfg.NumClass = numClass
	cfg.LearningRate = learningRate

	if objType, err := ParseObjectiveType(lm.Objective.Name); err == nil {
		cfg.Objective = objType
		// 对于二分类/回归 logistic，base_score 是类别比例，而非对数几率。
		// XGBoost 存储例如 0.356（比例），内部转换为 log(0.356/0.644) = -0.59。
		// 我们将 BaseScore 设置为原始预测的初始值。
		if objType == ObjBinaryLogistic || objType == ObjRegLogistic {
			if baseScore > 0 && baseScore < 1 {
				cfg.BaseScore = math.Log(baseScore / (1 - baseScore))
			} else {
				cfg.BaseScore = 0
			}
		} else {
			cfg.BaseScore = baseScore
		}
	} else {
		cfg.BaseScore = baseScore
	}

	gbt := NewGBTree(cfg)

	for _, xt := range lm.GradientBooster.Model.Trees {
		nn := len(xt.SplitIndices)
		tree := &RegTree{Nodes: make([]RegTreeNode, nn)}
		for ni := 0; ni < nn; ni++ {
			isLeaf := ni < len(xt.LeftChildren) && xt.LeftChildren[ni] == -1

			node := RegTreeNode{
				FeatureIndex: -1,
				LeftChild:    -1,
				RightChild:   -1,
			}

			if !isLeaf {
				node.FeatureIndex = xt.SplitIndices[ni]
				if ni < len(xt.SplitConditions) {
					node.Threshold = float64(xt.SplitConditions[ni])
				}
				node.LeftChild = xt.LeftChildren[ni]
				node.RightChild = xt.RightChildren[ni]
			}

			if ni < len(xt.BaseWeights) {
				// 撤销 XGBoost 在保存时应用的学习率缩放
				node.LeafValue = float64(xt.BaseWeights[ni]) / cfg.LearningRate
			}
			if ni < len(xt.LossChanges) {
				node.LossChange = float64(xt.LossChanges[ni])
			}
			if ni < len(xt.SumHessian) {
				node.SumHess = float64(xt.SumHessian[ni])
			}
			if ni < len(xt.DefaultLeft) {
				node.DefaultLeft = xt.DefaultLeft[ni] != 0
			}

			tree.Nodes[ni] = node
		}

		numFeat, _ := strconv.Atoi(xt.TreeParam.NumFeature)
		tree.Param = TreeParam{
			NumNodes:    nn,
			NumFeatures: numFeat,
		}
		gbt.Trees = append(gbt.Trees, tree)
	}
	gbt.TreeInfo = lm.GradientBooster.Model.TreeInfo

	return gbt, nil
}

func loadLegacy(data legacyModel) (*GBTree, error) {
	cfg := data.Config
	if cfg == nil {
		cfg = DefaultConfig()
	}
	cfg.BaseScore = data.BaseScore
	cfg.LearningRate = data.LearningRate
	cfg.NumClass = data.NumClass
	if objType, err := ParseObjectiveType(data.Objective); err == nil {
		cfg.Objective = objType
	}

	gbt := NewGBTree(cfg)

	for _, td := range data.Trees {
		tree := &RegTree{Nodes: make([]RegTreeNode, len(td.Nodes))}
		for j, nd := range td.Nodes {
			leafValue := nd.LeafValue
			if nd.Feature != -1 {
				leafValue = 0
			}
			tree.Nodes[j] = RegTreeNode{
				FeatureIndex: nd.Feature,
				Threshold:    nd.Threshold,
				LeftChild:    nd.LeftChild,
				RightChild:   nd.RightChild,
				LeafValue:    leafValue,
				LossChange:   nd.Gain,
				SumGrad:      nd.SumGrad,
				SumHess:      nd.SumHess,
			}
		}
		tree.Param.NumNodes = len(td.Nodes)
		gbt.Trees = append(gbt.Trees, tree)
	}
	gbt.TreeInfo = data.TreeInfo
	return gbt, nil
}

// ── 辅助函数 ────────────────────────────────────────────────────

func itoa(i int) string     { return strconv.Itoa(i) }
func ftoa(f float64) string { return strconv.FormatFloat(f, 'E', -1, 64) }
func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func makeIterIndptr(n int) []int {
	r := make([]int, n+1)
	for i := range r {
		r[i] = i
	}
	return r
}
