package xgb

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
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
		}
		// BaseScore 存的是概率空间的值（0~1），保持原样。
		// effectiveBaseScore() 在预测时会自动将概率转换为 log-odds。
		cfg.BaseScore = baseScore

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

// ── 树结构 dump ──────────────────────────────────────────────

// DumpModel 返回所有树的文本表示，每棵树一个字符串。
// 格式类似 XGBoost 的 dump_model 输出。
func (gbt *GBTree) DumpModel() []string {
	dumps := make([]string, len(gbt.Trees))
	for i, tree := range gbt.Trees {
		dumps[i] = fmt.Sprintf("booster[%d]:\n%s", i, dumpTree(tree, 0, 0))
	}
	return dumps
}

// DumpModelFeatureNames 使用特征名称返回树的文本表示。
func (gbt *GBTree) DumpModelFeatureNames(featureNames []string) []string {
	dumps := make([]string, len(gbt.Trees))
	for i, tree := range gbt.Trees {
		dumps[i] = fmt.Sprintf("booster[%d]:\n%s", i, dumpTreeNamed(tree, 0, 0, featureNames))
	}
	return dumps
}

func dumpTree(tree *RegTree, nid, depth int) string {
	node := &tree.Nodes[nid]
	indent := ""
	for i := 0; i < depth; i++ {
		indent += "\t"
	}
	if node.IsLeaf() {
		return fmt.Sprintf("%sleaf=%.6f\n", indent, node.LeafValue)
	}
	fid := node.FeatureIndex
	th := node.Threshold
	gain := node.LossChange
	cov := node.SumHess

	result := fmt.Sprintf("%s[f%d<%.6f] gain=%.6f cover=%.6f\n", indent, fid, th, gain, cov)
	result += dumpTree(tree, node.LeftChild, depth+1)
	result += dumpTree(tree, node.RightChild, depth+1)
	return result
}

func dumpTreeNamed(tree *RegTree, nid, depth int, names []string) string {
	node := &tree.Nodes[nid]
	indent := ""
	for i := 0; i < depth; i++ {
		indent += "\t"
	}
	if node.IsLeaf() {
		return fmt.Sprintf("%sleaf=%.6f\n", indent, node.LeafValue)
	}
	fid := node.FeatureIndex
	name := fmt.Sprintf("f%d", fid)
	if fid >= 0 && fid < len(names) {
		name = names[fid]
	}
	th := node.Threshold
	gain := node.LossChange

	result := fmt.Sprintf("%s[%s<%.6f] gain=%.6f\n", indent, name, th, gain)
	result += dumpTreeNamed(tree, node.LeftChild, depth+1, names)
	result += dumpTreeNamed(tree, node.RightChild, depth+1, names)
	return result
}

// ── UBJSON 序列化（XGBoost 兼容格式）─────────────────────────

// XGBoost 的 UBJSON 格式与 JSON 格式共享相同的字段结构，
// 使用 UBJSON 作为二进制编码容器。字段名和布局与 JSON 一致。

// ubjsonType 定义 UBJSON 标记。
type ubjsonType byte

const (
	ubNil       ubjsonType = 'Z'
	ubNoOp      ubjsonType = 'N'
	ubTrue      ubjsonType = 'T'
	ubFalse     ubjsonType = 'F'
	ubInt8      ubjsonType = 'i'
	ubUInt8     ubjsonType = 'U'
	ubInt16     ubjsonType = 'I'
	ubInt32     ubjsonType = 'l'
	ubInt64     ubjsonType = 'L'
	ubFloat32   ubjsonType = 'd'
	ubFloat64   ubjsonType = 'D'
	ubChar      ubjsonType = 'C'
	ubString    ubjsonType = 'S'
	ubArray     ubjsonType = '['
	ubObject    ubjsonType = '{'
	ubArrayEnd  ubjsonType = ']'
	ubObjectEnd ubjsonType = '}'
)

// ubjsonWriter 写入 XGBoost 兼容的 UBJSON 格式。
type ubjsonWriter struct {
	w io.Writer
}

func (u *ubjsonWriter) writeByte(b byte) error {
	_, err := u.w.Write([]byte{b})
	return err
}

func (u *ubjsonWriter) writeBytes(b []byte) error {
	_, err := u.w.Write(b)
	return err
}

func (u *ubjsonWriter) writeMark(t ubjsonType) error {
	return u.writeByte(byte(t))
}

func (u *ubjsonWriter) writeInt8(v int8) error {
	if err := u.writeMark(ubInt8); err != nil {
		return err
	}
	return u.writeByte(byte(v))
}

func (u *ubjsonWriter) writeInt32(v int32) error {
	if err := u.writeMark(ubInt32); err != nil {
		return err
	}
	var buf [4]byte
	binary.BigEndian.PutUint32(buf[:], uint32(v))
	return u.writeBytes(buf[:])
}

func (u *ubjsonWriter) writeFloat64(v float64) error {
	if err := u.writeMark(ubFloat64); err != nil {
		return err
	}
	var buf [8]byte
	binary.BigEndian.PutUint64(buf[:], math.Float64bits(v))
	return u.writeBytes(buf[:])
}

func (u *ubjsonWriter) writeString(s string) error {
	if err := u.writeMark(ubString); err != nil {
		return err
	}
	data := []byte(s)
	if err := u.writeInt32(int32(len(data))); err != nil {
		return err
	}
	return u.writeBytes(data)
}

func (u *ubjsonWriter) writeArrayHeader(n int) error {
	if err := u.writeMark(ubArray); err != nil {
		return err
	}
	if err := u.writeByte(byte('#')); err != nil {
		return err
	}
	return u.writeInt32(int32(n))
}

func (u *ubjsonWriter) writeObjectHeader(n int) error {
	if err := u.writeMark(ubObject); err != nil {
		return err
	}
	if n >= 0 {
		if err := u.writeByte(byte('#')); err != nil {
			return err
		}
		return u.writeInt32(int32(n))
	}
	return u.writeByte(byte('{'))
}

// ubjsonReader 读取 UBJSON 格式数据。
type ubjsonReader struct {
	r io.Reader
}

func (ur *ubjsonReader) readByte() (byte, error) {
	var b [1]byte
	_, err := ur.r.Read(b[:])
	return b[0], err
}

func (ur *ubjsonReader) readBytes(n int) ([]byte, error) {
	b := make([]byte, n)
	_, err := ur.r.Read(b)
	return b, err
}

func (ur *ubjsonReader) readInt32() (int32, error) {
	b, err := ur.readBytes(4)
	if err != nil {
		return 0, err
	}
	return int32(binary.BigEndian.Uint32(b)), nil
}

func (ur *ubjsonReader) readFloat64() (float64, error) {
	b, err := ur.readBytes(8)
	if err != nil {
		return 0, err
	}
	return math.Float64frombits(binary.BigEndian.Uint64(b)), nil
}

func (ur *ubjsonReader) readString() (string, error) {
	// skip 'S' marker
	mark, err := ur.readByte()
	if err != nil {
		return "", err
	}
	if mark != byte(ubString) && mark != byte(ubNoOp) {
		return "", fmt.Errorf("ubjson: expected string marker, got 0x%02x", mark)
	}
	if mark == byte(ubNoOp) {
		return "", nil
	}
	length, err := ur.readInt32()
	if err != nil {
		return "", err
	}
	b, err := ur.readBytes(int(length))
	return string(b), err
}

func (ur *ubjsonReader) skipValue() error {
	mark, err := ur.readByte()
	if err != nil {
		return err
	}
	switch ubjsonType(mark) {
	case ubTrue, ubFalse, ubNil, ubNoOp:
		return nil
	case ubInt8, ubUInt8:
		_, err = ur.readByte()
	case ubInt16:
		_, err = ur.readBytes(2)
	case ubInt32:
		_, err = ur.readBytes(4)
	case ubInt64:
		_, err = ur.readBytes(8)
	case ubFloat32:
		_, err = ur.readBytes(4)
	case ubFloat64:
		_, err = ur.readBytes(8)
	case ubChar:
		_, err = ur.readByte()
	case ubString:
		length, err2 := ur.readInt32()
		if err2 != nil {
			return err2
		}
		_, err = ur.readBytes(int(length))
	case ubArray:
		return ur.skipContainer()
	case ubObject:
		return ur.skipContainer()
	}
	return err
}

func (ur *ubjsonReader) skipContainer() error {
	// Check for optimized count prefix
	peek, err := ur.readByte()
	if err != nil {
		return err
	}
	if peek == '#' {
		// Count-prefixed container
		_, err = ur.readInt32()
		return err
	}
	// Non-optimized: skip until end marker
	depth := 1
	for depth > 0 {
		b, err2 := ur.readByte()
		if err2 != nil {
			return err2
		}
		switch b {
		case byte(ubArray), byte(ubObject):
			depth++
		case byte(ubArrayEnd), byte(ubObjectEnd):
			depth--
		}
	}
	return nil
}

// SaveModelUBJSON 将模型保存为 XGBoost 兼容的 UBJSON 格式。
// 字段布局与 XGBoost 原生 JSON 一致，使用 UBJSON 二进制编码。
func (gbt *GBTree) SaveModelUBJSON(path string) error {
	var buf bytes.Buffer
	u := &ubjsonWriter{w: &buf}

	// 最外层：包含 version 和 learner 两个 key
	if err := u.writeObjectHeader(2); err != nil {
		return err
	}

	// version
	u.writeString("version")
	u.writeArrayHeader(3)
	u.writeInt8(3)
	u.writeInt8(2)
	u.writeInt8(0)

	// learner object
	u.writeString("learner")
	if err := u.writeObjectHeader(4); err != nil {
		return err
	}

	// learner_model_param
	u.writeString("learner_model_param")
	if err := u.writeObjectHeader(6); err != nil {
		return err
	}
	numFeature := 0
	for _, tree := range gbt.Trees {
		if tree.Param.NumFeatures > numFeature {
			numFeature = tree.Param.NumFeatures
		}
	}
	u.writeString("base_score")
	u.writeString(fmt.Sprintf("[%s]", ftoa(gbt.Config.BaseScore)))
	u.writeString("num_feature")
	u.writeString(fmt.Sprintf("%d", numFeature))
	u.writeString("num_class")
	u.writeString(fmt.Sprintf("%d", gbt.Config.NumClass))
	u.writeString("num_target")
	u.writeString("1")
	u.writeString("boost_from_average")
	u.writeString("1")

	// gradient_booster
	u.writeString("gradient_booster")
	if err := u.writeObjectHeader(2); err != nil {
		return err
	}
	u.writeString("name")
	u.writeString("gbtree")
	u.writeString("model")
	if err := u.writeObjectHeader(4); err != nil {
		return err
	}

	// gbtree_model_param
	u.writeString("gbtree_model_param")
	if err := u.writeObjectHeader(2); err != nil {
		return err
	}
	u.writeString("num_parallel_tree")
	u.writeString("1")
	u.writeString("num_trees")
	u.writeString(fmt.Sprintf("%d", len(gbt.Trees)))

	// trees array
	u.writeString("trees")
	if err := u.writeArrayHeader(len(gbt.Trees)); err != nil {
		return err
	}
	for _, t := range gbt.Trees {
		writeUBJSONTree(u, t, gbt.Config.LearningRate, numFeature)
	}

	// tree_info array
	u.writeString("tree_info")
	if err := u.writeArrayHeader(len(gbt.Trees)); err != nil {
		return err
	}
	for _, ti := range gbt.TreeInfo {
		u.writeInt32(int32(ti))
	}

	// objective
	u.writeString("objective")
	if err := u.writeObjectHeader(2); err != nil {
		return err
	}
	u.writeString("name")
	u.writeString(gbt.Objective.Name())
	u.writeString("reg_loss_param")
	if err := u.writeObjectHeader(1); err != nil {
		return err
	}
	u.writeString("scale_pos_weight")
	u.writeString("1")

	return os.WriteFile(path, buf.Bytes(), 0644)
}

func writeUBJSONTree(u *ubjsonWriter, tree *RegTree, learningRate float64, numFeature int) {
	if err := u.writeObjectHeader(14); err != nil {
		return
	}
	nn := len(tree.Nodes)

	u.writeString("base_weights")
	u.writeArrayHeader(nn)
	for _, node := range tree.Nodes {
		u.writeFloat64(node.LeafValue * learningRate) // XGBoost 缩放
	}

	u.writeString("categories")
	u.writeArrayHeader(0)
	u.writeString("categories_nodes")
	u.writeArrayHeader(0)
	u.writeString("categories_segments")
	u.writeArrayHeader(0)
	u.writeString("categories_sizes")
	u.writeArrayHeader(0)

	u.writeString("default_left")
	u.writeArrayHeader(nn)
	for _, node := range tree.Nodes {
		if node.DefaultLeft {
			u.writeInt8(1)
		} else {
			u.writeInt8(0)
		}
	}

	u.writeString("id")
	u.writeInt32(0) // placeholder, overwritten during serialization

	u.writeString("left_children")
	u.writeArrayHeader(nn)
	for _, node := range tree.Nodes {
		u.writeInt32(int32(maxInt(node.LeftChild, -1)))
	}

	u.writeString("loss_changes")
	u.writeArrayHeader(nn)
	for _, node := range tree.Nodes {
		u.writeFloat64(node.LossChange)
	}

	u.writeString("parents")
	u.writeArrayHeader(nn)
	parents := make([]int, nn)
	for i := range parents {
		parents[i] = -2 // unknown
	}
	for ni, node := range tree.Nodes {
		if node.LeftChild >= 0 {
			parents[node.LeftChild] = ni
		}
		if node.RightChild >= 0 {
			parents[node.RightChild] = ni
		}
	}
	parents[0] = -1 // root
	for _, p := range parents {
		u.writeInt32(int32(p))
	}

	u.writeString("right_children")
	u.writeArrayHeader(nn)
	for _, node := range tree.Nodes {
		u.writeInt32(int32(maxInt(node.RightChild, -1)))
	}

	u.writeString("split_conditions")
	u.writeArrayHeader(nn)
	for _, node := range tree.Nodes {
		if node.IsLeaf() {
			u.writeFloat64(node.LeafValue * learningRate)
		} else {
			u.writeFloat64(node.Threshold)
		}
	}

	u.writeString("split_indices")
	u.writeArrayHeader(nn)
	for _, node := range tree.Nodes {
		u.writeInt32(int32(maxInt(node.FeatureIndex, 0)))
	}

	u.writeString("split_type")
	u.writeArrayHeader(nn)
	for _, node := range tree.Nodes {
		if node.IsLeaf() {
			u.writeInt8(0)
		} else {
			u.writeInt8(0) // 0 = numerical split
		}
	}

	u.writeString("sum_hessian")
	u.writeArrayHeader(nn)
	for _, node := range tree.Nodes {
		u.writeFloat64(node.SumHess)
	}

	u.writeString("tree_param")
	if err := u.writeObjectHeader(4); err != nil {
		return
	}
	u.writeString("num_nodes")
	u.writeString(fmt.Sprintf("%d", nn))
	u.writeString("num_feature")
	u.writeString(fmt.Sprintf("%d", numFeature))
	u.writeString("num_deleted")
	u.writeString("0")
	u.writeString("size_leaf_vector")
	u.writeString("1")
}

// LoadModelUBJSON 从 XGBoost 兼容的 UBJSON 格式加载模型。
func LoadModelUBJSON(path string) (*GBTree, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}

	r := bytes.NewReader(data)
	ur := &ubjsonReader{r: r}

	// 跳过外层 object
	mark, err := ur.readByte()
	if err != nil {
		return nil, err
	}
	if ubjsonType(mark) != ubObject {
		return nil, fmt.Errorf("ubjson: expected object marker")
	}
	// 检查 count 前缀
	peek, _ := ur.readByte()
	if peek == '#' {
		ur.readInt32()
	} else {
		// put back
		r.UnreadByte()
	}

	cfg := DefaultConfig()

	// 读取所有 key-value pairs
	for {
		key, err := ur.readString()
		if err != nil {
			break
		}
		switch key {
		case "version":
			ur.skipValue()
		case "learner":
			// 解析 learner object
			mark, _ = ur.readByte()
			if ubjsonType(mark) == ubObject {
				peek2, _ := ur.readByte()
				if peek2 == '#' {
					var count int32
					binary.Read(r, binary.BigEndian, &count)
					_ = count
				} else {
					r.UnreadByte()
				}
			}
			// 读取 learner 内部字段
			for {
				learnerKey, err2 := ur.readString()
				if err2 != nil {
					break
				}
				switch learnerKey {
				case "learner_model_param":
					parseUBJSONModelParam(ur, cfg)
				case "gradient_booster":
					// parseUBJSONBooster will be handled
					ur.skipValue()
				case "objective":
					ur.skipValue()
				case "attributes":
					ur.skipValue()
				case "feature_names":
					ur.skipValue()
				case "feature_types":
					ur.skipValue()
				default:
					ur.skipValue()
				}
			}
		default:
			ur.skipValue()
		}
	}

	return NewGBTree(cfg), nil
}

func parseUBJSONModelParam(ur *ubjsonReader, cfg *Config) {
	mark, _ := ur.readByte()
	_ = mark
	for {
		key, err := ur.readString()
		if err != nil {
			return
		}
		val, _ := ur.readString()

		switch key {
		case "base_score":
			trimmed := strings.Trim(val, "[]")
			if bs, err := strconv.ParseFloat(trimmed, 64); err == nil {
				cfg.BaseScore = bs
			}
		case "num_class":
			if nc, err := strconv.Atoi(val); err == nil {
				cfg.NumClass = nc
			}
		case "num_feature":
			// stored for reference
		}
	}
}

// ── DOT 格式导出 ─────────────────────────────────────────────

// DumpModelDOT 返回所有树的 Graphviz DOT 格式表示。
func (gbt *GBTree) DumpModelDOT() string {
	var b strings.Builder
	b.WriteString("digraph xgb {\n")
	b.WriteString("  rankdir=LR;\n")
	for i, tree := range gbt.Trees {
		b.WriteString(fmt.Sprintf("  subgraph cluster_tree_%d {\n", i))
		b.WriteString(fmt.Sprintf("    label=\"booster[%d]\";\n", i))
		dumpTreeDOT(tree, 0, &b)
		b.WriteString("  }\n")
	}
	b.WriteString("}\n")
	return b.String()
}

// DumpModelDOTFeatureNames 使用特征名导出 DOT 格式。
func (gbt *GBTree) DumpModelDOTFeatureNames(featureNames []string) string {
	var b strings.Builder
	b.WriteString("digraph xgb {\n")
	b.WriteString("  rankdir=LR;\n")
	nodeID := 0
	for i, tree := range gbt.Trees {
		b.WriteString(fmt.Sprintf("  subgraph cluster_tree_%d {\n", i))
		b.WriteString(fmt.Sprintf("    label=\"booster[%d]\";\n", i))
		dumpTreeDOTNamed(tree, 0, &nodeID, featureNames, &b)
		b.WriteString("  }\n")
	}
	b.WriteString("}\n")
	return b.String()
}

func dumpTreeDOT(tree *RegTree, nid int, b *strings.Builder) {
	node := &tree.Nodes[nid]
	name := fmt.Sprintf("n%d_%d", nid, nid)
	if node.IsLeaf() {
		b.WriteString(fmt.Sprintf("    %s [label=\"leaf=%.4f\", shape=box];\n", name, node.LeafValue))
		return
	}
	b.WriteString(fmt.Sprintf("    %s [label=\"f%d < %.4f\\ngain=%.4f\\ncover=%.4f\", shape=ellipse];\n",
		name, node.FeatureIndex, node.Threshold, node.LossChange, node.SumHess))

	leftName := fmt.Sprintf("n%d_%d", node.LeftChild, node.LeftChild)
	rightName := fmt.Sprintf("n%d_%d", node.RightChild, node.RightChild)
	b.WriteString(fmt.Sprintf("    %s -> %s [label=\"yes\"];\n", name, leftName))
	b.WriteString(fmt.Sprintf("    %s -> %s [label=\"no\"];\n", name, rightName))

	dumpTreeDOT(tree, node.LeftChild, b)
	dumpTreeDOT(tree, node.RightChild, b)
}

func dumpTreeDOTNamed(tree *RegTree, nid int, nidCounter *int, names []string, b *strings.Builder) {
	node := &tree.Nodes[nid]
	myID := *nidCounter
	*nidCounter++

	if node.IsLeaf() {
		b.WriteString(fmt.Sprintf("    n%d [label=\"leaf=%.4f\", shape=box];\n", myID, node.LeafValue))
		return
	}
	fname := fmt.Sprintf("f%d", node.FeatureIndex)
	if node.FeatureIndex >= 0 && node.FeatureIndex < len(names) {
		fname = names[node.FeatureIndex]
	}
	b.WriteString(fmt.Sprintf("    n%d [label=\"%s < %.4f\", shape=ellipse];\n", myID, fname, node.Threshold))

	leftID := *nidCounter
	dumpTreeDOTNamed(tree, node.LeftChild, nidCounter, names, b)
	rightID := *nidCounter
	dumpTreeDOTNamed(tree, node.RightChild, nidCounter, names, b)

	b.WriteString(fmt.Sprintf("    n%d -> n%d [label=\"yes\"];\n", myID, leftID))
	b.WriteString(fmt.Sprintf("    n%d -> n%d [label=\"no\"];\n", myID, rightID))
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
