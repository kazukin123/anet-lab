#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <sstream>
#include <iomanip>
#include <type_traits>

namespace anet::graphviz {


    // ======================================================
    // GraphViz
    // ======================================================

    class GraphViz {
    public:
        virtual std::string ToDotString() const = 0;    ///< dot形式のグラフ文字列を取得
        virtual ~GraphViz() = default;
    };


    // ======================================================
    // Enums
    // ======================================================

    enum class LayoutDirection {
        TopToBottom, // TB
        LeftToRight, // LR
        BottomToTop, // BT
        RightToLeft  // RL
    };

    enum class NodeShape {
        Box, Ellipse, Circle, Plain, Diamond
    };

    enum class BorderStyle {
        Solid, Dashed, Dotted, Bold, Filled
    };

    enum class ArrowType {
        Normal, None, Vee, Dot, Empty
    };


    // ======================================================
    // Styles
    // ======================================================

    struct NodeStyle {
        NodeShape shape = NodeShape::Box;
        BorderStyle border_style = BorderStyle::Solid;
        std::string color = "";      // 例: "black", "#FF0000"
        std::string fillcolor = "";  // border_style = Filled の時に有効
    };

    struct EdgeStyle {
        BorderStyle line_style = BorderStyle::Solid;
        ArrowType arrowhead = ArrowType::Normal;
        std::string color = "";
        float penwidth = 1.0f;
    };


    // ======================================================
    // LabelData
    // ======================================================

    class LabelData {
    public:
        LabelData& SetTitle(const std::string& title)
        {
            title_ = title;
            return *this;
        }

        template<typename T>
        LabelData& AddAttr(const std::string& key, const T& value)
        {
            std::ostringstream oss;
            // 浮動小数点は見やすさのために桁数を丸める
            if constexpr (std::is_floating_point_v<T>) {
                oss << std::fixed << std::setprecision(3) << value;
            } else {
                oss << value;
            }
            attributes_.push_back({ key, oss.str() });
            return *this;
        }

        bool IsEmpty() const { return title_.empty() && attributes_.empty(); }
        std::string ToGraphvizLabel() const;
    private:
        std::string title_;
        std::vector<std::pair<std::string, std::string>> attributes_;
    };


    // ======================================================
    // Node & Edge
    // ======================================================

    class Node {
    public:
        explicit Node(const std::string& id_val, const std::string& style = "")
            : id(id_val), style_id(style)
        {
        }

        std::string id;
        LabelData label;
        std::string style_id;
    };

    class Edge {
    public:
        Edge(const std::string& from, const std::string& to, const std::string& style = "")
            : from_node_id(from), to_node_id(to), style_id(style)
        {
        }

        std::string from_node_id;
        std::string to_node_id;
        LabelData label;
        std::string style_id;
        float weight = 1.0f;
    };

    // ======================================================
    // Tree
    // ======================================================

    class Tree final : public GraphViz {
    public:
        // 名前は必須。レイアウトはデフォルトで上から下。
        explicit Tree(const std::string& name, LayoutDirection layout = LayoutDirection::TopToBottom);

        // スタイル管理
        void RegisterNodeStyle(const std::string& style_id, const NodeStyle& style);
        void RegisterEdgeStyle(const std::string& style_id, const EdgeStyle& style);

        // 要素の追加（参照を返すため、メソッドチェーンでそのままラベル編集可能）
        Node& AddNode(const std::string& id, const std::string& style_id = "");
        Edge& AddEdge(const std::string& from_id, const std::string& to_id, const std::string& style_id = "");

        // 要素の取得
        Node* GetNode(const std::string& id);
        Edge* GetEdge(const std::string& from_id, const std::string& to_id);

        // DOT文字列の生成
        std::string ToDotString() const override;
    private:
        std::string name_;
        LayoutDirection layout_;

        std::unordered_map<std::string, NodeStyle> node_styles_;
        std::unordered_map<std::string, EdgeStyle> edge_styles_;

        std::unordered_map<std::string, std::unique_ptr<Node>> nodes_;
        std::vector<std::unique_ptr<Edge>> edges_;
    };


    // ======================================================
    // GraphVizProvider
    // ======================================================

    class GraphVizProvider {
    public:
        virtual std::unique_ptr<GraphViz> CreateGraph(const std::string& key, int64_t index) const { return nullptr; }
        virtual ~GraphVizProvider() = default;
    };


} // namespace anet::graphviz