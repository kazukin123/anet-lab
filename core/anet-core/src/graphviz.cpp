// graphviz.cpp

#include "anet/graphviz.hpp"
#include <sstream>

using namespace anet::graphviz;


// --- Enum ToString Helpers ---

std::string ToString(LayoutDirection dir)
{
    switch (dir) {
    case LayoutDirection::TopToBottom: return "TB";
    case LayoutDirection::LeftToRight: return "LR";
    case LayoutDirection::BottomToTop: return "BT";
    case LayoutDirection::RightToLeft: return "RL";
    }
    return "TB";
}

std::string ToString(NodeShape shape)
{
    switch (shape) {
    case NodeShape::Box: return "box";
    case NodeShape::Ellipse: return "ellipse";
    case NodeShape::Circle: return "circle";
    case NodeShape::Plain: return "plain";
    case NodeShape::Diamond: return "diamond";
    }
    return "box";
}

std::string ToString(BorderStyle style)
{
    switch (style) {
    case BorderStyle::Solid: return "solid";
    case BorderStyle::Dashed: return "dashed";
    case BorderStyle::Dotted: return "dotted";
    case BorderStyle::Bold: return "bold";
    case BorderStyle::Filled: return "filled";
    }
    return "solid";
}

std::string ToString(ArrowType type)
{
    switch (type) {
    case ArrowType::Normal: return "normal";
    case ArrowType::None: return "none";
    case ArrowType::Vee: return "vee";
    case ArrowType::Dot: return "dot";
    case ArrowType::Empty: return "empty";
    }
    return "normal";
}

// HTMLエスケープヘルパー (HTML-like label内でのパースエラーを防ぐ)
std::string EscapeHtml(const std::string& str)
{
    std::string out;
    out.reserve(str.size());
    for (char c : str) {
        switch (c) {
        case '<': out += "&lt;"; break;
        case '>': out += "&gt;"; break;
        case '&': out += "&amp;"; break;
        case '"': out += "&quot;"; break;
        default:  out += c; break;
        }
    }
    return out;
}

std::string EscapeDotString(const std::string& str)
{
    std::string out;
    out.reserve(str.size());
    for (char c : str) {
        switch (c) {
        case '\\': out += "\\\\"; break;
        case '"': out += "\\\""; break;
        case '\n': out += "\\n"; break;
        case '\r': break;
        default: out += c; break;
        }
    }
    return out;
}


// ======================================================
// LabelData
// ======================================================

std::string LabelData::ToGraphvizLabel() const
{
    if (IsEmpty()) return "";
    if (!text_.empty()) return "\"" + EscapeDotString(text_) + "\"";

    std::ostringstream oss;
    oss << "<" << "<TABLE BORDER=\"0\" CELLBORDER=\"1\" CELLSPACING=\"0\">";

    // タイトル行
    if (!title_.empty()) {
        oss << "<TR><TD COLSPAN=\"2\"><B>" << EscapeHtml(title_) << "</B></TD></TR>";
    }

    // 属性行
    for (const auto& [key, value] : attributes_) {
        oss << "<TR>"
            << "<TD ALIGN=\"LEFT\">" << EscapeHtml(key) << "</TD>"
            << "<TD ALIGN=\"RIGHT\">" << EscapeHtml(value) << "</TD>"
            << "</TR>";
    }

    oss << "</TABLE>" << ">";
    return oss.str();
}


// ======================================================
// Tree
// ======================================================

Tree::Tree(const std::string& name, LayoutDirection layout)
    : name_(name), layout_(layout)
{
    // --- 規定のノードスタイルを登録 ---

    // デフォルト
    RegisterNodeStyle("default", NodeStyle{});

    // ハイライト (注目ノード用)
    NodeStyle highlight;
    highlight.border_style = BorderStyle::Bold;
    highlight.color = "#E74C3C"; // 赤系
    RegisterNodeStyle("highlight", highlight);

    // ミュート (未訪問や低価値ノード用)
    NodeStyle muted;
    muted.border_style = BorderStyle::Dashed;
    muted.color = "#95A5A6"; // グレー系
    RegisterNodeStyle("muted", muted);


    // --- 規定のエッジスタイルを登録 ---
    RegisterEdgeStyle("default", EdgeStyle{});

    EdgeStyle highlight_edge;
    highlight_edge.line_style = BorderStyle::Bold;
    highlight_edge.color = "#E74C3C";
    highlight_edge.penwidth = 2.0f;
    RegisterEdgeStyle("highlight", highlight_edge);
}

void Tree::RegisterNodeStyle(const std::string& style_id, const NodeStyle& style)
{
    node_styles_[style_id] = style;
}

void Tree::RegisterEdgeStyle(const std::string& style_id, const EdgeStyle& style)
{
    edge_styles_[style_id] = style;
}

Node& Tree::AddNode(const std::string& id, const std::string& style_id)
{
    auto node = std::make_unique<Node>(id, style_id);
    auto* ptr = node.get();
    nodes_[id] = std::move(node);
    return *ptr;
}

Edge& Tree::AddEdge(const std::string& from_id, const std::string& to_id, const std::string& style_id)
{
    auto edge = std::make_unique<Edge>(from_id, to_id, style_id);
    auto* ptr = edge.get();
    edges_.push_back(std::move(edge));
    return *ptr;
}

Node* Tree::GetNode(const std::string& id)
{
    auto it = nodes_.find(id);
    if (it != nodes_.end()) {
        return it->second.get();
    }
    return nullptr;
}

Edge* Tree::GetEdge(const std::string& from_id, const std::string& to_id)
{
    for (const auto& edge : edges_) {
        if (edge->from_node_id == from_id && edge->to_node_id == to_id) {
            return edge.get();
        }
    }
    return nullptr;
}

std::string Tree::ToDotString() const
{
    std::ostringstream dot;
    dot << "digraph \"" << name_ << "\" {\n";
    dot << "  rankdir=\"" << ToString(layout_) << "\";\n\n";

    // --- Nodes ---
    for (const auto& [id, node] : nodes_) {
        std::string target_style = node->style_id.empty() ? "default" : node->style_id;

        // 未登録スタイルの場合は default にフォールバック
        auto it = node_styles_.find(target_style);
        if (it == node_styles_.end()) {
            it = node_styles_.find("default");
        }
        const NodeStyle& style = it->second;

        dot << "  \"" << node->id << "\" [";
        dot << "shape=" << ToString(style.shape);
        dot << ", style=" << ToString(style.border_style);
        if (!style.color.empty()) dot << ", color=\"" << style.color << "\"";
        if (style.border_style == BorderStyle::Filled && !style.fillcolor.empty()) {
            dot << ", fillcolor=\"" << style.fillcolor << "\"";
        }

        std::string label_str = node->label.ToGraphvizLabel();
        if (!label_str.empty()) {
            dot << ", label=" << label_str;
        }
        dot << "];\n";
    }

    dot << "\n";

    // --- Edges ---
    for (const auto& edge : edges_) {
        std::string target_style = edge->style_id.empty() ? "default" : edge->style_id;

        auto it = edge_styles_.find(target_style);
        if (it == edge_styles_.end()) {
            it = edge_styles_.find("default");
        }
        const EdgeStyle& style = it->second;

        dot << "  \"" << edge->from_node_id << "\" -> \"" << edge->to_node_id << "\" [";
        dot << "style=" << ToString(style.line_style);
        dot << ", arrowhead=" << ToString(style.arrowhead);
        if (!style.color.empty()) dot << ", color=\"" << style.color << "\"";
        dot << ", penwidth=" << style.penwidth;
        dot << ", weight=" << edge->weight;

        std::string label_str = edge->label.ToGraphvizLabel();
        if (!label_str.empty()) {
            dot << ", label=" << label_str;
        }
        dot << "];\n";
    }

    dot << "}\n";

    return dot.str();
}

