#pragma once
#include <string>
#include <unordered_map>
#include <fstream>
#include <stdexcept>

// ---- key 結合 ("group" + "." + field) ----
static inline std::string PROPS_JOIN(const std::string& group, const char* field) {
    if (!group.empty() && group.back() == '.')
        return group + field;
    return group + "." + field;
}

// ---- 読み込みマクロ（group と field を指定する）----
#ifndef ANET_READ_PROPS
#define ANET_READ_PROPS(props, group, field) \
    (props)->Read(PROPS_JOIN(std::string(group), #field), (field), (field))
#endif

//#include <vector>
//#include <unordered_map>
//#include <stdexcept>
//
//template <class Key, class Value>
//class OrderedMap {
//public:
//    using KV = std::pair<Key, Value*>;
//
//    class const_iterator {
//    public:
//        using vec_it = typename std::vector<KV>::const_iterator;
//        explicit const_iterator(vec_it it) : it_(it) {}
//
//        std::pair<const Key&, const Value&> operator*()  const {
//            return { it_->first, *it_->second };
//        }
//        const std::pair<const Key&, const Value&>* operator->() const {
//            cache_ = { it_->first, *it_->second };
//            return &cache_;
//        }
//
//        const_iterator& operator++() { ++it_; return *this; }
//        bool operator==(const const_iterator& rhs) const { return it_ == rhs.it_; }
//        bool operator!=(const const_iterator& rhs) const { return it_ != rhs.it_; }
//
//    private:
//        vec_it it_;
//        mutable std::pair<const Key&, const Value&> cache_;
//    };
//
//    const_iterator begin() const { return const_iterator(order.begin()); }
//    const_iterator end()   const { return const_iterator(order.end()); }
//
//    const_iterator find(const Key& key) const {
//        for (auto it = order.begin(); it != order.end(); ++it)
//            if (it->first == key)
//                return const_iterator(it);
//        return end();
//    }
//
//    const Value& operator[](const Key& key) const {
//        return map.at(key);
//    }
//
//    Value& operator[](const Key& key) {
//        Value& ref = map[key]; // map が唯一の本体
//
//        for (auto& kv : order)
//            if (kv.first == key)
//                return ref;
//
//        order.emplace_back(key, &ref);
//        return ref;
//    }
//
//    void insert(const Key& key, const Value& value) {
//        map[key] = value; // 値更新
//
//        for (auto it = order.begin(); it != order.end(); ++it)
//            if (it->first == key) { order.erase(it); break; }
//
//        order.emplace_back(key, &map[key]); // ✅ Value のアドレスを使う
//    }
//
//private:
//    std::vector<KV> order;               // 挿入順：Key + Value*
//    std::unordered_map<Key, Value> map;  // 値本体はここだけ
//};


// linked_hash_map.h
#pragma once

#include <cstddef>
#include <list>
#include <unordered_map>
#include <utility>
// linked_hash_map.h
#pragma once

#include <cstddef>
#include <list>
#include <unordered_map>
#include <utility>

template <class K, class V>
class LinkedHashMap {
public:
    explicit LinkedHashMap(bool access_order = false) : access_order_(access_order) {}

    std::size_t size() const { return map_.size(); }
    bool empty() const { return map_.empty(); }
    void clear() {
        map_.clear();
        order_.clear();
    }

    // ==== iterator ====
    struct iterator {
        using list_iterator = typename std::list<K>::iterator;
        iterator(list_iterator it, LinkedHashMap* owner) : it_(it), owner_(owner) {}

        iterator& operator++() { ++it_; return *this; }
        bool operator==(const iterator& rhs) const { return it_ == rhs.it_; }
        bool operator!=(const iterator& rhs) const { return it_ != rhs.it_; }

        std::pair<const K&, V&> operator*() const {
            auto mit = owner_->map_.find(*it_);
            return { mit->first, mit->second.first };
        }

        list_iterator it_;
        LinkedHashMap* owner_;
    };

    struct const_iterator {
        using list_iterator = typename std::list<K>::const_iterator;
        const_iterator(list_iterator it, const LinkedHashMap* owner) : it_(it), owner_(owner) {}

        const_iterator& operator++() { ++it_; return *this; }
        bool operator==(const const_iterator& rhs) const { return it_ == rhs.it_; }
        bool operator!=(const const_iterator& rhs) const { return it_ != rhs.it_; }

        std::pair<const K&, const V&> operator*() const {
            auto mit = owner_->map_.find(*it_);
            return { mit->first, mit->second.first };
        }

        list_iterator it_;
        const LinkedHashMap* owner_;
    };

    iterator begin() { return iterator(order_.begin(), this); }
    iterator end() { return iterator(order_.end(), this); }

    const_iterator begin() const { return const_iterator(order_.cbegin(), this); }
    const_iterator end()   const { return const_iterator(order_.cend(), this); }

    const_iterator cbegin() const { return const_iterator(order_.cbegin(), this); }
    const_iterator cend()   const { return const_iterator(order_.cend(), this); }

    // ==== find (修正版) ====
    iterator find(const K& key) {
        auto it = map_.find(key);
        if (it == map_.end()) return end();
        return iterator(it->second.second, this);
    }

    const_iterator find(const K& key) const {
        auto it = map_.find(key);
        if (it == map_.end()) return end();
        return const_iterator(it->second.second, this);
    }

    bool contains(const K& key) const {
        return map_.find(key) != map_.end();
    }

    V& operator[](const K& key) {
        auto it = map_.find(key);
        if (it == map_.end()) {
            order_.push_back(key);
            auto list_it = std::prev(order_.end());
            auto inserted = map_.emplace(key, MapValue(V{}, list_it));
            return inserted.first->second.first;
        }
        if (access_order_) Touch(it);
        return it->second.first;
    }

    void put(const K& key, const V& value) {
        auto it = map_.find(key);
        if (it != map_.end()) {
            it->second.first = value;
            if (access_order_) Touch(it);
            return;
        }
        order_.push_back(key);
        map_.emplace(key, MapValue(value, std::prev(order_.end())));
    }

    V* get(const K& key) {
        auto it = map_.find(key);
        if (it == map_.end()) return nullptr;
        if (access_order_) Touch(it);
        return &it->second.first;
    }

    bool erase(const K& key) {
        auto it = map_.find(key);
        if (it == map_.end()) return false;
        order_.erase(it->second.second);
        map_.erase(it);
        return true;
    }

private:
    using ListIter = typename std::list<K>::iterator;
    using MapValue = std::pair<V, ListIter>;
    using Map = std::unordered_map<K, MapValue>;

    void Touch(typename Map::iterator it) {
        order_.splice(order_.end(), order_, it->second.second);
    }

    bool access_order_;
    std::list<K> order_;
    Map map_;
};


class Properties {
public:
    explicit Properties(const std::string& filename) {
        Load(filename);
    }

    void Set(const std::string& key, const std::string& value) {
        kv_[key] = value;
    }

    bool Has(const std::string& key) const {
        return kv_.find(key) != kv_.end();
    }

public:
    // --- std::string 用オーバーロード ---
    std::string Get(const std::string& key, const std::string& defaultValue = "") const
    {
        auto it = kv_.find(key);
        return (it != kv_.end()) ? (*it).second : defaultValue;
    }

    // --- 汎用テンプレート ---
    template<typename T>
    T Get(const std::string& key, const T& defaultValue = T()) const
    {
        auto it = kv_.find(key);
        if (it == kv_.end()) return defaultValue;

        std::istringstream iss((*it).second);
        T value;
        iss >> value;
        //if (!(iss >> value)) return defaultValue;
        return value;
    }

public:

    void Read(const std::string& key, std::string& value, const std::string& defaultValue) const {
        auto it = kv_.find(key);
        value = (it != kv_.end()) ? (*it).second:defaultValue;
    }

    void Read(const std::string& key, int& value, int defaultValue) const {
        auto it = kv_.find(key);
        if (it == kv_.end()) { value = defaultValue; return; }
        try { value = std::stoi((*it).second); }
        catch (...) { value = defaultValue; }
    }

    void Read(const std::string& key, float& value, float defaultValue) const {
        auto it = kv_.find(key);
        if (it == kv_.end()) { value = defaultValue; return; }
        try { value = std::stof((*it).second); }
        catch (...) { value = defaultValue; }
    }

    void Read(const std::string& key, double& value, double defaultValue) const {
        auto it = kv_.find(key);
        if (it == kv_.end()) { value = defaultValue; return; }
        try { value = std::stod((*it).second); }
        catch (...) { value = defaultValue; }
    }

    void Read(const std::string& key, bool& value, bool defaultValue) const {
        auto it = kv_.find(key);
        if (it == kv_.end()) { value = defaultValue; return; }
        const auto& v = (*it).second;
        if (v == "true" || v == "TRUE" || v == "1" || v == "yes" || v == "on") { value = true; return; }
        if (v == "false" || v == "FALSE" || v == "0" || v == "no" || v == "off") { value = false; return; }
        value = defaultValue;
    }

private:
    LinkedHashMap<std::string, std::string> kv_;

    static std::string Trim(const std::string& s) {
        const char* ws = " \t\r\n";
        size_t b = s.find_first_not_of(ws);
        if (b == std::string::npos) return "";
        size_t e = s.find_last_not_of(ws);
        return s.substr(b, e - b + 1);
    }

    void Load(const std::string& filename) {
        std::ifstream ifs(filename);
        if (!ifs) throw std::runtime_error("Properties: Cannot open: " + filename);

        std::string line;
        while (std::getline(ifs, line)) {
			if (line.empty() || line[0] == '#') continue; // コメント行スキップ

			size_t posHash = line.find('#');   // '#' 以降はコメント扱い
            if (posHash != std::string::npos)
                line = line.substr(0, posHash);

			size_t posSlash = line.find("//");  // '//' 以降はコメント扱い
            if (posSlash != std::string::npos)
                line = line.substr(0, posSlash);

			line = Trim(line);  // 前後の空白除去
            if (line.empty()) continue;

			size_t pos = line.find('=');    // '=' または ':' で区切る
            if (pos == std::string::npos)
                pos = line.find(':');
            if (pos == std::string::npos)
                continue;

            std::string key = Trim(line.substr(0, pos));
            std::string value = Trim(line.substr(pos + 1));

            // 末尾 ';' を除去
            while (!value.empty() && value.back() == ';')
                value.pop_back();
            value = Trim(value);

            if (!key.empty())
                kv_[key] = value;
        }
    }
};

// サンプル
//struct RLAgent::Param {
//    static constexpr const char* GROUP = "agent";
//
//    float alpha = 1e-3f;
//    float gamma = 0.99f;
//
//    void Load(const Properties& props) {
//        ANET_READ_PROPS(props, GROUP, alpha);
//        ANET_READ_PROPS(props, GROUP, gamma);
//    }
//};

