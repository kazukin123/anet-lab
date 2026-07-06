// ImageData.hpp

#pragma once

#include <string>
#include <vector>
#include <map>
#include <fstream>
#include <stdexcept>
#include <filesystem>
#include <algorithm>
#include <torch/torch.h>
#include <wx/image.h>
#include "anet/profile.hpp"


namespace anet::img {

    class ImageDataSource : public torch::data::datasets::Dataset<ImageDataSource> {
    private:
        std::vector<std::string> image_paths_;
        std::vector<int64_t> labels_;
        std::vector<std::string> classes_;
        int target_w_;
        int target_h_;
    public:
        /**
         * コンストラクタ
         * @param root_dir 画像のルートディレクトリ（プレフィクスとして使用）
         * @param list_txt_path "train.txt" や "test.txt" へのパス
         * @param classes_txt_path クラス名一覧 "classes.txt" へのパス（行番号がIDとなる）
         * @param suffix 画像ファイルの拡張子（サフィックス。例: ".jpg"）
         */
        ImageDataSource(const std::string& root_dir, const std::string& list_txt_path, const std::string& classes_txt_path, 
            int target_w, int target_h, const std::string& suffix = ".jpg")
            : target_w_(target_w), target_h_(target_h)
        {
            ANET_PROFILE_FUNC();

            // classes.txt からクラス名とID（行番号）のマッピングを作成
            ANET_PROFILE_SCOPE(load_classes);
            std::map<std::string, int64_t> class_to_idx;
            std::ifstream cls_file(classes_txt_path);
            if (!cls_file.is_open()) {
                throw std::runtime_error("Failed to open classes file: " + classes_txt_path);
            }

            std::string line;
            int64_t current_id = 0;
            while (std::getline(cls_file, line)) {
                if (line.empty()) continue;
                line.erase(std::remove(line.begin(), line.end(), '\r'), line.end());
                line.erase(std::remove(line.begin(), line.end(), '\n'), line.end());

                class_to_idx[line] = current_id++;
                classes_.push_back(line); // ★追加: クラス名リストにも保存
            }

            // list_txt を読み込み、フルパスとラベルを決定
            ANET_PROFILE_SCOPE_NEXT(load_list);
            std::ifstream list_file(list_txt_path);
            if (!list_file.is_open()) {
                throw std::runtime_error("Failed to open list file: " + list_txt_path);
            }

            while (std::getline(list_file, line)) {
                if (line.empty()) continue;
                line.erase(std::remove(line.begin(), line.end(), '\r'), line.end());
                line.erase(std::remove(line.begin(), line.end(), '\n'), line.end());

                // "apple_pie/1005649" からクラス名 "apple_pie" を抽出
                size_t slash_pos = line.find('/');
                if (slash_pos == std::string::npos) continue;
                std::string class_name = line.substr(0, slash_pos);

                // IDの取得
                auto it = class_to_idx.find(class_name);
                if (it == class_to_idx.end()) {
                    // classes.txtに定義されていないクラスがリストにある場合
                    continue;
                }

                // パスの構築: root_dir + "apple_pie/1005649" + ".jpg"
                std::filesystem::path full_path = std::filesystem::path(root_dir) / (line + suffix);

                image_paths_.push_back(full_path.string());
                labels_.push_back(it->second);
            }
        }

        // データ抽出処理(LibTorchのDataLoaderから呼ばれる)
        torch::data::Example<> get(size_t index) override
        {
			ANET_PROFILE_FUNC();

            ANET_PROFILE_SCOPE(load_file);
            const std::string& path = image_paths_[index];

            wxImage image;
            // 第二引数を指定しない（wxBITMAP_TYPE_ANY）ことで自動判定
            if (!image.LoadFile(path)) {
                ANET_SYSTEM_ERROR("Failed to load image: " << path);
            }

			// 画像のリサイズ（必要な場合）
            ANET_PROFILE_SCOPE_NEXT(scale);
            if (target_w_ > 0 && target_h_ > 0 && (image.GetWidth() != target_w_ || image.GetHeight() != target_h_)) {
                image = image.Scale(target_w_, target_h_, wxIMAGE_QUALITY_BILINEAR);
            }

            ANET_PROFILE_SCOPE_NEXT(tensorize);
            int w = image.GetWidth();
            int h = image.GetHeight();
            unsigned char* rgb_data = image.GetData();

            // wxImageの生データ(RGB)からテンソルを作成 [H, W, C(3)]
            // 所有権を安全に移譲するため .clone()
            torch::Tensor img_tensor = torch::from_blob(rgb_data, { h, w, 3 }, torch::kByte).clone();

            // チャンネルを先頭に移動 [C, H, W]
            img_tensor = img_tensor.permute({ 2, 0, 1 });

            // ラベルをテンソル化
            torch::Tensor label_tensor = torch::tensor(labels_[index], torch::kInt64);

            return { img_tensor, label_tensor };
        }

        // データセットの総数
        std::optional<size_t> size() const override
        {
            return image_paths_.size();
        }

        // ClassID順に並んだClass名の全リストを取得
        const std::vector<std::string>& GetClassLabelList() const
        {
            return classes_;
        }

        // ClassIDからクラス名を取得
        std::string GetClassLabel(int64_t class_id) const
        {
            if (class_id >= 0 && class_id < static_cast<int64_t>(classes_.size())) {
                return classes_[class_id];
            }
            return "UNKNOWN";
        }
    };

} // namespace anet::img
