#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <json/json.h>
#include <yaml-cpp/yaml.h>
#include <opencv2/opencv.hpp>

#define CAMCHAIN_FILE "2023-05-03-23-11-22-camchain.yaml"
#define IMAGE_FILE "output/images/test/*.png"
#define POSE_FILE "pos.json"

typedef std::vector<std::vector<double>> Matrix;

Matrix camchain2nerf(const std::string& filepath) {
    YAML::Node data = YAML::LoadFile(filepath);
    Matrix intrin(4, std::vector<double>(4, 0.0));
    Matrix dist;

    intrin[0][0] = data["cam0"]["intrinsics"][0].as<double>();
    intrin[1][1] = data["cam0"]["intrinsics"][1].as<double>();
    intrin[0][2] = data["cam0"]["intrinsics"][2].as<double>();
    intrin[1][2] = data["cam0"]["intrinsics"][3].as<double>();

    const YAML::Node& dist_coeffs = data["cam0"]["distortion_coeffs"];
    for (const auto& coeff : dist_coeffs) {
        dist.push_back({ coeff.as<double>() });
    }

    return std::make_pair(intrin, dist);
}

std::vector<std::string> images2nerf(const std::string& image_file_paths) {
    std::vector<std::string> image_paths;
    cv::glob(image_file_paths, image_paths);
    cv::Mat cv_img = cv::imread(image_paths[0]);
    int h = cv_img.rows;
    int w = cv_img.cols;

    return std::make_pair(image_paths, std::make_pair(h, w));
}

Matrix pos2nerf(const std::string& json_file) {
    std::ifstream file(json_file);
    Json::Value json_pos;
    file >> json_pos;

    Matrix Plist;

    const Json::Value& poses = json_pos["poses"];
    const Json::Value& orientations = json_pos["orientations"];

    for (int i = 0; i < orientations.size(); ++i) {
        Matrix ori = orientations[i].as<Matrix>();
        ori.push_back({ 0, 0, 0 });
        Matrix pos = poses[i].as<Matrix>();
        pos.push_back({ 1 });

        Matrix P(ori.size(), std::vector<double>(ori[0].size() + pos[0].size(), 0.0));
        for (int row = 0; row < ori.size(); ++row) {
            for (int col = 0; col < ori[row].size(); ++col) {
                P[row][col] = ori[row][col];
            }
        }
        for (int row = 0; row < pos.size(); ++row) {
            for (int col = 0; col < pos[row].size(); ++col) {
                P[row][col + ori[row].size()] = pos[row][col];
            }
        }

        Plist.push_back(P);
    }

    return Plist;
}

void integrate() {
    std::map<std::string, double> data_format;
    Matrix intrin, dist;
    std::tie(intrin, dist) = camchain2nerf(CAMCHAIN_FILE);

    double k1 = dist[0][0];
    double k2 = dist[1][0];
    double p1 = dist[2][0];
    double p2 = dist[3][0];
    double fl_x = intrin[0][0];
    double fl_y = intrin[1][0];
    double cx = intrin[2][0];
    double cy = intrin[3][0];

    std::vector<std::string> file_paths;
    int h, w;
    std::tie(file_paths, std::tie(h, w)) = images2nerf(IMAGE_FILE);

    double camera_angle_x = 2 * std::tanh(2 * fl_x / w);
    double camera_angle_y = 2 * std::tanh(2 * fl_y / h);

    data_format["camera_angle_x"] = camera_angle_x;
    data_format["camera_angle_y"] = camera_angle_y;
    data_format["fl_x"] = fl_x;
    data_format["fl_y"] = fl_y;
    data_format["k1"] = k1;
    data_format["k2"] = k2;
    data_format["p1"] = p1;
    data_format["p2"] = p2;
    data_format["cx"] = cx;
    data_format["cy"] = cy;
    data_format["w"] = w;
    data_format["h"] = h;

    Matrix Plist = pos2nerf(POSE_FILE);
    std::vector<std::map<std::string, std::string>> frame_list;
    for (int i = 0; i < file_paths.size(); ++i) {
        std::map<std::string, std::string> frame;
        frame["file_path"] = file_paths[i];
        frame["transform_matrix"] = Json::FastWriter().write(Plist[i]);
        frame_list.push_back(frame);
    }
    data_format["frames"] = frame_list;

    Json::Value json_data;
    for (const auto& entry : data_format) {
        json_data[entry.first] = entry.second;
    }

    std::ofstream output_file("data_format_half.json");
    output_file << json_data;
    output_file.close();
