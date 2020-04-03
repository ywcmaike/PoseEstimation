#include <iostream>

#include <Eigen/Core>

#include "base/camera.h"
#include "base/camera_models.h"
#include "base/pose.h"
#include "estimators/pose.h"
#include "pose/pose_estimation.h"
#include "util/logging.h"
#include "util/misc.h"
#include "util/io.h"
#include "util/random.h"

std::vector<std::pair<std::string, PE::Camera>> ReadAachen(const std::string &path) {
    std::ifstream file(path, std::ios::in);
    CHECK(file.is_open()) << path << std::endl;

    std::vector<std::pair<std::string, PE::Camera>> imcams;
    std::string line;
    while (getline(file, line)) {
        std::stringstream ss(line);
        std::string name, cam_type;
        double w, h, f, cx, cy, r;
        ss >> name >> cam_type >> w >> h >> f >> cx >> cy >> r;

        // construct camera
        PE::Camera camera;
        camera.SetWidth(static_cast<size_t>(w));
        camera.SetHeight(static_cast<size_t>(h));
        camera.SetModelId(PE::SimpleRadialCameraModel::model_id);
        camera.SetParams(std::vector<double>{f, cx, cy, r});

        // construct item
        imcams.emplace_back(std::make_pair(name, camera));
    }
    return imcams;
}

std::string ExtractIndetification(const std::string &path) {
    std::string s = path;
    size_t  pos = s.find_last_of('/'/*, s.find_last_of('/')-1*/);
    return s.substr(pos+1);
}

int main(int argc, char **argv) {
    if (argc != 5) {
        std::cout << "Usage: solve_pose correspondence_path dataset iteration_num result_path" << std::endl;
        return 1;
    }

    std::string correspondence_path = argv[1];
    std::string dataset_path = argv[2];
    std::string result_path = argv[4];
    double inlier_ratio = std::stod(argv[3]);

    PE::AbsolutePoseEstimationOptions options;
    options.ransac_options.max_error = 12;
    options.ransac_options.min_num_trials = 1;
    options.ransac_options.confidence = 0.9999;
    options.ransac_options.min_inlier_ratio = inlier_ratio;


    std::map<std::string, std::pair<Eigen::Vector4d, Eigen::Vector3d>> results;

    auto dataset = ReadAachen(dataset_path);
    PE::SetPRNGSeed(99995039);

    for (size_t i = 0; i != dataset.size(); ++i) {
        auto item_path = dataset[i].first;
        auto camera = dataset[i].second;

        auto item_dir = PE::JoinPaths(correspondence_path, item_path);

        auto files = PE::GetFileList(item_dir);
        std::sort(files.begin(), files.end());

        // best
        size_t best_inlier_num = 0;
        Eigen::Vector4d best_qvec;
        Eigen::Vector3d best_tvec;

        std::cout << "[" << i+1 << "/" << dataset.size() << "]" << item_path << ": ";
        for (size_t j = 0; j != files.size(); ++j) {
            // data
            std::vector<Eigen::Vector2d> points2D;
            std::vector<Eigen::Vector3d> points3D;
            std::vector<double> priors;
            PE::ReadCorresspondencesTxt(files[j], points2D, points3D);

            for (size_t k = 0; k != points2D.size(); ++k) {
                points2D[k] = camera.WorldToImage(points2D[k]);
            }

            Eigen::Vector4d qvec;
            Eigen::Vector3d tvec;
            std::vector<char> mask;

            size_t inlier_num = 0;
            inlier_num = PE::RansacPnP(points2D, points3D, camera, qvec, tvec, mask, options);
            // inlier_num = PE::WeightedRansacPnP(points2D, points3D, priors, camera, qvec, tvec, mask, options);
            // inlier_num = PE::ProsacPnP(points2D, points3D, priors, camera, qvec, tvec, mask, options);

            if (inlier_num == 0) continue;

            inlier_num = PE::RefinePose(points2D, points3D, camera, qvec, tvec, mask);
            // inlier_num = PE::RefinePoseWithPrior(points2D, points3D, priors, camera, qvec, tvec, mask);

            std::cout << inlier_num << " ";

            if (inlier_num > best_inlier_num) {
                best_inlier_num = inlier_num;
                best_qvec = qvec;
                best_tvec = tvec;
            }

            if (best_inlier_num >= 100) break;
        }
        std::cout << std::endl;
        auto indentity = ExtractIndetification(item_path);
        results[indentity] = std::make_pair(best_qvec, best_tvec);
    }

    std::ofstream file(result_path);
    for (auto result : results) {
        auto qvec = result.second.first;
        auto tvec = result.second.second;
        file << result.first << " ";
        file << qvec[0] << " " << qvec[1] << " " << qvec[2] << " " << qvec[3] << " ";
        file << tvec[0] << " " << tvec[1] << " " << tvec[2] << std::endl;
    }
    file.close();

    return 0;
}