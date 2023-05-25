#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/aruco/dictionary.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <nlohmann/json.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
using namespace std;
using namespace cv;
using json = nlohmann::json;

cv::Mat camera_pose_board(std::vector<std::vector<cv::Point2f>>& corners, std::vector<int>& ids, cv::Mat& K, cv::Mat& dist, cv::Ptr<cv::aruco::Dictionary> arucoDict, cv::Mat& img) {
    double gap = 0.005;
    double markerLength = 0.052;
    cv::Mat rvec, tvec;
    cv::Ptr<cv::aruco::GridBoard> board = cv::aruco::GridBoard::create(5, 7, markerLength, gap, arucoDict);
    
    cv::aruco::estimatePoseBoard(corners, ids, board, K, dist, rvec, tvec);
    
    cv::Mat Rmat;
    cv::Rodrigues(rvec, Rmat);
    cv::Mat M;
    cv::Rodrigues(cv::Vec3d(-CV_PI, 0, 0), M);
    
    cv::Mat cam_pos = -Rmat.t() * tvec;
    cv::Mat cam_ori = Rmat.t() * M;
    
    return cam_pos.clone(), cam_ori.clone();
}


int main(int argc, char** argv)
{
    CommandLineParser parser(argc, argv);
    parser.addArgument("--type", "-t", true);
    parser.setDefaultArgument("def");
    parser.parse();

    string type = parser.get<string>("type");

    // Init AruCo
    cv::Ptr<cv::aruco::Dictionary> arucoDict = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);

    cv::FileStorage fs("/2023-05-03-23-11-22-camchain.yaml", cv::FileStorage::READ);
    cv::Mat K, dist;
    fs["cam0"]["intrinsics"] >> K;
    fs["cam0"]["distortion_coeffs"] >> dist;
    fs.release();

    // Camera pose estimation with ArUco tags
    vector<vector<cv::Point2f>> corners;
    vector<int> ids;
    vector<cv::Point2f> rejected;

    vector<string> images;
    glob("output/images/test/*.png", images);

    cout << "Estimation in progress..." << endl;

    for (const auto& image : images)
    {
        cv::Mat img = cv::imread(image);
        cv::Mat gray;
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

        // Detect ArUco markers
        cv::aruco::detectMarkers(gray, arucoDict, corners, ids);

        // Visualize detected tags and origin
        cv::aruco::drawDetectedMarkers(img, corners, ids);
        cv::aruco::drawAxis(img, K, dist, cv::noArray(), cv::noArray(), 0.1f);

        cv::imshow("res", img);

        cv::waitKey(1);

        vector<cv::Vec3d> rvecs, tvecs;
        if (type == "def")
        {
            cv::aruco::estimatePoseSingleMarkers(corners, 0.1f, K, dist, rvecs, tvecs);
        }
        else if (type == "board")
        {
            cv::Mat board = cv::Mat::zeros(cv::Size(1, 1), CV_8UC1); // Placeholder for the board configuration
            pos , ori = camera_pose_board(corners, ids, K, dist, arucoDict)
        }

        json ret;
        for (size_t i = 0; i < rvecs.size(); ++i)
        {
            json pose;
            pose["translation"] = { tvecs[i][0], tvecs[i][1], tvecs[i][2] };
            pose["rotation"] = { rvecs[i][0], rvecs[i][1], rvecs[i][2] };
            ret["poses"].push_back(pose);
        }

        cv::destroyAllWindows();
    }

    cout << "- finished -" << endl;

    ofstream outputFile("pos.json");
    outputFile << ret.dump(4);
    outputFile.close();

    return 0;
}
