#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <torch/script.h>
#include <algorithm>
#include <time.h>
#include <vector>
#include <cassert>
#include <cstdlib>
#include <cmath>
#include <map>
#include <chrono>

struct Object
{
    int id;
    int w, h;
    cv::Point2i corner;
    std::string name = "";
    float score = 1.0;

    Object(int &id_, int &x, int &y, int &w_, int &h_, std::string &name_, float &score_) : id(id_), corner(cv::Point2i(x,y)), w(w_), h(h_), name(name_), score(score_){};

    Object(){};

    void print() { std::cout << "[INFO] Object " << id << ":  " << corner << "\t" << w << "\t" << h << std::endl; };
};

struct ChordFeature
{
    static const int bl = 4;
    static const int b_phi = 8;
    static const int b_theta = 8;
    static const int N = bl * b_phi * b_theta * b_theta;

    float norm = 1e-5;

    int chord[bl][b_phi][b_theta][b_theta] = {{{{0},{0}}}};
    float flattenChord[N] = {0.0};

    void GetHistogram()
    {
        std::cout << "[INFO] 直方图分布：\n";
        for (size_t i = 0; i < bl; ++i)
        {
            for (size_t j = 0; j < b_phi; j++)
                for (size_t n = 0; n < b_theta; n++)
                    for (size_t m = 0; m < b_theta; m++)
                        std::cout << chord[i][j][n][m] << ";";
        }
        std::cout << std::endl;
    }

    void ArrayFlatten()
    {
        size_t p = 0;
        for (size_t i = 0; i < bl; ++i)
        {
            for (size_t j = 0; j < b_phi; j++)
                for (size_t n = 0; n < b_theta; n++)
                    for (size_t m = 0; m < b_theta; m++)
                    {
                        flattenChord[p++] = chord[i][j][n][m];
                        norm += fabs(chord[i][j][n][m]);
                    }
        }
        FlattenNorm();
    }

    void FlattenNorm()
    {
        for (size_t i = 0; i < N; ++i)
            flattenChord[i] = flattenChord[i] * 1.0 / norm;
    }
};

struct EdgePoint
{
    cv::Point2i pt;
    ushort depth;
    int areaIdx;

    float grad;
    float angle;

    EdgePoint(cv::Point2i _pt, int _areaIdx, float _grad, float _angle)
        : pt(_pt), areaIdx(_areaIdx), grad(_grad), angle(_angle){};
};

class CompGreater
{
public:
    bool operator()(EdgePoint &ep1, EdgePoint &ep2)
    {
        return abs(ep1.grad) > abs(ep2.grad);
    }
};

typedef std::vector<ChordFeature> ChordioGram;

int GetAngle8Bin(const float &theta);

void GetAngle8BinOverlapping(const float &theta, int &bin1, int &bin2);

void LoadObjectFile(const std::string &strPath, std::vector<Object> &vObject);

void ImageProcessing(cv::Mat &objectRGB, cv::Mat &objectCanny, cv::Mat &magnitudeImage, cv::Mat &angleImage);

void EdgeProcessing(cv::Mat &objectRGB, cv::Mat &objectCanny, cv::Mat &magnitudeImage, cv::Mat &angleImage, ChordioGram &chordGram, ChordFeature &totalCF);

float ArrayNormL1(float a[], int N);

float CompareScores(ChordFeature &a, ChordFeature &b);

void CutImageEdge(cv::Mat &objectCanny);

std::vector<torch::Tensor> non_max_suppression(torch::Tensor preds, float score_thresh=0.5, float iou_thresh=0.5);

void PreProcessingOffLine(const cv::Mat &imgRGB, const std::string &strPath, cv::Mat &objectRGB);

void PreProcessingOnLine(const cv::Mat &imgRGB, cv::Mat &objectRGB);
