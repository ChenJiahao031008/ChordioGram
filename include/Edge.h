#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include <cassert>
#include <cstdlib>
#include <cmath>
#include <map>
#include <chrono>

struct Object
{
    int id;
    cv::Point2i corner;
    int w, h;
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

typedef std::vector<ChordFeature> ChordioGram;

int GetAngle8Bin(const float &theta);

void LoadObjectFile(const std::string &strPath, std::vector<Object> &vObject);

void ImageProcessing(cv::Mat &objectRGB, cv::Mat &objectCanny, cv::Mat &magnitudeImage, cv::Mat &angleImage);

void EdgeProcessing(cv::Mat &objectRGB, cv::Mat &objectCanny, cv::Mat &magnitudeImage, cv::Mat &angleImage, ChordioGram &chordGram, ChordFeature &totalCF);

float ArrayNormL1(float a[], int N);

float CompareScores(ChordFeature &a, ChordFeature &b);

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




