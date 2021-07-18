#include "Edge.h"

using namespace std;
using namespace cv;

// 过大的网格会使得边缘点的分布不均匀，过小的网格则会使噪点较多
#define CELL_NUM 2
#define ANGLE_OVERLAP 5

int main(int argc, char const *argv[])
{
    if ( argc != 5){
        cerr << "[ERROR] Please check argv! " << endl;
        return -1;
    }

    // 获得深度图和彩色图
    cv::Mat objectRGB, objectRGB2;
    {
        cv::Mat imgRGB = imread(argv[1], CV_LOAD_IMAGE_UNCHANGED);
        cv::Mat imgRGB2 = imread(argv[2], CV_LOAD_IMAGE_UNCHANGED);

        vector<Object> vObject, vObject2;
        LoadObjectFile(argv[3], vObject);
        LoadObjectFile(argv[4], vObject2);
        Object obj  = vObject[0];
        Object obj2 = vObject2[0];
        obj.print();
        obj2.print();

        cv::Mat rgbROI = imgRGB(cv::Rect(obj.corner.x, obj.corner.y, obj.w, obj.h));
        cv::Mat rgbROI2 = imgRGB2(cv::Rect(obj2.corner.x, obj2.corner.y, obj2.w, obj2.h));

        // cv::Mat objectRGB_ = Mat::zeros(rgbROI.size(), rgbROI.type());
        // cv::Mat objectRGB2_ = Mat::zeros(rgbROI2.size(), rgbROI2.type());
        cv::Mat objectRGB_, objectRGB2_;

        // 彩色图像采用双边滤波去噪
        bilateralFilter(rgbROI, objectRGB, 15, 20, 50);
        bilateralFilter(rgbROI2, objectRGB2, 15, 20, 50);
    }


    // 获得边缘信息，包括canny边缘，边缘强度和边缘角度
    cv::Mat magnitudeImage, angleImage, objectCanny;
    cv::Mat magnitudeImage2, angleImage2, objectCanny2;
    ImageProcessing(objectRGB, objectCanny, magnitudeImage, angleImage);
    ImageProcessing(objectRGB2, objectCanny2, magnitudeImage2, angleImage2);

    ChordFeature totalCF;
    ChordFeature totalCF2;
    ChordioGram chordGram, chordGram2;

    EdgeProcessing(objectRGB, objectCanny, magnitudeImage, angleImage, chordGram, totalCF);
    EdgeProcessing(objectRGB2, objectCanny2, magnitudeImage2, angleImage2, chordGram2, totalCF2);

    cv::waitKey(0);

    cout << "==============SUMMARY==============" << endl;
    float score = CompareScores(totalCF, totalCF2);
    cout << "[INFO] 非权重分配得分 : " << score << endl;

    float score_weight = 0.0;
    std::vector<float> scores;
    for (size_t i=0; i<chordGram.size();++i ){
        float tmp_score = CompareScores(chordGram[i], chordGram2[i]);
        // cout << "tmp_score : " <<  tmp_score << endl;
        scores.push_back(tmp_score);
    }
    sort(scores.begin(), scores.end());
    for (size_t i = 0; i < floor(chordGram.size() * 0.5); ++i)
    {
        score_weight += scores[i];
    }
    cout << "[INFO] 回环模式权重分配得分 : " << score_weight << endl;

    float score_tracking =0.0;
    float min_tracking = 1000.0;
    std::vector<float> min_scores;
    for (size_t i = 0; i < chordGram.size() ; ++i)
        for (size_t j = 0; j < chordGram2.size(); ++j)
        {
            float tmp_score = CompareScores(chordGram[i], chordGram2[j]);
            if (tmp_score <min_tracking)
                min_tracking = tmp_score;
        }
    min_scores.push_back(min_tracking);
    sort(min_scores.begin(),min_scores.end());
    for (size_t i = 0; i < floor(chordGram.size() * 0.7); ++i)
    {
        score_tracking += scores[i];
    }
    score_tracking /= floor(chordGram.size() * 0.7);
    cout << "[INFO] 追踪模式权重分配得分 : " << score_tracking << endl;

    return 0;
}

float CompareScores(ChordFeature &a, ChordFeature &b)
{
    float score = -1;
    a.ArrayFlatten();
    b.ArrayFlatten();
    float p[a.N];
    for (size_t i=0; i<a.N; ++i){
        p[i] = a.flattenChord[i] - b.flattenChord[i];
    }
    score = ArrayNormL1(p, a.N);
    // cout << score << "\t";
    return score;

}

float ArrayNormL1(float a[], int N){
    float norm = 0;
    for (size_t i=0; i<N; ++i)
        norm += fabs(a[i]);
    return norm;
}

void ImageProcessing(cv::Mat &objectRGB, cv::Mat &objectCanny, cv::Mat &magnitudeImage, cv::Mat &angleImage)
{
    // TODO:尝试用彩色图像或者经过处理后的灰度图像进行边缘提取看看有什么不同
    cv::Mat objectGray, objectGray_;
    cvtColor(objectRGB, objectGray_, COLOR_BGR2GRAY);
    // 直方图均值化
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
    clahe->apply(objectGray_, objectGray);

    Mat grad_x, grad_y;
    Sobel(objectGray, grad_x, CV_32FC1, 1, 0); //求梯度
    Sobel(objectGray, grad_y, CV_32FC1, 0, 1);
    blur(grad_x, grad_x, Size(3, 3));
    blur(grad_y, grad_y, Size(3, 3));
    int edgeThreshScharr_low = 50 ;
    int edgeThreshScharr_high = 80;

    Mat grad16S_x, grad16S_y;
    grad_x.convertTo(grad16S_x, CV_16S);
    grad_y.convertTo(grad16S_y, CV_16S);
    Canny(grad16S_x, grad16S_y, objectCanny, edgeThreshScharr_low, edgeThreshScharr_high, true);
    imshow("objectCanny: " + (objectCanny.cols), objectCanny);
    CutImageEdge(objectCanny);
    imshow("objectCanny after processed: " + (objectCanny.cols), objectCanny);

    // 如果直接加权会导致负数部分比较难处理
    // // addWeighted(grad_x, 0.5, grad_y, 0.5, 0, magnitudeImage);

    // 默认为弧度，即[0,2π）；如果angleInDegrees==true,则为则角度数[0,360
    // phase(grad_x, grad_y, angleImage, true);
    cv::cartToPolar(grad_x, grad_y, magnitudeImage, angleImage, true);

    // check points
    cv::Mat abs_grad_x, abs_grad_y, showGradImage;
    convertScaleAbs(grad_x, abs_grad_x);
    convertScaleAbs(grad_y, abs_grad_y);
    addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, showGradImage);
    // imshow("showGradImage for abs_grad", showGradImage);
    // cv::waitKey(0);
    // 如果采用这种方法，则因为梯度比例已经被改变了，所以在后面需要加上梯度选择，大于4-5的才进行处理
    // 这种方法得到的点相对稠密一点，先选择这个试试
    // showGradImage.convertTo(magnitudeImage, CV_32FC1);

    Mat outGrad, outAngle;
    magnitudeImage.convertTo(outGrad, CV_16UC1);
    angleImage.convertTo(outAngle, CV_16UC1);
    cv::imwrite("../result/magnitudeImage.png", outGrad);
    cv::imwrite("../result/angle.png", outAngle);
    cv::imwrite("../result/objectCanny.png", objectCanny);

    // cout << "[INFO] magnitudeImage 的类型: " << magnitudeImage.type() << endl;
    // cout << "[INFO] angle 的类型: " << angleImage.type() << endl;
}

void EdgeProcessing(cv::Mat &objectRGB, cv::Mat &objectCanny, cv::Mat &magnitudeImage, cv::Mat &angleImage, ChordioGram &chordGram, ChordFeature &totalCF)
{
    // 划分网格，对每个网格取点
    vector<vector<EdgePoint>> vedgePoints;
    int nCell = CELL_NUM * CELL_NUM; // 所有cell的总数
    vedgePoints.resize(nCell);
    int cell_y_size = floor(objectRGB.rows / CELL_NUM);
    int cell_x_size = floor(objectRGB.cols / CELL_NUM);
    cout << "[INFO] 网格大小: [" << cell_x_size << ", " << cell_y_size << "]" << endl;
    cout << "       图像大小: [" << objectRGB.cols << ", " << objectRGB.rows << "]" << endl;

    int totalVotes = 0;
    double p = 0.2;
    cout << "[INFO] 边缘点的比例系数 p: " << p << endl;

    for (size_t i = 0; i < nCell; ++i)
    {
        vector<EdgePoint> vedge;
        Point2i area_start((i % CELL_NUM) * cell_x_size, (i / CELL_NUM) * cell_y_size);
        // cout << "area_start : " << i << ": " << area_start.x << "," << area_start.y << endl;
        // 计算每个网格左上方角点坐标 area_start
        for (size_t y = area_start.y; y < area_start.y + cell_y_size; ++y)
        {
            for (size_t x = area_start.x; x < area_start.x + cell_x_size; ++x)
            {
                if (objectCanny.at<char>(y, x) == 0) continue;
                float theta = angleImage.at<float>(y, x);
                float grad = magnitudeImage.at<float>(Point(x, y));
                if (grad < 50)
                    continue;
                // TODO：非极大值抑制方法处理，现有方法效率太低
                // if (magnitudeImage.at<float>(y, x + 1) > grad || magnitudeImage.at<float>(y + 1, x + 1) > grad || magnitudeImage.at<float>(y + 1, x) > grad || magnitudeImage.at<float>(y, x - 1) > grad || magnitudeImage.at<float>(y - 1, x - 1) > grad || magnitudeImage.at<float>(y - 1, x) > grad)
                //     continue;
                vedge.emplace_back(Point2i(x, y), i, grad, theta);
            }
        }
        // 根据梯度大小进行排序，比例为p
        sort(vedge.begin(), vedge.end(), CompGreater());
        for (size_t j = 0; j < floor(p * vedge.size()); ++j)
            vedgePoints[i].emplace_back(vedge[j]);

        ChordFeature cf;

        // TODO：加速加速 OpenMP上起来
        for (size_t k = 0; k < vedgePoints[i].size(); ++k)
        {
            EdgePoint ep_p = vedgePoints[i][k];
            for (size_t j = 0; j < vedgePoints.size() && j != k; ++j)
            {
                EdgePoint ep_q = vedgePoints[i][j];
                float phi = atan2(ep_q.pt.y - ep_p.pt.y, ep_q.pt.x - ep_p.pt.x) / 3.1415 * 180;
                float len = sqrt(pow(ep_q.pt.y - ep_p.pt.y, 2) + pow(ep_q.pt.x - ep_p.pt.x, 2));
                int len_bin = floor(len * 4.0 / (sqrt(pow(cell_x_size, 2) + pow(cell_y_size, 2))));

                if (ANGLE_OVERLAP == 0){
                    int angleBin = GetAngle8Bin(phi);
                    int p_bin = GetAngle8Bin(ep_p.angle - phi);
                    int q_bin = GetAngle8Bin(ep_q.angle - phi);
                    // cout << len_bin << "; " << angleBin - 1 << "; " << p_bin - 1 << "; " << q_bin - 1<< endl;
                    cf.chord[len_bin][angleBin - 1][p_bin - 1][q_bin - 1]++;
                    totalCF.chord[len_bin][angleBin - 1][p_bin - 1][q_bin - 1]++;
                    totalVotes++;
                }else{
                    int angleBin1, angleBin2, p_bin1, p_bin2, q_bin1, q_bin2;
                    GetAngle8BinOverlapping(phi, angleBin1, angleBin2);
                    GetAngle8BinOverlapping(ep_p.angle - phi, p_bin1, p_bin2);
                    GetAngle8BinOverlapping(ep_q.angle - phi, q_bin1, q_bin2);
                    cf.chord[len_bin][angleBin1 - 1][p_bin1 - 1][q_bin1 - 1]++;
                    cf.chord[len_bin][angleBin2 - 1][p_bin2 - 1][q_bin2 - 1]++;
                    totalCF.chord[len_bin][angleBin1 - 1][p_bin1 - 1][q_bin1 - 1]++;
                    totalCF.chord[len_bin][angleBin2 - 1][p_bin2 - 1][q_bin2 - 1]++;
                    totalVotes = totalVotes+2;
                }

            }
        }
        // cf.GetHistogram();
        chordGram.push_back(cf);
    }
    cout << "[INFO] 总票数为： " << totalVotes << endl;
    // totalCF.GetHistogram();
    // totalCF.ArrayFlatten();

    cv::Mat showCircles = objectRGB.clone();
    for (size_t i = 0; i < vedgePoints.size(); ++i)
    {
        for (size_t j = 0; j < vedgePoints[i].size(); ++j)
        {
            circle(showCircles, vedgePoints[i][j].pt, 1, Scalar(0, 255, 0), -1);
        }
    }
    cv::imshow("showCircles" + to_string(totalVotes), showCircles);
}

void LoadObjectFile(const string &strPath, vector<Object> &vObject)
{
    ifstream ObjectFile;
    ObjectFile.open(strPath.c_str());
    cout << "[LOAD] Open Object File ..." << endl;
    while (!ObjectFile.eof())
    {
        string s;
        getline(ObjectFile, s);
        if (!s.empty())
        {
            stringstream ss;
            ss << s;
            Object obj;
            ss >> obj.id;
            ss >> obj.corner.x;
            ss >> obj.corner.y;
            ss >> obj.w;
            ss >> obj.h;
            vObject.emplace_back(obj);
        }
    }
}

int GetAngle8Bin(const float &theta)
{
    int angleBin;
    vector<float> BinDiv = {22.5, 67.5, 112.5, 157.5, 202.5, 247.5, 292.5, 337.5};
    if (theta > BinDiv.back())
    {
        angleBin = 1;
    }
    else
    {
        for (size_t bn = 0; bn < BinDiv.size(); ++bn)
        {
            if (theta < BinDiv[bn])
            {
                angleBin = bn+1;
                break;
            }
        }
    }
    return angleBin;
}

void GetAngle8BinOverlapping(const float &theta, int &bin1, int &bin2)
{
    vector<float> BinDiv = {22.5, 67.5, 112.5, 157.5, 202.5, 247.5, 292.5, 337.5};
    bin1 = GetAngle8Bin(theta);
    bin2 = bin1;
    for (size_t bn = 0; bn < BinDiv.size(); ++bn)
    {
        if (theta < (BinDiv[bn] + ANGLE_OVERLAP) && theta > (BinDiv[bn] - ANGLE_OVERLAP))
        {
            if (bin1 == bn + 1) bin2 = bn + 2;
            if (bin1 == bn + 2) bin2 = bn + 1;
            break;
        }
    }
}

void CutImageEdge(cv::Mat &objectCanny)
{
    Mat labels = Mat::zeros(objectCanny.size(), CV_32S);
    cv::Mat stats, centroids;
    int num_labels = connectedComponentsWithStats(objectCanny, labels, stats, centroids, 8, 4);
    for (int i = 1; i < num_labels; i++)
    {
    }
    for (int y = 0; y < objectCanny.rows; y++)
        for (int x = 0; x < objectCanny.cols; x++)
        {
            int label = labels.at<int>(y, x);
            if (stats.at<int>(label, cv::CC_STAT_AREA) < 30)
                objectCanny.at<uchar>(Point(x, y)) = 0;
        }
}
