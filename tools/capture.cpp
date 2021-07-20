#include <opencv2/opencv.hpp>
#include <iostream>
#include <time.h>

int main()
{

    cv:: VideoCapture cap = cv::VideoCapture(0);
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    cv::Mat frame, img;
    while(cap.isOpened())
    {
        clock_t time = clock();
        cap.read(frame);
        if(frame.empty())
        {
           std::cout << "Read frame failed!" << std::endl;
           break;
        }
        cv::imshow("", frame);
        if(cv::waitKey(1)=='s'){
            cv::imwrite("../result/capture_image/" + std::to_string(time) + ".jpg", frame);
        }
    }
    return 0;
}

