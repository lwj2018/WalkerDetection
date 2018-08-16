#ifndef _WALKER_DECTECTION_
#define _WALKER_DECTECTION_

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>     
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <string>   
#include <iostream>   
#include <algorithm>   
#include <iterator>  

#include <stdio.h>  
#include <string.h>  
#include <ctype.h>  

using namespace cv;
using namespace std;

class WalkerDetection
{
    public:
        Rect walkerRect;
        vector<float> features;
        Mat walkerImg;
    
    public:
        WalkerDetection();
        ~WalkerDetection();
        void Init(Mat src);
        void Detect(Mat src);
    protected:
        vector<float> getFeatures(Mat src);
};

#endif