#include"WalkerDetection.hpp"

#define DEBUG 1

WalkerDetection::WalkerDetection()
{

}

WalkerDetection::~WalkerDetection()
{

}

void WalkerDetection::Init(Mat src)
{
    HOGDescriptor hog;
    hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());//得到检测器  
    src.copyTo(walkerImg);   //拷贝，防止破坏原始数据

    /********************************************初始化行人框*********************************/
    vector<Rect> found, found_filtered;
    double t = (double)getTickCount();
    // run the detector with default parameters. to get a higher hit-rate  
    // (and more false alarms, respectively), decrease the hitThreshold and  
    // groupThreshold (set groupThreshold to 0 to turn off the grouping completely).  
    hog.detectMultiScale(walkerImg, found, 0, Size(8, 8), Size(32, 32), 1.05, 2);
    t = (double)getTickCount() - t;
    printf("tdetection time = %gms\n", t*1000. / cv::getTickFrequency());
    size_t i, j;
    for (i = 0; i < found.size(); i++)
    {
        Rect r = found[i];
        //筛选出框住同一行人的box中较大的
        for (j = 0; j < found.size(); j++)
            if (j != i && (r & found[j]) == r)
                break;
        if (j == found.size())
            found_filtered.push_back(r);
    }
    walkerRect = found_filtered[0];
    for (i = 0; i < found_filtered.size(); i++)
    {
        Rect r = found_filtered[i];
        rectangle(walkerImg, r.tl(), r.br(), cv::Scalar(0, 255, 0), 3);
    }
    // for (i = 0; i < found_filtered.size(); i++)
    // {
    //     Rect r = found_filtered[i];
    //     // the HOG detector returns slightly larger rectangles than the real objects.  
    //     // so we slightly shrink the rectangles to get a nicer output.  
    //     r.x += cvRound(r.width*0.1);
    //     r.width = cvRound(r.width*0.8);
    //     r.y += cvRound(r.height*0.07);
    //     r.height = cvRound(r.height*0.8);
    //     rectangle(walkerImg, r.tl(), r.br(), cv::Scalar(0, 255, 0), 3);
    //     // 初始化时直接将矩形序列中的第一个作为检测到的行人box
    //     if(i==0) walkerRect = r;
    // }

    /********************************************初始化追踪行人的特征向量*********************************/
    Mat walker = src(walkerRect); //  剪切出行人图像
    features = getFeatures(walker);
   
    #ifdef DEBUG
        namedWindow("people detector", 1);
        imshow("people detector", walkerImg);
        waitKey(0);
    #endif
        
}

void WalkerDetection::Detect(Mat src)
{
    HOGDescriptor hog;
    hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());//得到检测器  
    src.copyTo(walkerImg);   //拷贝，防止破坏原始数据

    /*******************************************根据上次的检测结果在可能区域内检测行人*********************************/
    Rect possibleRect;
    float scale = 10; //决定搜索范围的参数，需要调整
    float center_x = walkerRect.x + walkerRect.width/2.0;
    float center_y = walkerRect.y + walkerRect.height/2.0;
    possibleRect.x = (center_x - walkerRect.width)>0?(walkerRect.x - walkerRect.width):0;
    possibleRect.y = (walkerRect.y - walkerRect.height)>0?(walkerRect.y - walkerRect.height):0;
    possibleRect.width = (scale*walkerRect.width+possibleRect.x)<walkerImg.cols?scale*walkerRect.width:walkerRect.width;
    possibleRect.height = (scale*walkerRect.height+possibleRect.y)<walkerImg.rows?scale*walkerRect.height:walkerRect.height;
    Mat possibleArea = walkerImg(possibleRect);
    //Mat possibleArea; walkerImg.copyTo(possibleArea);
    vector<Rect> found, found_filtered;
    double t = (double)getTickCount();
    // run the detector with default parameters. to get a higher hit-rate  
    // (and more false alarms, respectively), decrease the hitThreshold and  
    // groupThreshold (set groupThreshold to 0 to turn off the grouping completely).  
    hog.detectMultiScale(possibleArea, found, 0, Size(8, 8), Size(32, 32), 1.05, 2);
    t = (double)getTickCount() - t;
    printf("tdetection time = %gms\n", t*1000. / cv::getTickFrequency());
    size_t i, j;
    for (i = 0; i < found.size(); i++)
    {
        Rect r = found[i];
        //筛选出框住同一行人的box中较大的
        for (j = 0; j < found.size(); j++)
            if (j != i && (r & found[j]) == r)
                break;
        if (j == found.size())
            found_filtered.push_back(r);
    }
    /***************************计算各个矩形框特征向量距离上个特征向量的距离，确定行人所在矩形框**************************/
    float minDist = 9999999;
    int minIndex = 0;
    vector<vector<float> > featuresTab;
    for (i = 0; i < found_filtered.size(); i++)
    {
        Rect r = found_filtered[i];
        Mat tempImg = possibleArea(r);
        vector<float> tempFeatures = getFeatures(tempImg);
        float dist = 0;
        for( j = 0; j < tempFeatures.size();j++)
        {
            dist += (features[i]-tempFeatures[i])*(features[i]-tempFeatures[i]);    
        }
        if(dist<minDist)
        {
            minDist = dist;
            minIndex = i;
        }
        featuresTab.push_back(tempFeatures);
        vector<float>().swap(tempFeatures); // 释放内存空间
    }
    features.assign(featuresTab[minIndex].begin(),featuresTab[minIndex].end());// 更新特征向量
    walkerRect = found_filtered[minIndex];// 更新行人box
    rectangle(possibleArea, walkerRect.tl(), walkerRect.br(), cv::Scalar(0, 255, 0), 3);

    #ifdef DEBUG
        namedWindow("people detector", 1);
        imshow("people detector", possibleArea);
        waitKey(0);
    #endif
}

vector<float> WalkerDetection::getFeatures(Mat src)
{
    //提取HOG特征
    //winSize = 48*64, blockSize = 16*16, cellSize = 8*8, nBins = 8
    //共有 5*7*2*2*8 = 1120 个特征
    HOGDescriptor hog_walker(Size(48,64),Size(16,16),Size(8,8),Size(8,8),8);
    Mat tempImg;
    src.copyTo(tempImg);
    resize(tempImg,tempImg,Size(48,64));
    vector<float> descriptors; // HOG描述子向量
    hog_walker.compute(tempImg,descriptors); // 计算HOG特征向量

    //提取颜色直方图特征
    int histSize[3];    //直方图每一维项的数量
    float hranges[2];     //H通道像素的最小和最大值
    float sranges[2];     //S通道像素的最小和最大值
    float vranges[2];     //V通道像素的最小和最大值
    const float *ranges[3];    //各通道的范围
    int channels[3];        //参与计算的通道,分别为0,1,2
    int hBins = 10;         //H通道项数
    int sBins = 10;         //S通道项数
    int vBins = 10;         //V通道项数
    histSize[0] = hBins;
    histSize[1] = sBins;
    histSize[2] = vBins;
    hranges[0] = 0;hranges[1] = 180;
    sranges[0]=0; sranges[1]=256;
    vranges[0]=0; vranges[1]=256;
    ranges[0]=hranges;
    ranges[1]=sranges;
    ranges[2]=vranges;
    channels[0]=0;
    channels[1]=1;
    channels[2]=2;
    int dims=3;

    //计算颜色直方图特征
    Mat hist;
    Mat HSVImage;
    cvtColor(tempImg, HSVImage, COLOR_BGR2HSV);
    calcHist(&HSVImage, 1, channels, Mat(), hist, dims, histSize, ranges, true, false);

    //将两种特征拼接在一起形成特征向量features
    vector<float> my_features;
    my_features.clear();// 首先清空
    //拷贝HOG特征向量(归一化)
    float sumHog = 0;
    for(int i = 0; i < descriptors.size();i++)
    {
        sumHog += descriptors[i];
    }
    //if(_finite(sumHog)==0) cout<<"sumHOG overflow";
    for(int i = 0; i < descriptors.size();i++)
    {
        my_features.push_back(descriptors[i]/sumHog);
    }
    //将颜色直方图特征拼接到末尾
    float sumHist = 0;
    for(int h = 0; h < hBins; h++)
        for(int s = 0; s < sBins; s++)
            for(int v = 0; v < vBins; v++)
                sumHist += hist.at<float>(h,s,v);
    //if(_finite(sumHist)==0) cout<<"sumHist overflow";
    for(int h = 0; h < hBins; h++)
        for(int s = 0; s < sBins; s++)
            for(int v = 0; v < vBins; v++)
                my_features.push_back(hist.at<float>(h,s,v)/sumHist);
    
    return my_features;
}


