#include <iostream>
#include "WalkerDetection.hpp"
#include <fstream>

using namespace std;

int main(int argc,char ** argv)
{
    char imageFileName[50];
    WalkerDetection detector;
    
    for(int i = 0; i < 24; i++)
    {
        sprintf(imageFileName,"../input/images/image_%08d_0.png",i);
        Mat src = imread(imageFileName);
        if(!src.data) cout<<"can not read image file"<<endl;
        if (i == 0)
            detector.Init(src); 
        else
            detector.Detect(src);
    }
    
    
}