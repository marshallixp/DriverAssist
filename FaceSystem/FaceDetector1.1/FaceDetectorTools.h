//
//  FaceDetectorTools.h
//  FaceDetector1.1
//
//  Created by Xuanpeng Li on 09/13.
//  Copyright (c) 2012 ESIEE-AMIENS. All rights reserved.
//

#ifndef FaceDetector_FaceDetectorTools_h
#define FaceDetector_FaceDetectorTools_h

#include <opencv2/opencv.hpp>

namespace FaceDetector{   
    void DrawHistC3(const cv::Mat &mat, const int *histSize, const float *ranges[], cv::Mat &histImg, cv::MatND &backProject);
    
    void PrintMat(const cv::Mat &mat);
    
    void imShowSub(const std::string &winname, const cv::Mat &mat, 
                   const int &Cols, const int &Rows, const int &winPos);
}
#endif
