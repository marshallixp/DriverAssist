//
//  DetectFace.h
//  FaceDetector1.1
//
//  Created by Xuanpeng Li on 05/13.
//  Copyright (c) 2012 ESIEE-AMIENS. All rights reserved.
//

#ifndef FaceDetector_DetectFace_h
#define FaceDetector_DetectFace_h

#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <ctime>
#include <string>

namespace FaceDetector{    
    void FindEyePts(const cv::Mat &src, 
                    int &top, int &bottom,
                    int &left, int &right, int &flag, const int &idx);

	bool SortRectYBigger(const cv::Rect &value1, const cv::Rect &value2);
    
	void DetectFace(cv::Mat &faceMat,
                    cv::CascadeClassifier &faceCascade,
                    cv::CascadeClassifier &eyesCascade,
                    cv::Rect &faceRoiRect, 
                    std::vector<double> &eyesHeightVec, 
                    std::vector<cv::Point> &eyesPointVec, 
                    int &trackKalman, const int idx);
    
    void DetectEyes(cv::Mat &faceMat, 
                    cv::CascadeClassifier &eyesCascade, 
                    const cv::Rect &faceRoiRect, 
                    std::vector<double> &eyesHeightVec,
                    std::vector<cv::Point> &eyesPointVec,
                    const int idx, std::vector<cv::Rect> &eyesRoiRect);
	
}
#endif
