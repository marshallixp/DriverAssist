//
//  Process.h
//  FaceRecorder1.1
//
//  Created by LI XUANPENG on 10/13.
//  Copyright (c) 2012 ESIEE-Amiens. All rights reserved.
//

#ifndef FaceRecorder_Process_h
#define FaceRecorder_Process_h

#include "../FaceDetector/DetectFace.h"
#include "../FaceDetector/TrackFace.h"
#include "../FaceDetector/GenerateFaceIndicators.h"

#include <string>
#include <vector>
#include <iterator>
#include <iostream>
#include <fstream>
#include <cassert>
#include <iomanip>
#include <dirent.h>
#include <opencv2/opencv.hpp>

//!IPC
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/msg.h>
#include <sys/ipc.h>
#include <errno.h>

namespace FaceRecorder{
    const int NUM_FACE_FEATURES  = 10;
    const std::string faceFeatureName[NUM_FACE_FEATURES] = 
    {
        "Frame", 
        "Height", "Height_b",
        "PERCLOS", 
        "MTime", "MICROSLEEP", 
        "BLINK", "BLINK_b",
        "execTime", "pastTime"
    };
    
    void ProcessFaceImage(cv::Mat &faceMat, 
                          cv::CascadeClassifier &faceCascade,
                          cv::CascadeClassifier &eyesCascade,
                          cv::Rect &faceRoiRect,
                          std::vector<double> &eyesHeightVec, 
                          std::vector<cv::Point> &eyesPointVec, 
                          const int &idx, int &trackKalman, 
                          cv::KalmanFilter &faceKalmanFilter, 
                          cv::Mat &faceKalmanMeasureMat,
                          int &faceKalmanIndex, double &execTime);
    
    void InitRecordData(std::ofstream &file, const char* fileName, const std::string *strName, const int &elemNum);
    
    void RecordFaceFeatures(std::ofstream &file, const FaceDetector::FaceFeature &faceFeatures, const double &execTime, const double &pastTime);
    
    void CodeMsg( const FaceDetector::FaceFeature &faceFeatures, char *str);
}
#endif
