//
//  Process.h
//  FaceDetectorSim1.1
//
//  Created by LI XUANPENG on 09/13.
//  Copyright (c) 2013 ESIEE-Amiens. All rights reserved.
//

#ifndef FaceDetectorSim_Process_h
#define FaceDetectorSim_Process_h

#include "../FaceDetector/DetectFace.h"
#include "../FaceDetector/TrackFace.h"
#include "../FaceDetector/GenerateFaceIndicators.h"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <vector>
#include <iterator>
#include <opencv2/opencv.hpp>

//!IPC
#include <sys/types.h>
#include <sys/msg.h>
#include <sys/ipc.h>
#include <errno.h>

namespace FaceDetectorSim{
    const int NUM_FACE = 9;
    const std::string faceFeatureName[NUM_FACE] =
    {
        "Frame", 
        "Height", "Height_b",
        "PERCLOS", 
        "MTime", "MICROSLEEP", 
        "BLINK", "BLINK_b",
        "Time"
    };
    
    void ProcessFaceImage(cv::Mat &faceMat, 
                          cv::CascadeClassifier &faceCascade,
                          cv::CascadeClassifier &eyesCascade,
                          cv::Rect &faceRoiRect,
                          std::vector<double> &eyesHeightVec, 
                          std::vector<cv::Point> &eyesPointVec, 
                          const double &startTime,
                          const int &idx, int &trackKalman, 
                          cv::KalmanFilter &faceKalmanFilter, 
                          cv::Mat &faceKalmanMeasureMat,
                          int &faceKalmanIndex, double &execTime,
                          cv::Mat &hist_camshift, cv::MatND &faceHist, 
                          std::vector<FaceDetector::PARTICLE_FACE> &particles);
    
    /* Acquire the sampling time, simulating the real application*/
    void GetSamplingTime(const char *fileName, std::vector<float> &samplingTime);


    void InitRecordData(std::ofstream &file, const char* fileName, const std::string *strName, const int &elemNum);

    
    void RecordFaceFeatures(std::ofstream &file, 
                            const FaceDetector::FaceFeature &faceFeatures,
                            const double &pastTime);
    
    
    void CodeMsg( const FaceDetector::FaceFeature &faceFeatures, char *str);
}
#endif
