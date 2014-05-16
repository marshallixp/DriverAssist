//
//  TrackFace.h
//  FaceDetector1.1
//
//  Created by Xuanpeng Li on 05/13.
//  Copyright (c) 2012 ESIEE-AMIENS. All rights reserved.
//

#ifndef FaceDetector_TrackFace_h
#define FaceDetector_TrackFace_h

#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <iomanip>

#include "DetectFace.h"
#include "FaceDetectorTools.h"

namespace FaceDetector{
    void CamShiftTrackFace(cv::Mat &faceMat, cv::Rect &faceRoiRect,
                           cv::CascadeClassifier &eyesCascade, 
                           std::vector<double> &eyesHeightVec,
                           std::vector<cv::Point> &eyesPointVec, int &trackKalman, 
                           cv::Mat &hist_camshift, const int idx);
    
    void CamShiftTrackFace2(cv::Mat &faceMat, cv::Rect &faceRoiRect,
                            cv::CascadeClassifier &eyesCascade, 
                            std::vector<double> &eyesHeightVec, 
                            std::vector<cv::Point> &eyesPointVec, int &trackKalman, 
                            cv::Mat &hist_camshift, const int idx);
    
    /// Face Kalman Filter
    void InitFaceKalmanFilter(cv::KalmanFilter &faceKalmanFilter, cv::Mat &faceKalmanMeasureMat);
    
    void TrackFace_KF(cv::Mat &faceMat, cv::Rect &faceRoiRect,
                      cv::CascadeClassifier &eyesCascade, 
                      std::vector<double> &eyesHeightVec, 
                      std::vector<cv::Point> &eyesPointVec, 
                      cv::KalmanFilter &faceKalmanFilter,cv::Mat &faceKalmanMeasureMat,
                         int &faceKalmanIndex, int &trackKalman, const int idx);
    
    void CatchFace(cv::Mat &faceMat, 
                   cv::KalmanFilter &faceKalmanFilter, cv::Mat &faceKalmanMeasureMat,
                   cv::Rect &faceRoiRect,int &faceKalmanIndex, 
                   const int &flagEyeUpdated, std::vector<cv::Point> &eyesPredictPointVec);
    
    
    #define KALMAN_PREDICT  0
    #define LAST_INSTEAD    1
    void TrackEyes(cv::Mat &faceMat, const cv::Rect &faceRoiRect,
                   std::vector<double> &eyesHeightVec, 
                   const std::vector<cv::Point> &eyesTrackPointVec,
                   std::vector<cv::Point> &eyesPointVec, 
                   int &trackKalman, const int idx,
                   std::vector<cv::Rect> &eyesRoiRect);
    
    
    /* Particle Filter Tracking */
    // Struct of particle
    typedef struct particle {
        cv::Point2d p;          //current window position
        double scale;       //scaling of window
        cv::Point2d pPre;       //predicted window position
        double scalePre;    //predicted scaling of window
        cv::Point2d pOri;       //Original window position
        cv::Rect rect;          //Original region size
        cv::MatND hist;
        double weight;      //weight of particle
    } PARTICLE_FACE;
    
    bool particle_cmp(const PARTICLE_FACE &p1, const PARTICLE_FACE &p2);
    
    void InitFaceParticleFilter(cv::Mat &faceMat, const cv::Rect &faceRoiRect, cv::MatND &faceHist, std::vector<PARTICLE_FACE> &particles);
    
    void TrackFace_Particle(cv::Mat &faceMat, int &trackParticle, const cv::MatND &faceHist, std::vector<PARTICLE_FACE> &particles, const int &idx);
}

#endif
