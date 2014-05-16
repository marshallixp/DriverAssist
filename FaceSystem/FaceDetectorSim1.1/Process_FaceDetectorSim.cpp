  
//  Process.cpp
//  FaceDetectorSim1.1
//
//  Created by LI XUANPENG on 09/13.
//  Copyright (c) 2013 ESIEE-Amiens. All rights reserved.
//
#include "Process_FaceDetectorSim.h"

extern const int    WIDTH;
extern const int    HEIGHT;
extern const double COEF;

extern const int    FRAME_START;

extern const int    WIN_COLS;
extern const int    WIN_ROWS;



namespace FaceDetectorSim{    
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
                          std::vector<FaceDetector::PARTICLE_FACE> &particles)
    {
        cv::Mat faceCopy = faceMat.clone();
        /// Clear the record of eyes height for next sampling
        eyesHeightVec.clear();
        
        //! Reduce the size of raw image
        cv::resize(faceMat, faceMat, cv::Size(cvRound(WIDTH * COEF), cvRound(HEIGHT * COEF)), cv::INTER_AREA);
        
        if(trackKalman) {
            //printf("Kalman face x:%d, y:%d, width:%d, height:%d\n", faceRoiRect.x, faceRoiRect.y, faceRoiRect.width, faceRoiRect.height);
            FaceDetector::TrackFace_KF(faceMat, faceRoiRect, eyesCascade, eyesHeightVec, eyesPointVec, faceKalmanFilter, faceKalmanMeasureMat, faceKalmanIndex, trackKalman, idx);
//            FaceDetector::CamShiftTrackFace(faceMat, faceRoiRect, eyesCascade, eyesHeightVec, eyesPointVec, trackKalman, hist_camshift, idx);
//            FaceDetector::TrackFace_Particle(faceMat, trackKalman, faceHist, particles, idx);
        }
        else {
            FaceDetector::DetectFace(faceMat, faceCascade, eyesCascade, faceRoiRect, eyesHeightVec, eyesPointVec, trackKalman, idx);
//            FaceDetector::InitFaceParticleFilter(faceCopy, faceRoiRect, faceHist, particles);
        }
        
        
        execTime = ((double)cv::getTickCount() - startTime)/cv::getTickFrequency();

        /* Draw Face information */
        /* Show index of frames */
        char *text = new char[30];
        sprintf(text, "Frame: %d", idx);
        cv::putText(faceMat, text, cv::Point(0, 20), cv::FONT_HERSHEY_SIMPLEX, 0.8, CV_RGB(0,255,0));
        /* Show the process time */
        sprintf(text, "D&T time: %.2f Hz", 1.0/execTime);
        cv::putText(faceMat, text, cv::Point(0, 40), cv::FONT_HERSHEY_SIMPLEX, 0.8, CV_RGB(0, 255, 0));
        delete text;

    }

    /* Acquire the sampling time, simulating the real application*/
    void GetSamplingTime(const char *fileName, std::vector<float> &samplingTime)
    {
        // std::cout << "get sampling time" << fileName << std::endl;
        std::ifstream file;
        file.open(fileName);
        std::string text;
        int flag = 0; // odd: frame, even: pastTime
        while (file >> text)
        {
            // std::cout << text << std::endl;
            if(isdigit(text[0])){
                // std::cout << "digit" << std::endl;
                flag++;
                if(flag % 2 == 0 ){
                    samplingTime.push_back(atof(text.c_str())); // string to float
                }
            }
        }   

        // for(size_t i = 0; i < samplingTime.size(); i++)
        // {
        //     std::cout << samplingTime.at(i) << std::endl;
        // }

        file.close();
    }


    void InitRecordData(std::ofstream &file, const char* fileName, const std::string *strName, const int &elemNum)
    {
        file.open(fileName);
        
        for(int i = 0; i < elemNum; i++ )
        {
            file <<  std::setiosflags(std::ios::fixed) << std::setw(15)  << strName[i];
        }
        file << std::endl;
    }
    
    void RecordFaceFeatures(std::ofstream &file, 
                            const FaceDetector::FaceFeature &faceFeatures,
                            const double &pastTime)
    {
        file << std::setiosflags(std::ios::fixed) << std::setprecision(0) << std::setw(15) << faceFeatures.frame;
        file << std::setiosflags(std::ios::fixed) << std::setprecision(0) << std::setw(15) << faceFeatures.Height;
        file << std::setiosflags(std::ios::fixed) << std::setprecision(0) << std::setw(15) << faceFeatures.Height_Baseline;
        file << std::setiosflags(std::ios::fixed) << std::setprecision(5) << std::setw(15) << faceFeatures.PERCLOS;
        file << std::setiosflags(std::ios::fixed) << std::setprecision(5) << std::setw(15) << faceFeatures.MICROSLEEPTime;
        file << std::setiosflags(std::ios::fixed) << std::setprecision(5) << std::setw(15) << faceFeatures.MICROSLEEP;
        file << std::setiosflags(std::ios::fixed) << std::setprecision(5) << std::setw(15) << faceFeatures.BLINK;
        file << std::setiosflags(std::ios::fixed) << std::setprecision(5) << std::setw(15) << faceFeatures.BLINK_Baseline;
        file << std::setiosflags(std::ios::fixed) << std::setprecision(3) << std::setw(15) << pastTime;
        file << std::endl;
    }
    
    void CodeMsg(const FaceDetector::FaceFeature &faceFeatures, char *str)
    {
        char *temp = new char[50];
        sprintf(temp, "%d", 1);//Face Marker
        strcat(str, temp);
        sprintf(temp, ", %d", faceFeatures.frame);
        strcat(str, temp);
        sprintf(temp, ", %f", faceFeatures.Height);
        strcat(str, temp);
        sprintf(temp, ", %f", faceFeatures.lastHeight);
        strcat(str, temp);
        sprintf(temp, ", %f", faceFeatures.Height_Baseline);
        strcat(str, temp);
        sprintf(temp, ", %f", faceFeatures.PERCLOS);
        strcat(str, temp);
        sprintf(temp, ", %f", faceFeatures.MICROSLEEPTime);
        strcat(str, temp);
        sprintf(temp, ", %f", faceFeatures.MICROSLEEP);
        strcat(str, temp);
        sprintf(temp, ", %f", faceFeatures.BLINK);
        strcat(str, temp);
        sprintf(temp, ", %f", faceFeatures.BLINK_Baseline);
        strcat(str, temp);
        delete temp;
    }
       
}//FaceDetectorSim 
