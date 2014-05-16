//
//  TrackFace.cpp
//  FaceDetector1.1
//
//  Created by Xuanpeng Li on 05/13.
//  Copyright (c) 2012 ESIEE-AMIENS. All rights reserved.
//
//  If Eye detector can detect eyes correctly, we use Kalman filter to track face region.
//  // Param: trackKalman
//      0 eyes not detected, 
//      -1 eyes detected, 
//      1 track face and eyes detected
//      2 track eyes
//
#include "TrackFace.h"

extern const double  COEF;

extern const int  EYES_RECORD;


extern const char EYE_COLOR_NAME[];
extern const char EYE_BIN_NAME[];


extern const int  FRAME_START;

extern const int WIN_COLS;
extern const int WIN_ROWS;

namespace FaceDetector{
    /****************************************************************/
    //! CamShift Tracking based on Back Project
    //! trackKalman : -1 Detected; 1 Tracked; 0 Redetect
    //! use the Hue or Hue_Saturation to calculate the BackProject
    /****************************************************************/
    
    void CamShiftTrackFace(cv::Mat &faceMat, cv::Rect &faceRoiRect,
                           cv::CascadeClassifier &eyesCascade, 
                           std::vector<double> &eyesHeightVec, 
                           std::vector<cv::Point> &eyesPointVec, int &trackKalman, 
                           cv::Mat &hist_camshift, const int idx)
    { 
        int hsize = 20;
        float hranges[] = {0,180};
        const float* phranges = hranges;
        
        cv::Mat hsv, hue, mask, backproj, histimg = cv::Mat::zeros(200, 320, CV_8UC3);
        cv::cvtColor(faceMat, hsv, cv::COLOR_RGB2HSV);
        
        //char text[100];
        //! trackKalman: -1(eyes detected), 1(eyes detected), 2(eyes tracked)
        if (trackKalman) {
            int _vmin = 10, _vmax = 256, smin = 50;
            
            cv::inRange(hsv, cv::Scalar(0, smin, MIN(_vmin,_vmax)),
                    cv::Scalar(180, 256, MAX(_vmin, _vmax)), mask);
            
            //! Copy h-channel of hsv to hue mat
            int ch[] = {0, 0};
            hue.create(hsv.size(), hsv.depth());
            cv::mixChannels(&hsv, 1, &hue, 1, ch, 1);

            //cv::imshow("hsv", hsv);cv::waitKey(1);
            
            //! Just work when face detect finishes
            if( trackKalman < 0 ) {
                cv::Mat roi_face(hue, faceRoiRect), maskroi(mask, faceRoiRect);
                imshow("hue", roi_face);imshow("mask", maskroi);
                
//                sprintf(text, "/Users/xuanpengli/Desktop/PIC/CAM_hue_%d.jpg", idx);
//                cv::imwrite(text, roi_face);
                
                cv::calcHist(&roi_face, 1, 0, maskroi, hist_camshift, 1, &hsize, &phranges);
                cv::normalize(hist_camshift, hist_camshift, 0, 255, CV_MINMAX);
                
                for(int i = 0; i < hist_camshift.rows; i++)
                    std::cout << hist_camshift.at<float>(i,0) << std::endl;
                
                trackKalman = 1;
                
                //! Calculate Hist Image              
                histimg = cv::Scalar::all(0);
                int binW = histimg.cols / hsize;
                cv::Mat buf(1, hsize, CV_8UC3);
                for( int i = 0; i < hsize; i++ ) {
                    buf.at<cv::Vec3b>(i) = cv::Vec3b(cv::saturate_cast<uchar>(i*180./hsize), 255, 255);
                }
                cv::cvtColor(buf, buf, cv::COLOR_HSV2BGR);
                
                for( int i = 0; i < hsize; i++ ) {
                    int val = cv::saturate_cast<int>(hist_camshift.at<float>(i)*histimg.rows/255.0);
                    cv::rectangle( histimg, cv::Point(i*binW,histimg.rows), cv::Point((i+1)*binW, histimg.rows - val), cv::Scalar(buf.at<cv::Vec3b>(i)), -1, 8 );
                }
                cv::imshow("Histogram", histimg);cv::waitKey(1);
                
//                sprintf(text, "/Users/xuanpengli/Desktop/PIC/CAM_hist_%d.jpg", idx);
//                cv::imwrite(text, histimg);
            }
            
            cv::calcBackProject(&hue, 1, 0, hist_camshift, backproj, &phranges);
//            backproj &= mask;
            

            //! CamShift will change faceRoiRect.
            cv::RotatedRect trackBox = cv::CamShift(backproj, faceRoiRect, cv::TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 10, 1 ));
           
            if(1) {
                cv::cvtColor( backproj, backproj, cv::COLOR_GRAY2BGR );
                cv::ellipse( backproj, trackBox, CV_RGB(0,0,255), 3, CV_AA );
            }
            cv::imshow("backProj", backproj);
            
//            sprintf(text, "/Users/xuanpengli/Desktop/PIC/CAM_backproj_%d.jpg", idx);
//            cv::imwrite(text, backproj);
            
            
            cv::Rect brect = trackBox.boundingRect();
            cv::rectangle(faceMat, brect, CV_RGB(255, 0, 0), 2);
            

            
            //! Update the faceRoiRect
            faceRoiRect.x = cvRound(trackBox.center.x) - faceRoiRect.width/2;
            faceRoiRect.y = cvRound(trackBox.center.y) - faceRoiRect.height/2;
//            cv::rectangle(faceMat, cv::Point(faceRoiRect.x, faceRoiRect.y), cv::Point(faceRoiRect.x+ faceRoiRect.width, faceRoiRect.y+ faceRoiRect.height), CV_RGB(255, 255, 0), 3);
            
            //! Detect eyes in face tracking window
//            std::vector<cv::Rect> eyesRoiRect;
//            DetectEyes(faceMat, eyesCascade, faceRoiRect, eyesHeightVec, eyesPointVec, idx, eyesRoiRect);
//            
//            if(EYES_RECORD)
//            {
//                int iter_time = 0;
//                for (std::vector<cv::Rect>::const_iterator iter = eyesRoiRect.begin();  iter != eyesRoiRect.end(); ++iter) {
//                    cv::Mat eyeRoiMat = faceMat(*iter);
//                    char * colorEyesName = new char[50];
//                    sprintf(colorEyesName, EYE_COLOR_NAME, idx, iter_time);
//                    cv::imwrite(colorEyesName, eyeRoiMat);
//                    delete colorEyesName;
//                    
//                    ++iter_time;
//                }
//            }
        }
    
    }//end CamShiftTrackFace
 
    void CamShiftTrackFace2(cv::Mat &faceMat, cv::Rect &faceRoiRect,
                            cv::CascadeClassifier &eyesCascade, 
                            std::vector<double> &eyesHeightVec, 
                            std::vector<cv::Point> &eyesPointVec, int &trackKalman, 
                            cv::Mat &hist_camshift, const int idx)
    {
        //        Mat faceRoi1(faceMat, faceRoiRect); 
        //        imshow("faceROI", faceRoi1);
        
        int h_bins = 30, s_bins = 32;
        int histSize[] = {h_bins, s_bins};
        float hranges[] = {0,180};
        float sranges[] = {0,255};
        const float* phranges[] = {hranges, sranges};
        int ch[] = {0, 1};
        
        cv::Mat hsv, mask, histimg = cv::Mat::zeros(200, 320, CV_8UC3);
        cv::cvtColor(faceMat, hsv, cv::COLOR_RGB2HSV);
        
        
        //! trackKalman: -1(eyes detected), 1(eyes detected), 2(eyes tracked)
        if (trackKalman) {
            int _vmin = 10, _vmax = 256, smin = 50;
            
            cv::inRange(hsv, cv::Scalar(0, smin, MIN(_vmin,_vmax)),
                        cv::Scalar(180, 256, MAX(_vmin, _vmax)), mask);
            
//            //! Copy h-channel of hsv to hue mat
//            hs.create(hsv.size(), hsv.depth());
//            cv::mixChannels(&hsv, 1, &hue, 1, ch, 1);
            
            //! Just work when face detect finishes
            if( trackKalman < 0 ) {
                cv::Mat roi_face(hsv, faceRoiRect), maskroi(mask, faceRoiRect);
                imshow("hue", roi_face);imshow("mask", maskroi);
                
                cv::calcHist(&roi_face, 1, ch, maskroi, hist_camshift, 2, histSize, phranges);
                cv::normalize(hist_camshift, hist_camshift, 0, 255, CV_MINMAX);
                
//                for(int i = 0; i < hist_camshift.rows; i++)
//                    std::cout << hist_camshift.at<float>(i,0) << std::endl;
                
                trackKalman = 1;
            }
            
            cv::MatND backproj;
            cv::calcBackProject(&hsv, 1, ch, hist_camshift, backproj, phranges);
            cv::imshow("backProj", backproj);
            //backproj &= mask;
            
            //! CamShift will change faceRoiRect.
            cv::RotatedRect trackBox = cv::CamShift(backproj, faceRoiRect, cv::TermCriteria( CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 10, 1 ));
            //ellipse( faceMat, trackBox, CV_RGB(0,0,255), 3, CV_AA );
            
            if(0)
                cvtColor( backproj, faceMat, cv::COLOR_GRAY2BGR );
            
            cv::Rect brect = trackBox.boundingRect();
            cv::rectangle(faceMat, brect, CV_RGB(255, 255, 0), 2);
            
            //! Update the faceRoiRect
            faceRoiRect.x = cvRound(trackBox.center.x) - faceRoiRect.width/2;
            faceRoiRect.y = cvRound(trackBox.center.y) - faceRoiRect.height/2;
            //!!! Need adjust the edge
            cv::rectangle(faceMat, cv::Point(faceRoiRect.x, faceRoiRect.y), cv::Point(faceRoiRect.x+ faceRoiRect.width, faceRoiRect.y+ faceRoiRect.height), CV_RGB(255, 0, 255), 3);
            
            //! Detect eyes in face tracking window
            std::vector<cv::Rect> eyesRoiRect;
            DetectEyes(faceMat, eyesCascade, faceRoiRect, eyesHeightVec, eyesPointVec, idx, eyesRoiRect);
            
            if(EYES_RECORD)
            {
                int iter_time = 0;
                for (std::vector<cv::Rect>::const_iterator iter = eyesRoiRect.begin();  iter != eyesRoiRect.end(); ++iter) {
                    cv::Mat eyeRoiMat = faceMat(*iter);
                    char * colorEyesName = new char[50];
                    sprintf(colorEyesName, EYE_COLOR_NAME, idx, iter_time);
                    cv::imwrite(colorEyesName, eyeRoiMat);
                    delete colorEyesName;
                    
                    ++iter_time;
                }
            }
        }
        
    }//end CamShiftTrackFace

    
    
    /****************************************************************/
    // Kalman Filter Tracking
    //! trackKalman : -1 Detected; 1 Tracked; 0 Redetect
    /****************************************************************/
    void InitFaceKalmanFilter(cv::KalmanFilter &faceKalmanFilter, cv::Mat &faceKalmanMeasureMat)
    {
        faceKalmanFilter.transitionMatrix = (cv::Mat_<float>(4,4) <<
                                             1, 0, 0, 0, 
                                             0, 1, 0, 0,
                                             0, 0, 1, 0,
                                             0, 0, 0, 1);
        
        cv::setIdentity(faceKalmanFilter.measurementMatrix);
        cv::setIdentity(faceKalmanFilter.processNoiseCov, cv::Scalar::all(1e-5));
        cv::setIdentity(faceKalmanFilter.measurementNoiseCov, cv::Scalar::all(1e-1));
        cv::setIdentity(faceKalmanFilter.errorCovPost, cv::Scalar::all(1));
        cv::randn(faceKalmanFilter.statePost, cv::Scalar(0), cv::Scalar::all(0.1));
        
        //!Reset the State and Measure Mat
        faceKalmanMeasureMat.zeros(4, 1, CV_32F);
    }

    
    void TrackFace_KF(cv::Mat &faceMat, cv::Rect &faceRoiRect,
                      cv::CascadeClassifier &eyesCascade, 
                      std::vector<double> &eyesHeightVec, 
                      std::vector<cv::Point> &eyesPointVec, 
                      cv::KalmanFilter &faceKalmanFilter, cv::Mat &faceKalmanMeasureMat,
                      int &faceKalmanIndex, int &trackKalman, const int idx)
    {
        
        int flagEyeUpdated = 0;
        switch ((int)eyesPointVec.size()) {
            case 1:
//                printf("/*KalmanTrack*/ measure from last frame\n");
//                printf("eyes_pos: x %d, y %d, size %d\n", eyesPointVec.at(0).x, eyesPointVec.at(0).y, (int)eyesPointVec.size());
                
                if (eyesPointVec.at(0).x < faceRoiRect.x + faceRoiRect.width/2) {
                    faceKalmanMeasureMat.at<float>(0) = eyesPointVec.at(0).x;
                    faceKalmanMeasureMat.at<float>(1) = eyesPointVec.at(0).y;
                    faceKalmanMeasureMat.at<float>(2) = cvRound(100*COEF) + eyesPointVec.at(0).x;
                    faceKalmanMeasureMat.at<float>(3) = eyesPointVec.at(0).y;
                    flagEyeUpdated = 0;//left eye updated
                }
                else {
                    faceKalmanMeasureMat.at<float>(0) = eyesPointVec.at(0).x-cvRound(100*COEF);
                    faceKalmanMeasureMat.at<float>(1) = eyesPointVec.at(0).y;
                    faceKalmanMeasureMat.at<float>(2) = eyesPointVec.at(0).x;
                    faceKalmanMeasureMat.at<float>(3) = eyesPointVec.at(0).y;
                    flagEyeUpdated = 1;//right eye updated
                }
                
                break;
            case 2:
//                printf("/*KalmanTrack*/ measure from last frame\n");
//                printf("eyes_pos: x_a %d, y_a %d and x_b %d, y_b %d, size %d\n", eyesPointVec.at(0).x, eyesPointVec.at(0).y, eyesPointVec.at(1).x, eyesPointVec.at(1).y, (int)eyesPointVec.size());
                
                //left eye
                if (eyesPointVec.at(0).x < eyesPointVec.at(1).x)
                {
                    faceKalmanMeasureMat.at<float>(0) = eyesPointVec.at(0).x;
                    faceKalmanMeasureMat.at<float>(1) = eyesPointVec.at(0).y;
                    faceKalmanMeasureMat.at<float>(2) = eyesPointVec.at(1).x;
                    faceKalmanMeasureMat.at<float>(3) = eyesPointVec.at(1).y;

                }
                else
                {
                    faceKalmanMeasureMat.at<float>(0) = eyesPointVec.at(1).x;
                    faceKalmanMeasureMat.at<float>(1) = eyesPointVec.at(1).y;
                    faceKalmanMeasureMat.at<float>(2) = eyesPointVec.at(0).x;
                    faceKalmanMeasureMat.at<float>(3) = eyesPointVec.at(0).y;
                } 
                flagEyeUpdated = 2; //both eyes updated
                break;
            default:
                break;
        }
        
        std::vector<cv::Point> eyesTrackPointVec;
        std::vector<cv::Point> eyesPredictPointVec; 
        std::vector<cv::Point> eyesTempPointVec = eyesPointVec;//copy for use later
        
        // update face_roi from eyes_pos_predict updated;
        CatchFace(faceMat, faceKalmanFilter, faceKalmanMeasureMat, faceRoiRect, faceKalmanIndex, flagEyeUpdated, eyesPredictPointVec); 
        //imshow("CatchFace", faceMat(faceRoiRect));waitKey(1);
        
        /// Resize the ROI of face with eyes
        if (faceRoiRect.y < 0) {
            faceRoiRect.y = 0;
        }
        while (faceRoiRect.y + faceRoiRect.height > faceMat.rows) {
            faceRoiRect.height--;
        }
        if (faceRoiRect.x < 0) {
            faceRoiRect.x = 0;
        }
        while (faceRoiRect.x + faceRoiRect.width > faceMat.cols) {
            faceRoiRect.width--;
        }
        cv::rectangle(faceMat, cv::Point(faceRoiRect.x, faceRoiRect.y), 
                cv::Point(faceRoiRect.x + faceRoiRect.width, faceRoiRect.y + faceRoiRect.height), CV_RGB(255,0,255));
        
        std::vector<cv::Rect> eyesRoiRect;
        DetectEyes(faceMat, eyesCascade, faceRoiRect, eyesHeightVec, eyesPointVec, idx, eyesRoiRect);
        
//        for(int i = 0; i < (int)eyesHeightVec.size(); i++)
//            printf("Vec eyes height detected: %f\n", eyesHeightVec.at(i));
        
        if(EYES_RECORD) {
            std::cout << "TrackFace: Record color eyes" << std::endl;
            int iter_time = 0;
            for (std::vector<cv::Rect>::const_iterator iter = eyesRoiRect.begin();  iter != eyesRoiRect.end(); ++iter) {
                cv::Mat eyeRoiMat = faceMat(*iter);
                char * colorEyesName = new char[100];
                sprintf(colorEyesName, EYE_COLOR_NAME, idx, iter_time);
                cv::imwrite(colorEyesName, eyeRoiMat);
                delete colorEyesName;
                
                ++iter_time;
            }
        }
        
        trackKalman = 1;
        
        // If CANNOT detect eyes
        if (eyesHeightVec.empty()) {
            //printf("Cannot Detect Eyes...\n");
            // different type to track eye
            int trackType;
            if (faceKalmanIndex > 2) {
                trackType = KALMAN_PREDICT;
            }else {
                trackType = LAST_INSTEAD;
            }
            
            switch (trackType) {
                case KALMAN_PREDICT:
                    eyesTrackPointVec = eyesPredictPointVec;
                    break;
                case LAST_INSTEAD:
                    eyesTrackPointVec = eyesPointVec;
                    break;
                default:
                    break;
            }
            
            std::vector<cv::Rect> eyesTrackRoiRect;
            TrackEyes(faceMat, faceRoiRect, eyesHeightVec, eyesTrackPointVec, eyesPointVec, trackKalman, idx, eyesTrackRoiRect);
            
            if(EYES_RECORD){
                int iter_time = 0;
                for (std::vector<cv::Rect>::const_iterator iter = eyesRoiRect.begin();  iter != eyesRoiRect.end(); ++iter) {
                    cv::Mat eyeRoiMat = faceMat(*iter);
                    char * colorEyesName = new char[100];
                    sprintf(colorEyesName, EYE_COLOR_NAME, idx, iter_time);
                    cv::imwrite(colorEyesName, eyeRoiMat);
                    delete colorEyesName;
                    
                    ++iter_time;
                }
            }
            
            if (eyesPointVec.empty()) {
                eyesPointVec = eyesTempPointVec; 
            }
        }
//        else
            //printf("Detect Eyes Done...\n");
    }
    
    
    void CatchFace(cv::Mat &faceMat, 
                   cv::KalmanFilter &faceKalmanFilter, cv::Mat &faceKalmanMeasureMat,
                   cv::Rect &faceRoiRect, int &faceKalmanIndex, 
                   const int &flagEyeUpdated, std::vector<cv::Point> &eyesPredictPointVec)
    {
        const cv::Mat faceKalmanPredictMat = faceKalmanFilter.predict();

        cv::Point eyeCenter1 = cv::Point((int)faceKalmanPredictMat.at<float>(0), (int)faceKalmanPredictMat.at<float>(1));
        cv::Point eyeCenter2 = cv::Point((int)faceKalmanPredictMat.at<float>(2), (int)faceKalmanPredictMat.at<float>(3));
//        printf("/*CatchFace*/eye_predict: x_a %d, y_a %d and x_b %d, y_b %d\n", eyeCenter1.x, eyeCenter1.y, eyeCenter2.x, eyeCenter2.y);
        
        eyesPredictPointVec.clear();
        eyesPredictPointVec.push_back(eyeCenter1);
        eyesPredictPointVec.push_back(eyeCenter2);
        
        //update facei_roi after Kalman can track efficiently
        if (faceKalmanIndex > 2) {
            //printf("flag eye updated: %d\n", flagEyeUpdated);
            switch (flagEyeUpdated) {
                case 0://left updated
                    faceRoiRect = cv::Rect(eyeCenter1.x-cvRound(70*COEF), eyeCenter1.y-cvRound(60*COEF), cvRound(240*COEF), cvRound(240*COEF));
                    break;
                case 1://right updated
                    faceRoiRect = cv::Rect(eyeCenter2.x-cvRound(170*COEF), eyeCenter2.y-cvRound(60*COEF), cvRound(240*COEF), cvRound(240*COEF));
                    break;
                case 2://both updated
                    faceRoiRect = cv::Rect((eyeCenter1.x+eyeCenter2.x)/2-cvRound(120*COEF), (eyeCenter1.y+eyeCenter2.y)/2-cvRound(60*COEF), cvRound(240*COEF), cvRound(240*COEF));
                    break;
                default:
                    break;
            }
        }
        
        //printf("faceKalmanIndex: %d\n", faceKalmanIndex);
        faceKalmanIndex ++;
        
        faceKalmanFilter.correct(faceKalmanMeasureMat);
    }
    
    
    /// If Can Not Track eyes by Kalman, directly track eyes by last position
    /// Determine the region of eyes (eyes_pos) from eyes_pos_track;
    void TrackEyes(cv::Mat &faceMat, const cv::Rect &faceRoiRect,
                   std::vector<double> &eyesHeightVec, 
                   const std::vector<cv::Point> &eyesTrackPointVec, std::vector<cv::Point> &eyesPointVec, 
                   int &trackKalman, const int idx, std::vector<cv::Rect> &eyesRoiRect)
    { 
        cv::Mat grayMat;
		cv::cvtColor(faceMat, grayMat, cv::COLOR_BGR2GRAY);
        
        int iter_time = 0;
        for (int i=0; i < (int)eyesTrackPointVec.size(); i++) {
            int x = eyesTrackPointVec.at(i).x - cvRound(35*COEF); //printf("x : %d\n", x);
            int y = eyesTrackPointVec.at(i).y - cvRound(35*COEF); //printf("y : %d\n", y);
            cv::Rect eyeRoiRect = cv::Rect(x, y, cvRound(70*COEF), cvRound(70*COEF));
            
//            if (y + (int)eyeRoiRect.width/3 > (int)faceRoiRect.width/3 ) {
//                continue;
//            }
            
            eyesRoiRect.push_back(eyeRoiRect);
            
            /// HSV space 
            cv::Mat hsvMat;
            cv::cvtColor(faceMat(eyeRoiRect), hsvMat, cv::COLOR_RGB2HSV);

            int vPixelSum = 0, sPixelSum = 0, hPixelSum = 0;
            for(int i = 0; i < hsvMat.rows; i++){
                for (int j = 0; j < hsvMat.cols; j++) {
                    cv::Vec3b pixel = hsvMat.at<cv::Vec3b>(i,j); // read pixel (0,0) (make copy)
                    //printf("H: %d, S: %d, V: %d\n",  pixel[0], pixel[1], pixel[2]);
                    int vPixel = pixel[2];                
                    vPixelSum += vPixel;
                    
                    int sPixel = pixel[1];
                    sPixelSum += sPixel;
                    
                    int hPixel = pixel[0];
                    hPixelSum += hPixel;
                }
            }
            double vMean = (double)vPixelSum / (double)(hsvMat.rows * hsvMat.cols);
            double sMean = (double)sPixelSum / (double)(hsvMat.rows * hsvMat.cols);
//            double hMean = (double)hPixelSum / (double)(hsvMat.rows * hsvMat.cols);
            //printf("Idx: %d, v_mean: %d, s_mean: %d, h_mean: %d\n", idx, (int)vMean, (int)sMean, (int)hMean);
          
            
            /// Gray space
            cv::Mat grayEyeMat = grayMat(eyeRoiRect), binEyeMat;
            
            int operation = 2;//opening operation
            int morphSize = 2;//Kernel size: 2n +1
            int morphElem = 0;//Element: 0: Rect - 1: Cross - 2: Ellipse
            
            ///Hat Transform
            cv::Mat element = cv::getStructuringElement( morphElem, cv::Size( 2*morphSize + 1, 2*morphSize+1 ), cv::Point( morphSize, morphSize ) );
            /// Apply the specified morphology operation
            cv::morphologyEx( grayEyeMat, binEyeMat, operation, element );
//            cv::imshow("gray", grayEyeMat);cv::waitKey(0);
//            cv::imshow("hat", binEyeMat);
            
            /// Adaptive threshold_value based on satution and value in HSV. 
            int thresholdValue = 110;
            int maxBinValue = 255;
            int thresholdType = 0;
            
            /// Too light
            if (sMean < 14)                     thresholdValue = 170;
            else if(sMean >= 14 && sMean < 17)  thresholdValue = 150;
            else if(sMean >= 17 && sMean < 20)  thresholdValue = 130;
            else                                thresholdValue = 110;
            /// Too dark
            if (vMean < 200) thresholdValue = 80; 
            
            cv::threshold( grayEyeMat, binEyeMat, thresholdValue, maxBinValue, thresholdType);
            
//            cv::threshold( binEyeMat, binEyeMat, thresholdValue, maxBinValue, thresholdType);
//            cv::imshow("gray2", binEyeMat);cv::waitKey(0);
            
            if(EYES_RECORD){
                char * binEyesName = new char[50];
                sprintf(binEyesName, EYE_BIN_NAME, idx, iter_time);
                cv::imwrite(binEyesName, binEyeMat);
                delete binEyesName;
            }
            ++iter_time;

            /* Locate the eyes position */
            int top=0, bottom=0, left=0, right=0, flag = 0;// 1: can detect
            FindEyePts(binEyeMat, top, bottom, left, right, flag, idx);
            //printf("top: %d, bottom: %d, left: %d, right: %d", top, bottom, left, right);
            
            if(flag) {
                // Clear the last eyes points for recording the new one.
                eyesPointVec.clear();
                
                //pt1, pt2, center located in the original frame size
                cv::Point pt1(left + eyeRoiRect.x, top + eyeRoiRect.y);
                cv::Point pt2(right + eyeRoiRect.x, bottom + eyeRoiRect.y);
                cv::Point center((pt2.x + pt1.x)/2, (pt2.y + pt1.y)/2);
                
                if(0) {//EYES_RECORD
                    cv::rectangle(faceMat, pt1, pt2, CV_RGB(255,0,0), 1, 0, 0);
                    cv::circle(faceMat, center, 5, CV_RGB(255,255,0));
                }
                eyesPointVec.push_back(cv::Point((pt2.x + pt1.x)/2, (pt2.y + pt1.y)/2));//update eyes_pos vector;                
                
                int eyeHeight = abs(top - bottom);
                eyesHeightVec.push_back(eyeHeight);
            }
        }// for loop end

        /*Draw Eyes Region*/
        for (std::vector<cv::Rect>::const_iterator iter = eyesRoiRect.begin(); iter != eyesRoiRect.end(); ++iter)
            cv::rectangle( faceMat, cv::Point(iter->x, iter->y),
                  cv::Point(iter->x + iter->width, iter->y + iter->height), 
                  CV_RGB(0, 0, 255));
    
        trackKalman = 2;
    }
    
    
    /****************************************************************/
    //! Particle Filter Tracking
    //! trackParticle : -1 Detected; 1 Tracked; 0 Redetect
    //! Use the HSV histogram to track (Backproject has been tested)
    /****************************************************************/
    const int PARTICLE_NUM = 25; //Number of particles
    // Relative Areas
    const double A1 = 2.0;
    const double A2 = -1.0;
    const double B0 = 1.0;
    // Gaussian Sigma
    const double SIGMA_X = 1.0;
    const double SIGMA_Y = 0.5;
    const double SIGMA_SCALE = 0.001;
    
    //! Calculate the three channels (h, s, v) in HSV image
    int hbins = 180, sbins = 256, vbins = 256;
    int histSize[] = {hbins, sbins, vbins};
    // hue varies from 0 to 179, see cvtColor
    float hranges[] = { 0, 180 };
    // saturation varies from 0 (black-gray-white) to 255 (pure spectrum color)
    float sranges[] = { 0, 256 };
    // value varies from 0 (black-gray-white) to 255 (pure spectrum color)
    float vranges[] = { 0, 256 };
    const float* ranges[] = {hranges, sranges, vranges};
    // we compute the histogram from the 0-th and 1-st channels
    int channels[] = {0, 1, 2};
    
    
//    int histSize[] = {hbins, sbins};
//    // hue varies from 0 to 179, see cvtColor
//    float hranges[] = { 0, 180 };
//    // saturation varies from 0 (black-gray-white) to 255 (pure spectrum color)
//    float sranges[] = { 0, 256 };
//    // value varies from 0 (black-gray-white) to 255 (pure spectrum color)
//    float vranges[] = { 0, 256 };
//    const float* ranges[] = {hranges, sranges};
//    // we compute the histogram from the 0-th and 1-st channels
//    int channels[] = {0, 1};

    //! Fast sort algorithm
    bool particle_cmp(const PARTICLE_FACE &p1, const PARTICLE_FACE &p2)
    {
        return p1.weight > p2.weight;
    } 
    
    void InitFaceParticleFilter(cv::Mat &faceMat, const cv::Rect &faceRoiRect, cv::MatND &faceHist, std::vector<PARTICLE_FACE> &particles)
    {
        cv::Mat faceROI(faceMat, faceRoiRect);
        imShowSub("PF ROI Model", faceROI, WIN_COLS, WIN_ROWS, 1); 
        
        cv::Mat faceHsv;
        cv::cvtColor(faceROI, faceHsv, cv::COLOR_RGB2HSV);
        imShowSub("PF HSV Model", faceHsv, WIN_COLS, WIN_ROWS, 2);
        
        //! step 1: Get the histogram of face detected
        cv::calcHist(&faceHsv, 1, channels, cv::Mat(), faceHist, 3, histSize, ranges);
//        cv::calcHist(&faceHsv, 1, channels, cv::Mat(), faceHist, 2, histSize, ranges);
        cv::normalize(faceHist, faceHist);    //faceHist is a multi-dimention mat
 
#if 1
        cv::Mat histImg;
        cv::MatND backProject;
        DrawHistC3(faceHsv, histSize, ranges, histImg, backProject);
        imShowSub("Hist_Model", histImg, WIN_COLS, WIN_ROWS, 3);
        imShowSub("BackProj", backProject, WIN_COLS, WIN_ROWS, 4);
#endif
        
        // step 2: Init all the particles
        for(int i = 0; i < PARTICLE_NUM; i++)
        {
            PARTICLE_FACE particle;
            particle.p.x = faceRoiRect.x + 0.5 * faceRoiRect.width;
            particle.p.y = faceRoiRect.y + 0.5 * faceRoiRect.height;
            particle.pPre.x = particle.p.x;
            particle.pPre.y = particle.p.y;
            particle.pOri.x = particle.p.x;
            particle.pOri.y = particle.p.y;
            particle.rect = faceRoiRect;
            particle.scale = 1.0;
            particle.scalePre = 1.0;
            particle.hist = faceHist;
            particles.push_back(particle);
        }//end for
        
    }//end InitFaceParticleFilter
    
    void TrackFace_Particle(cv::Mat &faceMat, int &trackParticle, const cv::MatND &faceHist, std::vector<PARTICLE_FACE> &particles, const int &idx)
    {
        //! step 3: Update all the particles
        uint64 seed = idx + 1;
        cv::RNG rng(seed); 
        for (int i = 0; i < PARTICLE_NUM; i++) 
        {
            double x, y, scale; 
            
            particles[i].pPre = particles[i].p;//keep last position
            particles[i].scalePre = particles[i].scale;
            
            x = A1 * (particles[i].p.x - particles[i].pOri.x) +
                  A2 * (particles[i].pPre.x - particles[i].pOri.x) +
                  B0 * rng.gaussian(SIGMA_X) + particles[i].pOri.x;
            particles[i].p.x = std::max(0.0, std::min(x, faceMat.cols-1.0));
            
            y = A1 * (particles[i].p.y - particles[i].pOri.y) +
                  A2 * (particles[i].pPre.y - particles[i].pOri.y) +
                  B0 * rng.gaussian(SIGMA_Y) + particles[i].pOri.y;
            particles[i].p.y = std::max(0.0, std::min(y, faceMat.rows-1.0));
            
            scale = A1 * (particles[i].scale - 1.0) + 
                    A2 * (particles[i].scalePre - 1.0) + 
                    B0 * rng.gaussian(SIGMA_SCALE) + 1.0;
            particles[i].scale = std::max(0.5, std::min(scale, 2.0));
            
//            cout << x << ", " << y << ", " << scale << endl;
                
            particles[i].rect.x = std::max(0, std::min(cvRound(particles[i].p.x - 0.5 * particles[i].rect.width * particles[i].scale), faceMat.cols - 1));
            particles[i].rect.y = std::max(0, std::min(cvRound(particles[i].p.y - 0.5 * particles[i].rect.height * particles[i].scale), faceMat.rows - 1));
            particles[i].rect.width = std::min(cvRound(particles[i].rect.width *  particles[i].scale), faceMat.cols - particles[i].rect.x);
            particles[i].rect.height = std::min(cvRound(particles[i].rect.height *  particles[i].scale), faceMat.rows - particles[i].rect.y);
            //! Keep Pri
            
            //! step 4: Get the histogram of particle region
            cv::Mat particleMat(faceMat, particles[i].rect);
            cv::calcHist(&particleMat, 1, channels, cv::Mat(), particles[i].hist, 3, histSize, ranges);
//            cv::calcHist(&particleMat, 1, channels, cv::Mat(), particles[i].hist, 2, histSize, ranges);
            cv::normalize(particles[i].hist, particles[i].hist);
            
#if 0
//            double maxVal=0;
//            cv::minMaxLoc(particles[i].hist, 0, &maxVal, 0, 0);
//            int scaleHist = 2;
//            
//            cv::Mat histImg = cv::Mat::zeros(hbins * scaleHist, sbins * scaleHist, CV_8UC3);
//            for(int h = 0; h < hbins; h++)
//                for(int s = 0;  s < sbins; s++)
//                {
//                    float binVal = particles[i].hist.at<float>(h,s);
//                    int intensity = cvRound(binVal*255/maxVal);
//                    cv::rectangle(histImg, cv::Point(h * scaleHist,s * scaleHist), cv::Point((h+1)*scaleHist-1, (s+1)*scaleHist-1), cv::Scalar::all(intensity), CV_FILLED);
//                 }
//            
//            cv::imshow("Histogram", histImg);cv::waitKey(1); 
//            cv::waitKey(0);
#endif
            
            //CV_COMP_INTERSECT: 2                           
            particles[i].weight = compareHist(faceHist, particles[i].hist, 2);
        }
        
        
        
        // step 6: normalize the weight of all particles
        double sum = 0.0;
        int i;
        for(i=0; i<PARTICLE_NUM; i++)
        {
            sum += particles[i].weight;
        }
        for(i=0; i<PARTICLE_NUM; i++)
        {
            particles[i].weight /= sum;
        }
        
        sort(particles.begin(), particles.end(), particle_cmp);
        
        PARTICLE_FACE newParticles[PARTICLE_NUM];
        int np, k = 0;
        for(int i = 0; i < PARTICLE_NUM; i++)
        {
            np = cvRound(particles[i].weight * PARTICLE_NUM);
            for(int j = 0; j < np; j++)
            {
                newParticles[k++] = particles[i];
                if(k == PARTICLE_NUM)
                    goto EXITOUT;
            }
        }
        //! Complement the lacking particles
        while(k < PARTICLE_NUM)
        {
            newParticles[k++] = particles[0];
        }
        
    EXITOUT:
        for(int i = 0; i < PARTICLE_NUM; i++)
        {
            particles[i] = newParticles[i];
        }
        //! Renormalize the weights of particles
        double sum2 = 0.0;
        int j;
        for(j=0; j<PARTICLE_NUM; j++)
        {
            sum2 += particles[j].weight;
        }
        for(j=0; j<PARTICLE_NUM; j++)
        {
            particles[j].weight /= sum2;
        }
        
            
        
        sort(particles.begin(), particles.end(), particle_cmp);
        
        // step 8: Calculate the expect of particles
        cv::Rect_<double> rectTrackingTemp(0.0, 0.0, 0.0, 0.0);
        for(int i = 0; i < PARTICLE_NUM; i++)
        {
            rectTrackingTemp.x += particles[i].rect.x * particles[i].weight;
            rectTrackingTemp.y += particles[i].rect.y * particles[i].weight;
            rectTrackingTemp.width += particles[i].rect.width * particles[i].weight;
            rectTrackingTemp.height += particles[i].rect.height * particles[i].weight;
            printf("particle window: %f, %f, %f, %f\n", rectTrackingTemp.x, rectTrackingTemp.y, rectTrackingTemp.width, rectTrackingTemp.height);
        }
        cv::Rect_<double> rectTracking = rectTrackingTemp;
        
#if 0
        //! Draw the tracking results
        cv::Mat faceROI(faceMat, rectTracking);
        imShowSub("ROI_Tracked", faceROI, WIN_COLS, WIN_ROWS, 5);
        cv::Mat faceHsv;
        cv::cvtColor(faceROI, faceHsv, cv::COLOR_RGB2HSV);
        imShowSub("HSV_Tracked", faceHsv, WIN_COLS, WIN_ROWS, 6);
        
        cv::Mat histImg;
        cv::MatND backProject;
        DrawHistC3(faceHsv, histSize, ranges, histImg, backProject);
        imShowSub("Hist_Tracked", histImg, WIN_COLS, WIN_ROWS, 7);
        imShowSub("BackProj_Tracked", backProject, WIN_COLS, WIN_ROWS, 8);
        
        //! CamShift will change roiRect.
        cv::Mat mask;
        cv::inRange(faceROI, cv::Scalar(0, 50, 10), cv::Scalar(180, 256, 256), mask);
        imShowSub("mask_Tracked", mask, WIN_COLS, WIN_ROWS, 9);
        backProject &= mask;
        imShowSub("BackProj_Tracked2", backProject, WIN_COLS, WIN_ROWS, 10);
        
        cv::Rect roiRect = rectTracking;
        cv::RotatedRect trackBox = cv::CamShift(backProject, roiRect, cv::TermCriteria( CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 10, 1 ));
        
//        cv::Rect brect = trackBox.boundingRect();
//        //ellipse( faceMat, trackBox, CV_RGB(0,0,255), 3, CV_AA );
//        cv::rectangle(faceMat, brect, CV_RGB(255, 255, 0), 3);
#endif


        //! Show all the particles
        for(int i = 0; i < PARTICLE_NUM; i++)
        {
            cv::rectangle(faceMat, particles[i].rect, CV_RGB(0, 0, 255));
        }
        cv::rectangle(faceMat, rectTracking, CV_RGB(255, 0, 255),2);
        
        trackParticle = 1; //Tracked
    }//end TrackFace_Particle
    
    
    
}
