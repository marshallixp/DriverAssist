//
//  DetectFace.cpp
//  FaceDetector1.1
//
//  Created by Xuanpeng Li on 05/13.
//  Copyright (c) 2012 ESIEE-AMIENS. All rights reserved.
//
#include "DetectFace.h"

extern const double  COEF;

extern const int  EYES_RECORD;
extern const char EYE_COLOR_NAME[];
extern const char EYE_BIN_NAME[];

const char EYE_HSV_NAME[] = "/Users/xuanpengli/DriveAssist/Capture/hsv_eye/eye_bin_%d_%d_%d.jpg";

extern const int  MAX_VALUE = 255;

namespace FaceDetector{
    /* This function used the methodology in Gracia's work */
    void FindEyePts(const cv::Mat &src, 
                     int &top, int &bottom,
                     int &left, int &right, int &flag, const int &idx)
    {      
        int i, j;//i: rows, j: cols
        /// Get bottom
        for (i = src.rows-1; i >= 3; i--) {
            int flag_col0 = 0, flag_col1 = 0, flag_col2 = 0, flag_col3 = 0;
            for (j = 0; j < src.cols ; j++) {
                int val0 = (int)src.at<uchar>(i,j);//printf("val: %d", val);
                int val1 = (int)src.at<uchar>(i-1,j);
                int val2 = (int)src.at<uchar>(i-2,j);
                int val3 = (int)src.at<uchar>(i-3,j);
                if (val0 != MAX_VALUE) flag_col0++;
                if (val1 != MAX_VALUE) flag_col1++;
                if (val2 != MAX_VALUE) flag_col2++;
                if (val3 != MAX_VALUE) flag_col3++;
            }
            if (flag_col0 != 0 && (flag_col1 != 0 || flag_col2 != 0 || flag_col3 != 0)){
                bottom = i;
                goto TOP;
            }
            else {
                bottom = 0;
                continue;
            }
                
        }
       
        if (bottom == 0) {
            printf("Frame %d: FindEyePts: Cannot Find Eyes!\n", idx);
            return;
        } 

        /// Get top, must make sure that the three following position equal to 0
    TOP: for (i = bottom; i >= 2; i--){
            int flag_col0 = 0, flag_col1 = 0, flag_col2 = 0;
            for (j= 0; j < src.cols; j++) {
                int val0 = (int)src.at<uchar>(i,j);
                int val1 = (int)src.at<uchar>(i-1,j);
                int val2 = (int)src.at<uchar>(i-2,j);
                if (val0 != MAX_VALUE) flag_col0++;
                if (val1 != MAX_VALUE) flag_col1++;
                if (val2 != MAX_VALUE) flag_col2++;
            }
            if (flag_col0 == 0 && flag_col1== 0 && flag_col2 == 0) {
                top = i + 1;//printf("top: %d, bottom: %d\n", top, bottom);
                goto LEFT;
            }
        }
        /// Get left
    LEFT: for(j = 0; j< src.cols; j++){
            for(i = top; i <= bottom; i++){
                int val = (int)src.at<uchar>(i,j);
                if (val != MAX_VALUE) {
                    left = j;
                    goto RIGHT;
                }
            }
        }
        /// Get right
    RIGHT: for(j = src.cols-1; j >= 0; j--){
            for(i = top; i <= bottom; i++ ){
                int val = (int)src.at<uchar>(i,j);
                if (val != MAX_VALUE) {
                    right = j;
                    flag = 1;// It means get the top and bottom
                    return;
                }
            }
        }
    }
    
    bool SortRectYBigger(const cv::Rect &value1, const cv::Rect &value2) {
        return value1.y < value2.y;
    }
	
    
    void DetectEyes(cv::Mat &faceMat, 
                    cv::CascadeClassifier &eyesCascade, 
                    const cv::Rect &faceRoiRect, 
                    std::vector<double> &eyesHeightVec,
                    std::vector<cv::Point> &eyesPointVec, 
                    const int idx, std::vector<cv::Rect> &eyesRoiRect)
    {
        cv::Mat grayMat;
		cv::cvtColor(faceMat, grayMat, cv::COLOR_BGR2GRAY);
        // printf("FaceRoi_Region x:%d, y:%d, width:%d, height:%d\n", faceRoiRect.x, faceRoiRect.y, faceRoiRect.width, faceRoiRect.height);
        // cv::imshow("gray_face", grayMat(faceRoiRect));
        
        /* Detect possible eyes region */
        std::vector<cv::Rect> eyesRect;
        eyesCascade.detectMultiScale(grayMat(faceRoiRect), eyesRect, 1.1, 3, 0|cv::CASCADE_SCALE_IMAGE, cv::Size(cvRound(40*COEF), cvRound(20*COEF)));
        
        /* Keep potential eyes region */
        if  ((int)eyesRect.size() > 1) {
            sort(eyesRect.begin(), eyesRect.end(), SortRectYBigger);
            // for(size_t i = 0; i < eyesRect.size(); i++) {
            //     printf("Y: %d, X: %d\n", eyesRect.at(i).y, eyesRect.at(i).x);
            // }
            
            int th1 = cvRound(COEF * 20);
            for (size_t i = 0; i < eyesRect.size()-1; ) {
                int dis1 = abs(eyesRect.at(i).y - eyesRect.at(i+1).y);
                //printf("distance: %d, th: %d\n", dis_height, (int)th_height);
                if ( dis1 > th1) {
                    //printf("Exceed...\n");
                    eyesRect.erase(eyesRect.begin()+(i+1), eyesRect.end());
                    break;
                } else {
                    //printf("continue..\n");
                    i++;
                }
            } 
        }// end if loop
       
        
        /* Remove the incorrected eyes region */
        if (!eyesRect.empty() && !eyesPointVec.empty()) {
            int th2 = cvRound(COEF * 40); 
            int dis2 = abs(eyesRect.at(0).y + eyesRect.at(0).height/2 +faceRoiRect.y - eyesPointVec.at(0).y);
            if ( dis2 < th2 ) {
                //! Keep current eyes region
                eyesPointVec.clear();
            } else {
                eyesRect.clear();
            }
        }
        
        
        /* Locate eyes in the possible eyes region */
        if (!eyesRect.empty()) { 
            //printf("DetectEyes: Possibly Can Detect Eyes...\n");
            int iter_time = 0;
            for (std::vector<cv::Rect>::const_iterator iter = eyesRect.begin(); iter != eyesRect.end(); ++iter) { 
                const int scale = cvRound(10*COEF); //cause of inaccurate area of eyes detected
                cv::Rect eyeRoiRect = cv::Rect(faceRoiRect.x + iter->x, 
                                       faceRoiRect.y + iter->y + scale,
                                       iter->width, 
                                       iter->height-scale);
                /// Record eyes region
                eyesRoiRect.push_back(eyeRoiRect);
               
                /// This method needs a strong lighting in the front!
                /// HSV space 
                cv::Mat hsvEyeMat;
                cv::cvtColor(faceMat(eyeRoiRect), hsvEyeMat, cv::COLOR_RGB2HSV);
                std::vector<cv::Mat> planes;
                cv::split(hsvEyeMat, planes); //cv::imshow("hsv_eye", hsvEyeMat);
                
               // char hsvEye[100]; 
               // sprintf(hsvEye, EYE_HSV_NAME, idx, iter_time, 0);//h
               // cv::imwrite(hsvEye, planes[0]);
               // sprintf(hsvEye, EYE_HSV_NAME, idx, iter_time, 1);//h
               // cv::imwrite(hsvEye, planes[1]);
               // sprintf(hsvEye, EYE_HSV_NAME, idx, iter_time, 2);//h
               // cv::imwrite(hsvEye, planes[2]);
                
                int vPixelSum = 0, sPixelSum = 0, hPixelSum = 0;
                for(int i = 0; i < hsvEyeMat.rows; i++){
                    for (int j = 0; j < hsvEyeMat.cols; j++) {
                        cv::Vec3b pixel = hsvEyeMat.at<cv::Vec3b>(i,j); // read pixel (0,0) (make copy)
                        //printf("H: %d, S: %d, V: %d\n",  pixel[0], pixel[1], pixel[2]);
                        int vPixel = pixel[2];                
                        vPixelSum += vPixel;
                        
                        int sPixel = pixel[1];
                        sPixelSum += sPixel;
                        
                        int hPixel = pixel[0];
                        hPixelSum += hPixel;
                    }
                }
                double vMean = (double)vPixelSum / (double)(hsvEyeMat.rows * hsvEyeMat.cols);
                double sMean = (double)sPixelSum / (double)(hsvEyeMat.rows * hsvEyeMat.cols);
                // double hMean = (double)hPixelSum / (double)(hsvEyeMat.rows * hsvEyeMat.cols);
                // printf("Idx: %d, v_mean: %d, s_mean: %d, h_mean: %d\n", idx, (int)v_mean, (int)s_mean, (int)h_mean);
                
                /// Gray space
                cv::Mat grayEyeMat = grayMat(eyeRoiRect), binEyeMat;
                
                int operation = 2;//opening operation
                int morphSize = 2;//Kernel size: 2n +1
                int morphElem = 0;//Element: 0: Rect - 1: Cross - 2: Ellipse
                
                ///Hat Transform
                cv::Mat element = getStructuringElement( morphElem, cv::Size( 2*morphSize + 1, 2*morphSize+1 ), cv::Point( morphSize, morphSize ) );
                /// Apply the specified morphology operation
                cv::morphologyEx( grayEyeMat, binEyeMat, operation, element );
                
                /// Adaptive threshold_value based on satution and value in HSV. 
                int thresholdValue = 110;
                int maxBinValue = 255;
                int thresholdType = 0;
                
                /// Too light
                if (sMean < 14)                     thresholdValue = 170;
                else if(sMean > 13 && sMean < 17)   thresholdValue = 150;
                else if(sMean >= 17 && sMean < 20)  thresholdValue = 130;
                else                                thresholdValue = 110;
                
                if (vMean < 200) thresholdValue = 80; 
                
                cv::threshold( binEyeMat, binEyeMat, thresholdValue, maxBinValue, thresholdType);
                //imshow("DetectEyes_binEyes", binEyeMat);waitKey(1);
                
                if(EYES_RECORD){
                    char * binEyesName = new char[50];
                    sprintf(binEyesName, EYE_BIN_NAME, idx, iter_time);
                    cv::imwrite(binEyesName, binEyeMat);
                    delete binEyesName;
                }
                
                ++iter_time;
                
                /* Locate the eyes position       */
                int top=0, bottom=0, left=0, right=0, flag = 0;// 1: can detect
                FindEyePts(binEyeMat, top, bottom, left, right, flag, idx);
                //printf("top: %d, bottom: %d, left: %d, right: %d", top, bottom, left, right);
                
                if(flag) {
                //pt1, pt2, center located in the original frame size
                    cv::Point pt1(left + eyeRoiRect.x, top + eyeRoiRect.y);
                    cv::Point pt2(right + eyeRoiRect.x, bottom + eyeRoiRect.y);
                    cv::Point center((pt2.x + pt1.x)/2, (pt2.y + pt1.y)/2);
                    
                    if(!EYES_RECORD) {
                        cv::rectangle(faceMat, pt1, pt2, CV_RGB(255,0,0), 1, 0, 0);
                        cv::circle(faceMat, center, 5, CV_RGB(255,255,0));
                    }
                    eyesPointVec.push_back(cv::Point((pt2.x + pt1.x)/2, (pt2.y + pt1.y)/2));//update eyes_pos vector;                

                    double eyeHeight = abs(top - bottom);
                    eyesHeightVec.push_back(eyeHeight);
                } 

            }// for loop end


            /*Draw Eyes Region*/
            for (std::vector<cv::Rect>::const_iterator iter = eyesRoiRect.begin(); iter != eyesRoiRect.end(); ++iter)
                cv::rectangle( faceMat, cv::Point(iter->x, iter->y),
                      cv::Point(iter->x + iter->width, iter->y + iter->height), 
                      CV_RGB(0, 0, 255));
        } 
    }

    
	void DetectFace(cv::Mat &faceMat,
                    cv::CascadeClassifier &faceCascade,
                    cv::CascadeClassifier &eyesCascade,
                    cv::Rect &faceRoiRect, 
                    std::vector<double> &eyesHeightVec,
                    std::vector<cv::Point> &eyesPointVec, 
                    int &trackKalman, const int idx)
	{
        cv::Mat grayMat;
		cv::cvtColor(faceMat, grayMat, cv::COLOR_BGR2GRAY);
        
        std::vector<cv::Rect> faceRect;
		// Detect faces
        faceCascade.detectMultiScale(grayMat, faceRect, 1.1, 3, 0|cv::CASCADE_SCALE_IMAGE, cv::Size(cvRound(40*COEF),cvRound(20*COEF)));
        
        if((int)faceRect.size() > 0) {
            for (std::vector<cv::Rect>::const_iterator iter = faceRect.begin(); iter != faceRect.end(); ++iter) {
                cv::rectangle(faceMat, 
                          cv::Point(iter->x, iter->y),
                          cv::Point(iter->x + iter->width, iter->y + iter->height), 
                          CV_RGB(0,255,0));
                
                //Assume the face size is bigger than some threshold
                if(iter->width > cvRound(200*COEF) && iter->height > cvRound(200*COEF)) {  
                    faceRoiRect = *iter;
                    
                    std::vector<cv::Rect> eyesRoiRect;
                    DetectEyes(faceMat, eyesCascade, *iter, eyesHeightVec, eyesPointVec, idx, eyesRoiRect);

                    
                    if(EYES_RECORD) {
                        std::cout << "Record color eyes" << std::endl;
                        int iter_time = 0;
                        for (std::vector<cv::Rect>::const_iterator iter = eyesRoiRect.begin();  iter != eyesRoiRect.end(); ++iter) {
                            cv::Mat eyeRoiMat = faceMat(*iter);
                            
                            char * colorEyesName = new char[100];
                            sprintf(colorEyesName, EYE_COLOR_NAME, idx, iter_time);
                            cv::imwrite(colorEyesName, eyeRoiMat);
                            delete colorEyesName;
                            
                            ++iter_time;
                        }
                        cv::waitKey(1);
                    }
                    trackKalman = -1;
                }//if end
            }//for loop end
        } else {
            //printf("Can not detect face.\n");
        }
        //cv::imshow("Face Tracking", faceMat); waitKey(1);
	}
    
}//namespace
