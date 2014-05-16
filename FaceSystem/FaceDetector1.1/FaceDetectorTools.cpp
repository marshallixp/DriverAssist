//
//  FaceDetectorTools.cpp
//  FaceDetector1.1
//
//  Created by XUANPENG LI on 11/09/13.
//  Copyright (c) 2013 __MyCompanyName__. All rights reserved.
//
#include "FaceDetectorTools.h"
#include <iostream>
extern int WIN_COLS;
extern int WIN_ROWS;

const char *HUE_NAME = "/Users/xuanpengli/Desktop/FIG/PF_hue.jpg";
const char *SAT_NAME = "/Users/xuanpengli/Desktop/FIG/PF_sat.jpg";
const char *VAL_NAME = "/Users/xuanpengli/Desktop/FIG/PF_val.jpg";
const char *HIST_NAME = "/Users/xuanpengli/Desktop/FIG/PF_hist.jpg";

namespace FaceDetector{
    void DrawHistC3(const cv::Mat &mat, const int *histSize, const float *ranges[], cv::Mat &histImg, cv::MatND &backProject)
    {
        //! split the channels(h,s,v), then 
        std::vector<cv::Mat> planes;
        cv::split(mat, planes);
//        cv::imshow("Track", mat);
        cv::imshow("Hue", planes[0]);
        cv::imshow("Saturation", planes[1]);
        cv::imshow("Value", planes[2]);
        
        cv::Mat a_hist, b_hist, c_hist;
        cv::calcHist(&planes[0], 1, 0, cv::Mat(), a_hist, 1, &histSize[0], &ranges[0]);
        cv::calcHist(&planes[1], 1, 0, cv::Mat(), b_hist, 1, &histSize[1], &ranges[1]);
        cv::calcHist(&planes[2], 1, 0, cv::Mat(), c_hist, 1, &histSize[2], &ranges[2]);
        
        int hist_weight = 300, hist_height = 256;
        //! bins for the rectangles
        int bin_a = cvRound((double)hist_weight / histSize[0]);
        int bin_b = cvRound((double)hist_weight / histSize[1]);
        int bin_c = cvRound((double)hist_weight / histSize[2]);
        
        histImg = cv::Mat(hist_height, hist_weight, CV_8UC3, cv::Scalar::all(255));
        //cv::imshow("hist", histImg);cv::waitKey(0);
            
        cv::normalize(a_hist, a_hist, 0, hist_height, cv::NORM_MINMAX, -1, cv::Mat()); //0~255
        cv::normalize(b_hist, b_hist, 0, hist_height, cv::NORM_MINMAX, -1, cv::Mat()); //0~255
        cv::normalize(c_hist, c_hist, 0, hist_height, cv::NORM_MINMAX, -1, cv::Mat()); //0~255
        
        //! Get the back project
        cv::calcBackProject(&planes[0], 1, 0, b_hist, backProject, ranges);
        
        //! Generate the histogram figure
        for(int i=1; i<histSize[0]; i++) //hue 180
        {
            cv::line( histImg, cv::Point( (i-1), hist_height - cvRound(a_hist.at<float>(i-1)) ), cv::Point( (i), hist_height - cvRound(a_hist.at<float>(i))), CV_RGB( 255, 0, 0), 2, 8, 0  );
//            cv::rectangle( histImg, cv::Point( i*bin_a, hist_height ), cv::Point( (i+1)*bin_a, hist_height - cvRound( a_hist.at<float>(i)*hist_height/255.0 ) ), CV_RGB( 255, 0, 0 ), -1 );
        }//end for
        
        for(int i=1; i < histSize[1]; i++) //sat 256
        {
            cv::line( histImg, cv::Point( (i-1), hist_height - cvRound(b_hist.at<float>(i-1)) ), cv::Point( (i), hist_height - cvRound(b_hist.at<float>(i)) ), CV_RGB( 0, 255, 0), 2, 8, 0  );
//            cv::rectangle( histImg, cv::Point( i*bin_b, hist_height ), cv::Point( (i+1)*bin_b, hist_height - cvRound( b_hist.at<float>(i)*hist_height/255.0 ) ), CV_RGB( 0, 255, 0), -1 );
        }//end for
        
        for(int i=1; i < histSize[2]; i++) //val 256
        {   
            cv::line( histImg, cv::Point( (i-1), hist_height - cvRound(c_hist.at<float>(i-1)) ) , cv::Point( (i), hist_height - cvRound(c_hist.at<float>(i)) ), CV_RGB( 0, 0, 255), 2, 8, 0  );
//            cv::rectangle( histImg, cv::Point( i*bin_c, hist_height ), cv::Point( (i+1)*bin_c, hist_height - cvRound( c_hist.at<float>(i)*hist_height/255.0 ) ), CV_RGB( 0, 0, 255 ), -1 );
        }//end for
        
//        char huePath[100], satPath[100], valPath[100], histPath[100]; 
//        sprintf(huePath, HUE_NAME);
//        cv::imwrite(huePath, planes[0]);
//        sprintf(satPath, SAT_NAME);
//        cv::imwrite(satPath, planes[1]);
//        sprintf(valPath, VAL_NAME);
//        cv::imwrite(valPath, planes[2]);
//        sprintf(histPath, HIST_NAME);
//        cv::imwrite(histPath, histImg);
//        cv::waitKey(0);
    }//end DrawHistC3
    
    
    void PrintMat(const cv::Mat &mat)
    {
        
        for(int i = 0; i < mat.rows; i++) {
            for(int j = 0; j < mat.cols; j++) {
                if(mat.type() == CV_64F)
                    std::cout << mat.at<double>(i,j) <<' ';
                else if(mat.type() == CV_32F)
                    std::cout << mat.at<float>(i,j) << ' ';
                else
                    std::cout << (int)mat.at<uchar>(i,j) << ' ';
            }
            std::cout << std::endl;
        }
    }//end PrintMat
    
    
    //! This function works like the same one in MATLAB
    void imShowSub(const std::string &winname, const cv::Mat &mat, 
                   const int &Cols, const int &Rows, const int &winPos)
    {
        int width = cvRound((double)1200 / (double)Cols);
        int height = cvRound((double)750 / (double)Rows);
        
        CV_Assert(winPos <= Rows * Cols);
        int col = (winPos - 1) % Cols;
        int yWin = height * (winPos-1-col) / Cols + 30;
        int xWin = width * col;//Add offset 
        
        //std::cout << "xWin: " << xWin << " yWin: " << yWin << std::endl;
        cv::imshow(winname, mat);
        cv::moveWindow(winname, xWin, yWin);
    }//end imShowSub
    
}//end FaceDetector