//
//  Tools.cpp
//  FusionCenter1.2
//
//  Created by XUANPENG LI on 11/11/13.
//  Copyright (c) 2013 __MyCompanyName__. All rights reserved.
//
#include "Tools.h"
#include <iostream>

namespace DST {
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
    
}