//
//  CombineRules.h
//  FusionCenter1.2
//
//  Created by XUANPENG LI on 11/11/13.
//  Copyright (c) 2013 __MyCompanyName__. All rights reserved.
//

#ifndef FusionCenter_CombineRules_h
#define FusionCenter_CombineRules_h

#include <opencv2/opencv.hpp>


namespace DST {
    void NormalizeBBA(const cv::Mat &bba, cv::Mat &bba_norm);
    
    void Combine_Conj(const cv::Mat &bba, cv::Mat &bba_cmb);
    
    void Combine_Disj(const cv::Mat &bba, cv::Mat &bba_cmb);
    
    void Combine_Yager(const cv::Mat &bba, cv::Mat &bba_cmb);
    
    void Combine_DuboisPrade(const cv::Mat &bba, cv::Mat &bba_cmb);
    
    void Combine_DPNonDistinct(const cv::Mat &bba, cv::Mat &bba_cmb);

    void Combine_Dempster(const cv::Mat &bba, cv::Mat &bba_cmb);
    
    void Combine_CautiousConj(const cv::Mat &bba, cv::Mat &bba_cmb);
    
    void Combine_CautiousConjNorm(const cv::Mat &bba, cv::Mat &bba_cmb);
    
    void Combine_BoldDisj(const cv::Mat &bba, cv::Mat &bba_cmb);
}


#endif
