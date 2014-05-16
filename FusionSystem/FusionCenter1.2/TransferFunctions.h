//
//  TransferFunctions.h
//  FusionCenter1.2
//
//  Created by XUANPENG LI on 11/11/13.
//  Copyright (c) 2013 __MyCompanyName__. All rights reserved.
//

#ifndef FusionCenter_TransferFunctions_h
#define FusionCenter_TransferFunctions_h

#include <opencv2/opencv.hpp>

namespace DST {
    //! B
    void BeltoB(const cv::Mat &bel, cv::Mat &b);

    void BtoBel(const cv::Mat &b, cv::Mat &bel);
    
    void BtoM(const cv::Mat &b, cv::Mat &m);
    
    void BtoPl(const cv::Mat &b, cv::Mat &pl);
    
    void BtoQ(const cv::Mat &b, cv::Mat &q);
    
    //! M
    void MtoB(const cv::Mat &m, cv::Mat &b);

    void MtoBel(const cv::Mat &m, cv::Mat &bel);
   
    void MtoPl(const cv::Mat &m, cv::Mat &pl);
    
    void MtoQ(const cv::Mat &m, cv::Mat &q);
    
    //! Pl
    void PltoB(const cv::Mat &pl, cv::Mat &b);
    
    void PltoBel(const cv::Mat &pl, cv::Mat &bel);
    
    void PltoM(const cv::Mat &pl, cv::Mat &m);
    
    void PltoQ(const cv::Mat &pl, cv::Mat &q);
    
    //! Q
    void QtoB(const cv::Mat &q, cv::Mat &b);
    
    void QtoBel(const cv::Mat &q, cv::Mat &bel);
    
    void QtoM(const cv::Mat &q, cv::Mat &m);
    
    void QtoPl(const cv::Mat &q, cv::Mat &pl);
    
    void QtoW(const cv::Mat &q, cv::Mat &w);
}

#endif
