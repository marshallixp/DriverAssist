//
//  TransferFunctions.cpp
//  FusionCenter1.2
//
//  Created by XUANPENG LI on 11/11/13.
//  Copyright (c) 2013 ESIEE-Amiens. All rights reserved.
//

#include "TransferFunctions.h"
#include <iostream>
#include <cmath>

namespace DST {
    void BeltoB(const cv::Mat &bel, cv::Mat &b)
    {
        cv::Mat temp = bel.clone();
        int c = bel.cols;
        double a = 1 - bel.at<double>(0, c-1);
        temp.setTo(a);
        cv::add(bel, temp, b);
    }
    
    void BtoBel(const cv::Mat &b, cv::Mat &bel)
    {
        cv::Mat temp = b.clone();
        temp.setTo(b.at<double>(0,0));
        cv::subtract(b, temp, bel);
    }
    
    void BtoM(const cv::Mat &b, cv::Mat &m)
    {
        double lb = b.cols;
        double nbr_b = log2(lb);
        
        cv::Mat temp = b.clone();
        if (pow(2, nbr_b) == lb) {
            for(int i=0; i < (int)nbr_b; i++ ) {
                //double i124 = pow(2, i);
                double i842 = pow(2, nbr_b-i);
                double i421 = pow(2, nbr_b-i-1);
                
                //!Function reshape different from matlab in cols
                temp = temp.reshape(0, (int)i842);
                
                for(int j = 0; j < temp.cols; j++) {
                    for(int i = 0; i < (int)i421; i++) {
                        temp.at<double>(i*2+1, j) = temp.at<double>(i*2+1, j) - temp.at<double>(i*2, j);
                    }
                }
            }
            m = temp.reshape(0, 1);
        } else {
            printf("ERROR: there is wrong in BtoM\n");
            exit(1);
        }
    }
    
    // param: Mat b, pl in single row
    void BtoPl(const cv::Mat &b, cv::Mat &pl)
    {
        double lb = b.cols;
        double nbr_b = log2(lb);
        
        if(pow(2, nbr_b) == lb) {
            cv::Mat temp1;
            cv::flip(b, temp1, 1); 
            cv::Mat temp2 = cv::Mat::ones(b.rows, b.cols, CV_64F)*b.at<double>(b.rows-1, b.cols-1);
            cv::subtract(temp2, temp1, pl);
            pl.at<double>(0,0) = 0;
        } else {
            printf("ERROR: there is wrong in  BtoPl\n");
            exit(1);
        }
    }
    
    void BtoQ(const cv::Mat &b, cv::Mat &q)
    {
        cv::Mat pl; 
        BtoPl(b, pl);
        PltoQ(pl, q);
    }
    
    void MtoB(const cv::Mat &m, cv::Mat &b)
    {
        double lm = m.cols;
        double nbr_m = log2(lm);
        
        cv::Mat temp = m.clone();
        if (pow(2, nbr_m) == lm) {
            for(int i=0; i < (int)nbr_m; i++ ) {
                //double i124 = pow(2, i);
                double i842 = pow(2, nbr_m-i);
                double i421 = pow(2, nbr_m-i-1);
                
                //!Function reshape different from matlab in cols
                temp = temp.reshape(0, (int)i842);
                
                for(int j = 0; j < temp.cols; j++) {
                    for(int i = 0; i < (int)i421; i++) {
                        temp.at<double>(i*2+1, j) = temp.at<double>(i*2+1, j) + temp.at<double>(i*2, j);
                    }
                }
            }
            b = temp.reshape(0, 1);
        } else {
            printf("ERROR: there is wrong in MtoB\n");
            exit(1);
        }
    }
    
    
    void MtoBel(const cv::Mat &m, cv::Mat &bel) 
    {
        MtoB(m, bel);
        cv::Mat temp = cv::Mat::ones(m.rows, m.cols, CV_64F) * bel.at<double>(0,0);
        bel = bel - temp;
    }
    
    
    void MtoPl(const cv::Mat &m, cv::Mat &pl)
    {
        cv::Mat b;
        MtoB(m, b);
        BtoPl(b, pl);
    }
    
    
    void MtoQ(const cv::Mat &m, cv::Mat &q)
    {
        double lm = m.cols;
        double nbr_m = log2(lm);
        
        cv::Mat temp = m.clone();
        if (pow(2, nbr_m) == lm) {
            for(int i=0; i < (int)nbr_m; i++ ) {
                //double i124 = pow(2, i);
                double i842 = pow(2, nbr_m-i);
                double i421 = pow(2, nbr_m-i-1);
                
                //!Function reshape different from matlab in cols
                temp = temp.reshape(0, (int)i842);
                
                for(int j = 0; j < temp.cols; j++) {
                    for(int i = 0; i < (int)i421; i++) {
                        temp.at<double>(i*2, j) = temp.at<double>(i*2, j) + temp.at<double>(i*2+1, j);
                    }
                }
            }
            
            q = temp.reshape(0, 1);
        } else {
            printf("ERROR: there is wrong in MtoQ\n");
            exit(1);
        }
    }
    
    void PltoB(const cv::Mat &pl, cv::Mat &b)
    {
        double lpl = pl.cols;
        double nbr_pl = log2(lpl);
        
        cv::Mat temp1;
        if (pow(2, nbr_pl) == lpl) {
            cv::flip(pl, temp1, 1); 
            b = cv::Mat::ones(pl.rows, pl.cols, CV_64F) - temp1;
        } else {
            printf("ERROR: there is wrong in PltoB\n");
            exit(1);
        }
    }
    
    void PltoBel(const cv::Mat &pl, cv::Mat &bel)
    {
        PltoB(pl, bel);
        bel = bel - cv::Mat::ones(bel.rows, bel.cols, CV_64F) * bel.at<double>(0,0);
    }
    
    void PltoM(const cv::Mat &pl, cv::Mat &m)
    {
        cv::Mat b;
        PltoB(pl, b);
        BtoM(b, m);
    }
    
    void PltoQ(const cv::Mat &pl, cv::Mat &q)
    {
        cv::Mat m;
        BtoM(pl, m);
        q = cv::abs(m);
        q.at<double>(0,0) = 1;
    }
    
    //! Q
    void QtoB(const cv::Mat &q, cv::Mat &b)
    {
        cv::Mat pl;
        QtoPl(q, pl);
        PltoB(pl, b);
    }
    
    void QtoBel(const cv::Mat &q, cv::Mat &bel)
    {
        cv::Mat b;
        QtoB(q, b);
        BtoBel(b, bel);
    }
    
    void QtoM(const cv::Mat &q, cv::Mat &m)
    {
        double lq = q.cols;
        double nbr_q = log2(lq);
        
        cv::Mat temp = q.clone();
        if (pow(2, nbr_q) == lq) {
            for(int i=0; i < (int)nbr_q; i++ ) {
                //double i124 = pow(2, i);
                double i842 = pow(2, nbr_q-i);
                double i421 = pow(2, nbr_q-i-1);
                
                //!Function reshape different from matlab in cols
                temp = temp.reshape(0, (int)i842);
                
                for(int j = 0; j < temp.cols; j++) {
                    for(int i = 0; i < (int)i421; i++) {
                        temp.at<double>(i*2, j) = temp.at<double>(i*2, j) - temp.at<double>(i*2+1, j);
                    }
                }
            }
            m = temp.reshape(0, 1);
        } else {
            printf("ERROR: there is wrong in QtoM\n");
            exit(1);
        }
    }
    
    void QtoPl(const cv::Mat &q, cv::Mat &pl)
    {
        cv::Mat b = q.clone();
        b.at<double>(0,0) = 0;
        BtoM(b, pl);
        cv::abs(pl);
    }
    
    void QtoW(const cv::Mat &q, cv::Mat &w)
    {
        double lq = q.cols;
        double nbr_q = log2(lq);
        
        cv::Mat temp = q.clone();
        if (pow(2, nbr_q) == lq) {
            for(int i=0; i < (int)nbr_q; i++ ) {
                //double i124 = pow(2, i);
                double i842 = pow(2, nbr_q-i);
                double i421 = pow(2, nbr_q-i-1);
                
                //!Function reshape different from matlab in cols
                temp = temp.reshape(0, (int)i842);
                
                for(int j = 0; j < temp.cols; j++) {
                    for(int i = 0; i < (int)i421; i++) {
                        temp.at<double>(i*2, j) = pow((temp.at<double>(i*2+1,j) / temp.at<double>(i*2,j)), (-1));
                    }
                }
            }
            cv::pow(temp, (-1), temp);
            w = temp.reshape(0, 1);
            w.at<double>(w.rows-1, w.cols-1) = 0;
        } else {
            printf("ERROR: there is wrong in QtoW\n");
            exit(1);
        }
    }
    
}//namespace
