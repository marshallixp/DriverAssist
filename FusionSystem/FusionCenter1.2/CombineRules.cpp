//
//  CombineRules.cpp
//  FusionCenter1.2
//
//  Created by XUANPENG LI on 11/11/13.
//  Copyright (c) 2013 __MyCompanyName__. All rights reserved.
//

#include "CombineRules.h"
#include "TransferFunctions.h"
#include "Tools.h"
#include <iostream>


namespace DST {
    //! Normalize Process
    void NormalizeBBA(const cv::Mat &bba, cv::Mat &bba_norm)
    {
        int nbr_elem = bba.cols;
        double scale = 1 - bba.at<double>(0,0);
        for (int i = 0; i < nbr_elem; i++) {
            bba_norm.at<double>(0,i) = bba.at<double>(0,i)/scale;
        }
        bba_norm.at<double>(0,0) = 0;
    }
    
    //! Conjunctive Rule
    void Combine_Conj(const cv::Mat &bba, cv::Mat &bba_cmb)
    {
        int nbr_src = bba.rows;
        int nbr_elem = bba.cols;
        
        cv::Mat q(nbr_src, nbr_elem, CV_64F), temp;
        for(int i = 0; i < nbr_src; i++) {
            cv::Mat m = bba.row(i);
            MtoQ(m, temp);
            temp.copyTo(q.row(i));
        }
        
        for(int i = 0; i < (int)nbr_src - 1; i++) {
            cv::multiply(q.row(i), q.row(i+1), q.row(i+1));
        }
        
        cv::Mat q_cmb = q.row(nbr_src-1);
        QtoM(q_cmb, bba_cmb);
    }//end Conjunctive Rule
    
    //! Disjunctive Rule
    void Combine_Disj(const cv::Mat &bba, cv::Mat &bba_cmb)
    {
        int nbr_src = bba.rows;
        int nbr_elem = bba.cols;
        
        cv::Mat b(nbr_src, nbr_elem, CV_64F), temp;
        for(int i = 0; i < nbr_src; i++) {
            cv::Mat m = bba.row(i);
            MtoB(m, temp);
            temp.copyTo(b.row(i));
        }
        
        for(int i = 0; i < (int)nbr_src - 1; i++) {
            cv::multiply(b.row(i), b.row(i+1), b.row(i+1));
        }
        
        cv::Mat b_cmb = b.row(nbr_src-1);
        BtoM(b_cmb, bba_cmb);
    }//end Disjunctive Rule

    
    //! Yager's Rule
    void Combine_Yager(const cv::Mat &bba, cv::Mat &bba_cmb)
    {
        Combine_Conj(bba, bba_cmb);
        
        int nbr_elem = bba.cols;
        bba_cmb.at<double>(0,nbr_elem-1) = bba_cmb.at<double>(0,nbr_elem-1) + bba_cmb.at<double>(0,0);
        bba_cmb.at<double>(0,0) = 0;
    }//end Yager's Rule
    
    //! Dubois & Prade's Rule
    void Combine_DuboisPrade(const cv::Mat &bba, cv::Mat &bba_cmb)
    {
        int nbr_elem = bba.cols;
        
        cv::Mat bba_conj, bba_disj;
        Combine_Conj(bba, bba_conj);
        Combine_Disj(bba, bba_disj);
//        PrintMat(bba_conj);
//        PrintMat(bba_disj);
        
        double conflict = bba_conj.at<double>(0,0);
        if(conflict != 0) {
            std::vector<int> pos;
            for(int i = 0; i < nbr_elem; i++)
            {
                if(bba_disj.at<double>(0,i) != 0)
                    pos.push_back(i);
            }
            
            double sum_c = 0;
            for(int i = 0; i < (int)pos.size(); i++)
            {
                if(pos[i] != nbr_elem - 1)
                    sum_c += bba_disj.at<double>(0, pos[i]);
            }
            
            if(sum_c <= conflict)
            {
                for(int i = 0; i < (int)pos.size(); i++)
                {
                    if(pos[i]!= nbr_elem - 1)
                        bba_conj.at<double>(0, pos[i]) += bba_disj.at<double>(0, pos[i]);
                }
                bba_conj.at<double>(0, nbr_elem-1) = conflict - sum_c;
            }
            else 
            {
                bba_conj.at<double>(0, pos[0]) += conflict;
            }
            bba_conj.at<double>(0,0) = 0;
        }
        bba_cmb = bba_conj;
        
        cv::Scalar m = cv::sum(bba_cmb);
        if(m[0] < 0.99999999) 
        {
            std::cout << "sum : " << m[0] << std::endl;
            cv::waitKey();
        }
    }//end Dubois & Prade Combination Rule
    
    
    
    //! Dubois & Prade's Rule in Condition of Non-distinct
    void Combine_DPNonDistinct(const cv::Mat &bba, cv::Mat &bba_cmb)
    {
        int nbr_elem = bba.cols;
        
        cv::Mat bba_conj, bba_disj;
        Combine_CautiousConj(bba, bba_conj);
        Combine_BoldDisj(bba, bba_disj);
        
        double conflict = bba_conj.at<double>(0,0);
        if(conflict != 0) {
            std::vector<int> pos;
            for(int i = 0; i < nbr_elem; i++)
            {
                if(bba_disj.at<double>(0,i) > 1.0e-5) //instead of value 0
                    pos.push_back(i);
            }
            
            double sum_c = 0;
            for(int i = 0; i < (int)pos.size(); i++)
            {
                if(pos[i] != nbr_elem - 1)
                    sum_c += bba_disj.at<double>(0, pos[i]);
            }
            
            if(sum_c <= conflict)
            {
                for(int i = 0; i < (int)pos.size(); i++)
                {
                    if(pos[i]!= nbr_elem - 1)
                        bba_conj.at<double>(0, pos[i]) += bba_disj.at<double>(0, pos[i]);
                }
                bba_conj.at<double>(0, nbr_elem-1) = conflict - sum_c;
            }
            else 
            {
                bba_conj.at<double>(0, pos[0]) += conflict;
            }
            bba_conj.at<double>(0,0) = 0;
        }
        bba_cmb = bba_conj;
        
        cv::Scalar m = cv::sum(bba_cmb);
        if(m[0] < 0.99999999) 
        {
            std::cout << "sum : " << m[0] << std::endl;
            cv::waitKey();
        }  
    }//end Combine_DPNonDistinct
    
    //! Dempster's Rule
    void Combine_Dempster(const cv::Mat &bba, cv::Mat &bba_cmb)
    {
        Combine_Conj(bba, bba_cmb);

		if(bba_cmb.at<double>(0,0) != 1)
        	NormalizeBBA(bba_cmb, bba_cmb);
    }//end Dempster's Rule
    
       
    
    //! Cautious Conjunctive Rule
    void Combine_CautiousConj(const cv::Mat &bba, cv::Mat &bba_cmb)
    {
        int nbr_src = bba.rows;
        int nbr_elem = bba.cols;
        double epsilon = 1.0e-10;
        
        cv::Mat w(nbr_src, nbr_elem, CV_64F), temp;
        for(int i = 0; i < nbr_src; i++) {
            cv::Mat m = bba.row(i);
            if(m.at<double>(0, nbr_elem -1) == 0) {
                int sum = 0;
                for(int i = 0; i < nbr_elem; i++) {
                    if(m.at<double>(0,i) != 0) {
                        m.at<double>(0,i) = m.at<double>(0,i) - epsilon;
                        sum++;
                    }
                }
                m.at<double>(0, nbr_elem-1) = epsilon*sum;
            }
            MtoQ(m, temp);
            QtoW(temp, temp);
            temp.copyTo(w.row(i));
        }
        //! Get the weight after combination
        cv::Mat w_cmb(1, nbr_elem, CV_64F);
        for(int i = 0; i < nbr_elem; i++) {
            double temp = w.at<double>(0,i);
            for(int j = 0; j < nbr_src-1; j++) {
                temp = temp < w.at<double>(j+1,i)? temp : w.at<double>(j+1,i);
            }
            w_cmb.at<double>(0,i) = temp; 
        }
        //PrintMat(w_cmb);
        
        int nbr_w_cmb = w_cmb.cols - 1;
        
        cv::Mat q(nbr_w_cmb, nbr_elem, CV_64F);
        for(int i = 0; i < nbr_w_cmb; i++) {
            cv::Mat m = cv::Mat::zeros(1, nbr_elem, CV_64F);
            m.at<double>(0,i) = 1 - w_cmb.at<double>(0, i);
            m.at<double>(0,nbr_elem - 1) = w_cmb.at<double>(0, i);
            cv::Mat q_i;
            MtoQ(m, q_i);
            q_i.copyTo(q.row(i));
        }
        
        for(int i = 0; i < (int)nbr_w_cmb - 1; i++) {
            cv::multiply(q.row(i), q.row(i+1), q.row(i+1));
        }
        
        cv::Mat q_cmb = q.row(nbr_w_cmb-1);
        QtoM(q_cmb, bba_cmb);
    }
    
    //! Cautious Conjunctive Normalize Rule
    void Combine_CautiousConjNorm(const cv::Mat &bba, cv::Mat &bba_cmb)
    {
        Combine_CautiousConj(bba, bba_cmb);
        NormalizeBBA(bba_cmb, bba_cmb);
    }
    
    //! Bold Disjunctive Rule
    void Combine_BoldDisj(const cv::Mat &bba, cv::Mat &bba_cmb)
    {
        int nbr_src = bba.rows;
        int nbr_elem = bba.cols;
        double epsilon = 1.0e-10;
        
        cv::Mat v(nbr_src, nbr_elem, CV_64F), temp;
        for(int i = 0; i < nbr_src; i++) {
            cv::Mat m = bba.row(i);
            cv::flip(m, m, 1);//hor
            
            if(m.at<double>(0, nbr_elem -1) == 0) {
                int sum = 0;
                for(int i = 0; i < nbr_elem; i++) {
                    if(m.at<double>(0,i) != 0) {
                        m.at<double>(0,i) = m.at<double>(0,i) - epsilon;
                        sum++;
                    }
                }
                m.at<double>(0, nbr_elem-1) = epsilon*sum;
            }
            
            MtoQ(m, temp);
            QtoW(temp, temp);
            temp.copyTo(v.row(i));
        }
    
        cv::flip(v, v, 1);//hor reverse
        v.col(0) = 1;
//        PrintMat(v);
        
        //! Get the weight after combination
        cv::Mat v_cmb(1, nbr_elem, CV_64F);
        for(int i = 0; i < nbr_elem; i++) {
            double temp = v.at<double>(0,i);
            for(int j = 0; j < nbr_src-1; j++) {
                temp = temp < v.at<double>(j+1,i)? temp : v.at<double>(j+1,i);
            }
            v_cmb.at<double>(0,i) = temp; 
        }
//        PrintMat(v_cmb);
        
        
        cv::Mat b(nbr_elem, nbr_elem, CV_64F);
        for(int i = 0; i < nbr_elem; i++) {
            cv::Mat m = cv::Mat::zeros(1, nbr_elem, CV_64F);
            m.at<double>(0,0) = v_cmb.at<double>(0, i);
            
            if( i != 0) {
                m.at<double>(0, i) = 1 - m.at<double>(0,0);
            }
            //PrintMat(m);

            cv::Mat b_i;
            MtoB(m, b_i);
            b_i.copyTo(b.row(i));
        }
        
//        PrintMat(b);
        
        for(int i = 0; i < (int)nbr_elem - 1; i++) {
            cv::multiply(b.row(i), b.row(i+1), b.row(i+1));
        }
        
        cv::Mat b_cmb = b.row(nbr_elem-1);
        BtoM(b_cmb, bba_cmb);

    }//end Combine_BoldDisj
    
    
}//namespace
