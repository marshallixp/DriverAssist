//
//  FusionCenter.cpp
//  FusionCenter1.2
//
//  Created by XUANPENG LI on 11/11/13.
//  Copyright (c) 2013 __MyCompanyName__. All rights reserved.
//
#include "FusionCenter.h"

#include <iostream>
#include <stdio.h>
#include <string.h>

extern const double SAMPLING_TIME;
extern const int    DEBUG_SHOW;

using namespace DST;
namespace FusionCenter {
    //! Single Level with ORD Model
    void FusionCenter_SingleORD(const LaneFeature *laneFeatures,
                          const FaceFeature *faceFeatures,
                          const float &ccpThreshold_ST,
                          const float &mstThreshold_ST,
                          const int &tagRule_LT, const int &tagRule_ST,
                          const int &tagDecision,
                          const double &intervalTime,
                          std::deque<std::vector<double> > &resultsSeq,
                          double &drowsyLevel, 
                          std::vector<double> &results_ST, 
                          std::vector<double> &conflict, cv::Mat &debugShow, 
                          double &mLATSD, double &mPERCLOS, double &mBLINK)
    {
        conflict.clear();
        const int nbr_hyp = 5;//ND, SD, MD, VD, ED
        
        int Model_LT = CONFIDENCE;
        
/************************************************/
//! Long-term Variables_SingleORD
/************************************************/
        const int nbr_lane = 1;//LATSD
        const int nbr_face = 2;//PERCLOS, BLINK
        
        cv::Mat bba_lane = cv::Mat::zeros(nbr_lane, (int)pow(2.0, nbr_hyp), CV_64F);
        cv::Mat bba_face = cv::Mat::zeros(nbr_face, (int)pow(2.0, nbr_hyp), CV_64F);
        
        
        Model_ORD_Lane(laneFeatures, Model_LT, nbr_hyp, bba_lane, mLATSD);
        Model_ORD_Face(faceFeatures, Model_LT, nbr_hyp, bba_face, mPERCLOS, mBLINK);
        
        const int nbr_unit_LT = nbr_lane * nbr_face; 
        cv::Mat bba_unit_LT[nbr_unit_LT];
        for(int i = 0; i < nbr_unit_LT; i++){
            bba_unit_LT[i] = cv::Mat::zeros(2, (int)pow(2.0, nbr_hyp), CV_64F);
        }
        for(int i = 0; i < nbr_lane; i++)
        {
            for(int j = 0; j < nbr_face; j++)
            {
                int k = (i+1)*(j+1)-1;
                bba_lane.row(i).copyTo(bba_unit_LT[k].row(0));
                bba_face.row(j).copyTo(bba_unit_LT[k].row(1));
            }
        }
        
        //! Choose the Combination Rule
        cv::Mat bba_unit_LT_cmb[nbr_unit_LT];
        for(int i = 0; i < nbr_unit_LT; i++){
            bba_unit_LT_cmb[i] = cv::Mat::zeros(1, (int)pow(2.0, nbr_hyp), CV_64F);
        }
        switch (tagRule_LT) {
            case Dempster:
                //std::cout << "Dempster" << std::endl;
                for (int i = 0; i < nbr_unit_LT; i++) {
                    Combine_Dempster(bba_unit_LT[i], bba_unit_LT_cmb[i]);
                }
                
                break;
                
            case Conjunctive:
                //std::cout << "Conjunctive" << std::endl;
                for (int i = 0; i < nbr_unit_LT; i++) {
                    Combine_Conj(bba_unit_LT[i], bba_unit_LT_cmb[i]);
                }
                break;
                
            case Disjunctive:
                //std::cout << "Disjunctive" << std::endl;
                for (int i = 0; i < nbr_unit_LT; i++) {
                    Combine_Disj(bba_unit_LT[i], bba_unit_LT_cmb[i]);
                }
                break;
                
            case CautiousConj:
                //std::cout << "CautiousConj" << std::endl;
                for (int i = 0; i < nbr_unit_LT; i++) {
                    Combine_CautiousConj(bba_unit_LT[i], bba_unit_LT_cmb[i]);
                }
                break;
                
            case CautiousConjNorm:
                //std::cout << "CautiousConjNorm" << std::endl;
                for (int i = 0; i < nbr_unit_LT; i++) {
                    Combine_CautiousConjNorm(bba_unit_LT[i], bba_unit_LT_cmb[i]);
                }
                break;
                
            case Yager:
                //std::cout << "Yager" << std::endl;
                for (int i = 0; i < nbr_unit_LT; i++) {
                    Combine_Yager(bba_unit_LT[i], bba_unit_LT_cmb[i]);
                }
                break;
                
            case DuboisPrade:
                for (int i = 0; i < nbr_unit_LT; i++) {
                    Combine_DuboisPrade(bba_unit_LT[i], bba_unit_LT_cmb[i]);
                }
                break;
                
            default:
                break;
        }
        
        
//        PrintMat(bba_unit_LT[0]);
//        PrintMat(bba_unit_LT_cmb[0]);
        
        //! Make Decision 
        std::vector<double> results;
//        MakeDecision_ORD(bba_LT_cmb, tagDecision, results);
//        conflict.push_back(bba_LT_cmb.at<double>(0,0)); //empty set
        
        MakeDecision_Single_ORD(bba_unit_LT_cmb, nbr_unit_LT, results);
        
        conflict.push_back(0);//Dempster
    
        //! 60-s Moving Weighted Mean Algorithm
        resultsSeq.push_back(results);
        double sumTime = 0;
        for(int i = 0; i < (int)resultsSeq.size(); i++)
        {
            sumTime += resultsSeq[i].back();//intervalTime
        }
        while (sumTime > SAMPLING_TIME) {
            sumTime -= resultsSeq.front().back();
            resultsSeq.pop_front();
        }
        
        drowsyLevel = 0; //reset 
        for(int i = 0; i < (int)resultsSeq.size(); i++)
        {
            drowsyLevel += resultsSeq[i].at(5);
        }
        drowsyLevel /= (double)resultsSeq.size();
        
        //! Show debug info
        if(DEBUG_SHOW) {
            char *text = new char[100];
    
            sprintf(text, "L: %.3f, P: %.3f, B: %.3f", mLATSD, mPERCLOS, mBLINK);
            cv::putText(debugShow, text, cv::Point(0, 40), cv::FONT_HERSHEY_SIMPLEX, 0.3, CV_RGB(0,255,0));
            
            // rows <- nbr_unit_LT
            for(int i = 0; i < nbr_unit_LT; i++) {
                for(int j = 0; j < (int)bba_unit_LT_cmb[i].cols; j++) {
                    sprintf(text, "%.3f", bba_unit_LT_cmb[i].at<double>(0,j));
                    cv::putText(debugShow, text, cv::Point(j*30, i*20+50), cv::FONT_HERSHEY_SIMPLEX, 0.3, CV_RGB(0,255,0));
                }
            }
            
            sprintf(text, "Combined Results: ND, SD, MD, ED, VD, DrowsyLevel, IntervalTime");
            cv::putText(debugShow, text, cv::Point(0, 110), cv::FONT_HERSHEY_SIMPLEX, 0.3, CV_RGB(0,255,0));
            for(int k = 0; k < (int)results.size(); k++) {
                sprintf(text, "%.3f", results[k]);
                cv::putText(debugShow, text, cv::Point(k*30, 130), cv::FONT_HERSHEY_SIMPLEX, 0.3, CV_RGB(0,255,0));
            }
            
            sprintf(text, "DrowsyLevel: %.3f", drowsyLevel);
            cv::putText(debugShow, text, cv::Point(0, 150), cv::FONT_HERSHEY_SIMPLEX, 0.3, CV_RGB(0,255,0));
            
            delete text;
        }
        
/************************************************/
//! Short-term Variables_SingleORD
/************************************************/
        const int nbr_lane_ST = 1;
        const int nbr_face_ST = 1;
        const int nbr_hyp_ST = 2; //??? Consider impared driving {D, ~D}
    
        cv::Mat bba_lane_ST = cv::Mat::zeros(nbr_lane_ST, (int)pow(2.0, nbr_hyp_ST), CV_64F);
        cv::Mat bba_face_ST = cv::Mat::zeros(nbr_face_ST, (int)pow(2.0, nbr_hyp_ST), CV_64F);
        
        double mLateralOffset, mMicroSleep;
        Model_SingleORD_ST(laneFeatures, faceFeatures, 
                           ccpThreshold_ST, mstThreshold_ST,
                           bba_lane_ST, bba_face_ST, mLateralOffset, mMicroSleep);
       
        
        const int nbr_ST = nbr_lane_ST + nbr_face_ST;
        cv::Mat bba_ST = cv::Mat::zeros(nbr_ST, (int)pow(2.0, nbr_hyp_ST), CV_64F);
        bba_lane_ST.copyTo(bba_ST.row(0));
        bba_face_ST.copyTo(bba_ST.row(1));
        
        //! Choose the Combination Rule
        cv::Mat bba_ST_cmb(1, (int)pow(2.0, nbr_hyp_ST), CV_64F);
        switch (tagRule_ST) {
            case Dempster:
                Combine_Dempster(bba_ST, bba_ST_cmb);
                break;
                
            case Conjunctive:
                Combine_Conj(bba_ST, bba_ST_cmb);
                break;
                
            case Disjunctive:
                Combine_Disj(bba_ST, bba_ST_cmb);
                break;
                
            case CautiousConj:
                Combine_CautiousConj(bba_ST, bba_ST_cmb);
                break;
                
            case CautiousConjNorm:
                Combine_CautiousConjNorm(bba_ST, bba_ST_cmb);
                break;
                
            case Yager:
                Combine_Yager(bba_ST, bba_ST_cmb);
                break;
                
            case DuboisPrade:
                Combine_DuboisPrade(bba_ST, bba_ST_cmb);
                break;
                
            default:
                break;
        }//end switch
        
        results_ST.clear();
        results_ST.push_back(bba_ST_cmb.at<double>(0,1));
        results_ST.push_back(bba_ST_cmb.at<double>(0,2));
        results_ST.push_back(bba_ST_cmb.at<double>(0,3));
        
        conflict.push_back(bba_ST_cmb.at<double>(0,0));
        
        if(DEBUG_SHOW) {
            char *text = new char[100];
            
            sprintf(text, "LO: %.3f, MS: %.3f", mLateralOffset, mMicroSleep);
            cv::putText(debugShow, text, cv::Point(0, 170), cv::FONT_HERSHEY_SIMPLEX, 0.3, CV_RGB(0,255,0));
            
            for(int i = 0; i < (int)bba_ST.rows; i++) {
                for(int j = 0; j < (int)bba_ST.cols; j++) {
                    sprintf(text, "%.3f", bba_ST.at<double>(i,j));
                    cv::putText(debugShow, text, cv::Point(j*30, i*20+180), cv::FONT_HERSHEY_SIMPLEX, 0.3, CV_RGB(0,255,0));
                }
            }
            
            for(int j = 0; j < (int)bba_ST_cmb.cols; j++) {
                sprintf(text, "%.3f", bba_ST_cmb.at<double>(0,j));
                cv::putText(debugShow, text, cv::Point(j*30, 260), cv::FONT_HERSHEY_SIMPLEX, 0.3, CV_RGB(0,255,0));
            }
            delete text;
        }
    }//end FusionCenter_ORD
    
    
    void FusionCenter_MultiFusion(const LaneFeature *laneFeatures, 
                                  const FaceFeature *faceFeatures, 
                                  const int &tagRule_LT1, const int &tagRule_LT2,
                                  const int &tagDecision_LT,
                                  const int &tagRule_ST, 
                                  const double &intervalTime, std::vector<double> &results,
                                  double &mLATSD, double &mPERCLOS, double &mBLINK,
                                  double &T, double &TuF, double &F, cv::Mat &debugShow)
    {
        const int nbr_hyp_LT = 5;//ND, SD, MD, VD, ED
        
        int Model_LT = TRIANGULAR;
        int Model_ST = CONFIDENCE;
        
        /************************************************/
        //! Long-term Fusion in the 1st Level
        /************************************************/
        const int nbr_lane = 1;//LATSD, 
        const int nbr_face = 2;//PERCLOS, BLINK
        
        cv::Mat bba_lane = cv::Mat::zeros(nbr_lane, (int)pow(2.0, nbr_hyp_LT), CV_64F);
        cv::Mat bba_face = cv::Mat::zeros(nbr_face, (int)pow(2.0, nbr_hyp_LT), CV_64F);
        
        Model_ORD_Lane(laneFeatures, Model_LT, nbr_hyp_LT, bba_lane, mLATSD);
        Model_ORD_Face(faceFeatures, Model_LT, nbr_hyp_LT, bba_face, mPERCLOS, mBLINK);
        
        //! Choose the Combination Rule
        cv::Mat bba_lane_cmb(1, (int)pow(2.0, nbr_hyp_LT), CV_64F);
        cv::Mat bba_face_cmb(1, (int)pow(2.0, nbr_hyp_LT), CV_64F);
        
        switch (tagRule_LT1) {             
            case CautiousConj:
                Combine_CautiousConj(bba_lane, bba_lane_cmb);
                Combine_CautiousConj(bba_face, bba_face_cmb);
                break;
                
            case CautiousConjNorm:
                Combine_CautiousConjNorm(bba_lane, bba_lane_cmb);
                Combine_CautiousConjNorm(bba_face, bba_face_cmb);
                break;
                
            default:
                break;
        }//end switch
        
        /************************************************/
        //! Lane and Face Fusion in the 2nd Level
        /************************************************/
        const int nbr_LT = 2; //lane, face
        cv::Mat bba_LT(nbr_LT, (int)pow(2.0, nbr_hyp_LT), CV_64F);
        bba_lane_cmb.copyTo(bba_LT.row(0));
        bba_face_cmb.copyTo(bba_LT.row(1));
        
        //! Choose the Combination Rule
        cv::Mat bba_LT_cmb(1, (int)pow(2.0, nbr_hyp_LT), CV_64F);
        switch (tagRule_LT2) {
            case Dempster:
                Combine_Dempster(bba_LT, bba_LT_cmb);
                break;
            case Conjunctive:
                Combine_Conj(bba_LT, bba_LT_cmb);
                break;
            case Disjunctive:
                Combine_Disj(bba_LT, bba_LT_cmb);
                break;   
            case CautiousConj:
                Combine_CautiousConj(bba_LT, bba_LT_cmb);
                break;  
            case CautiousConjNorm:
                Combine_CautiousConjNorm(bba_LT, bba_LT_cmb);
                break;  
            case Yager:
                Combine_Yager(bba_LT, bba_LT_cmb);
                break;  
            case DuboisPrade:
                Combine_DuboisPrade(bba_LT, bba_LT_cmb);
                break;   
            default:
                break;
        }//end switch
        
        results.clear();//ND,SD,MD,VD,ED,Drowsylevel
        MakeDecision_ORD(bba_LT_cmb, tagDecision_LT, results);
        
        
        /************************************************/
        //! Short-term Variables in MultiFusion
        /************************************************/
        const int nbr_lane_ST = 1;//lateralOffset
        const int nbr_face_ST = 1;//MicroSleep
        const int nbr_hyp_ST = 2; //impared driving {SAF, DAN}
        
        cv::Mat bba_lane_ST = cv::Mat::zeros(nbr_lane_ST, (int)pow(2.0, nbr_hyp_ST), CV_64F);
        cv::Mat bba_face_ST = cv::Mat::zeros(nbr_face_ST, (int)pow(2.0, nbr_hyp_ST), CV_64F);
        
        double mLateralOffset, mMicroSleep;
        Model_MultiORD_ST(laneFeatures, faceFeatures, Model_ST,
                          bba_lane_ST, bba_face_ST, mLateralOffset, mMicroSleep);
        
        
        const int nbr_ST = 2;
        cv::Mat bba_ST = cv::Mat::zeros(nbr_ST, (int)pow(2.0, nbr_hyp_ST), CV_64F);
        bba_lane_ST.copyTo(bba_ST.row(0));
        bba_face_ST.copyTo(bba_ST.row(1));
        
        //! Choose the Combination Rule
        cv::Mat bba_ST_cmb(1, (int)pow(2.0, nbr_hyp_ST), CV_64F);
        switch (tagRule_ST) {
            case Dempster:
                Combine_Dempster(bba_ST, bba_ST_cmb);
                break;
            case Conjunctive:
                Combine_Conj(bba_ST, bba_ST_cmb);
                break;
            case Disjunctive:
                Combine_Disj(bba_ST, bba_ST_cmb);
                break;
            case CautiousConj:
                Combine_CautiousConj(bba_ST, bba_ST_cmb);
                break;
            case CautiousConjNorm:
                Combine_CautiousConjNorm(bba_ST, bba_ST_cmb);
                break;
            case Yager:
                Combine_Yager(bba_ST, bba_ST_cmb);
                break; 
            case DuboisPrade:
                Combine_DuboisPrade(bba_ST, bba_ST_cmb);
                break;  
            default:
                break;
        }//end switch
        
        /************************************************/
        //! Fusion of LT and ST
        //! Final Decision: {T, F, TuF}
        // ----------------------------------------
        //      | ND | SD | MD | VD | ED |
        // ----------------------------------------
        //  SAF | T  | T  | TuF|  F |  F |
        // ----------------------------------------
        //  UNK | T  | T  | TuF|  F |  F |
        // ----------------------------------------
        //  DAN | TuF| TuF| TuF|  F |  F |
        // ----------------------------------------
        /************************************************/
        //! Empty == 0
        //! T
        double T_ND_SAF = results[0] * bba_ST_cmb.at<double>(0,1);//-I
        double T_ND_UNK = results[0] * bba_ST_cmb.at<double>(0,3);//ALL
        double T_SD_SAF = results[1] * bba_ST_cmb.at<double>(0,1);
        double T_SD_UNK = results[1] * bba_ST_cmb.at<double>(0,3);
        T = T_ND_SAF + T_ND_UNK + T_SD_SAF + T_SD_UNK;
        //! TuF
        double TuF_ND_DAN = results[0] * bba_ST_cmb.at<double>(0,2);
        double TuF_SD_DAN = results[1] * bba_ST_cmb.at<double>(0,2);
        double TuF_MD_SAF = results[2] * bba_ST_cmb.at<double>(0,1);
        double TuF_MD_UNK = results[2] * bba_ST_cmb.at<double>(0,3);
        double TuF_MD_DAN = results[2] * bba_ST_cmb.at<double>(0,2);
        TuF = TuF_ND_DAN + TuF_SD_DAN + TuF_MD_SAF + TuF_MD_UNK + TuF_MD_DAN;
        //! F
        double F_VD_SAF = results[3] * bba_ST_cmb.at<double>(0,1);
        double F_VD_UNK = results[3] * bba_ST_cmb.at<double>(0,3);
        double F_VD_DAN = results[3] * bba_ST_cmb.at<double>(0,2);
        double F_ED_SAF = results[4] * bba_ST_cmb.at<double>(0,1);
        double F_ED_UNK = results[4] * bba_ST_cmb.at<double>(0,3);
        double F_ED_DAN = results[4] * bba_ST_cmb.at<double>(0,2);
        F = F_VD_SAF + F_VD_UNK + F_VD_DAN + F_ED_SAF + F_ED_UNK + F_ED_DAN;
        
        //! Confirm
        double total = T + TuF + F;
        //std::cout << "Total Value: " << total << std::endl;
        //assert(total > 0.99999999999999);
        
        /************************************************/
        //! Show results 
        /************************************************/
        if(DEBUG_SHOW) {
            char *text = new char[100];
            //! Final Results
            sprintf(text, "Final Results >> T: %.3f, TuF: %.3f, F: %.3f, Total: %.3f", T, TuF, F, total);
            cv::putText(debugShow, text, cv::Point(0, 40), cv::FONT_HERSHEY_SIMPLEX, 0.4, CV_RGB(0,255,0));
            
            //! Indicators
            sprintf(text, "LongTerm >> L: %.3f, P: %.3f, B: %.3f", mLATSD, mPERCLOS, mBLINK);
            cv::putText(debugShow, text, cv::Point(0, 70), cv::FONT_HERSHEY_SIMPLEX, 0.3, CV_RGB(0,255,0));
            sprintf(text, "ShortTerm >> LO: %.3f, MS: %.3f", mLateralOffset, mMicroSleep);
            cv::putText(debugShow, text, cv::Point(0, 90), cv::FONT_HERSHEY_SIMPLEX, 0.3, CV_RGB(0,255,0));
            
            //! Long-term Results
            sprintf(text, "LT Resutls >> ND: %.3f, SD: %.3f, MD: %.3f, VD: %.3f, ED: %.3f, Drowsy: %.3f", results[0], results[1], results[2], results[3], results[4], results[5]);
            cv::putText(debugShow, text, cv::Point(0, 110), cv::FONT_HERSHEY_SIMPLEX, 0.3, CV_RGB(0,255,0));
            //! Short-term Results
            sprintf(text, "ST Results >> SAF: %.3f, DAN: %.3f, UNK: %.3f", bba_ST_cmb.at<double>(0,1), bba_ST_cmb.at<double>(0,2), bba_ST_cmb.at<double>(0,3));
            cv::putText(debugShow, text, cv::Point(0, 130), cv::FONT_HERSHEY_SIMPLEX, 0.3, CV_RGB(0,255,0));
            
            delete text;
        }
    }
    
    
    void FusionCenter_MultiORD(const LaneFeature *laneFeatures, 
                               const FaceFeature *faceFeatures, 
                               const int &tagRule_LT1, const int &tagRule_LT2, 
                               const int &tagRule_ST, const int &tagDecision,
                               const double &intervalTime,
                               std::deque<std::vector<double> > &resultsSeq, 
                               double &drowsyLevel, 
                               std::vector<double> &results_ST, 
                               std::vector<double> &conflict, cv::Mat &debugShow, 
                               double &mLATSD, double &mPERCLOS, double &mBLINK)
    {
        conflict.clear();
        const int nbr_hyp = 5;//ND, SD, MD, VD, ED
        
        int Model_LT = CONFIDENCE;
        int Model_ST = CONFIDENCE;
        
/************************************************/
//! Long-term Variables Fusion in the 1st Level
/************************************************/
        const int nbr_lane = 1;//LATSD, 
        const int nbr_face = 2;//PERCLOS, BLINK
        
        cv::Mat bba_lane = cv::Mat::zeros(nbr_lane, (int)pow(2.0, nbr_hyp), CV_64F);
        cv::Mat bba_face = cv::Mat::zeros(nbr_face, (int)pow(2.0, nbr_hyp), CV_64F);
        
        Model_ORD_Lane(laneFeatures, Model_LT, nbr_hyp, bba_lane, mLATSD);
        Model_ORD_Face(faceFeatures, Model_LT, nbr_hyp, bba_face, mPERCLOS, mBLINK);
        
        //! Choose the Combination Rule
        cv::Mat bba_lane_cmb(1, (int)pow(2.0, nbr_hyp), CV_64F);
        cv::Mat bba_face_cmb(1, (int)pow(2.0, nbr_hyp), CV_64F);
        switch (tagRule_LT1) {
            case Dempster:
                Combine_Dempster(bba_lane, bba_lane_cmb);
                Combine_Dempster(bba_face, bba_face_cmb);
                break;
                
            case Conjunctive:
                Combine_Conj(bba_lane, bba_lane_cmb);
                Combine_Conj(bba_face, bba_face_cmb);
                break;
                
            case Disjunctive:
                Combine_Disj(bba_lane, bba_lane_cmb);
                Combine_Disj(bba_face, bba_face_cmb);
                break;
                
            case CautiousConj:
                Combine_CautiousConj(bba_lane, bba_lane_cmb);
                Combine_CautiousConj(bba_face, bba_face_cmb);
                break;
                
            case CautiousConjNorm:
                Combine_CautiousConjNorm(bba_lane, bba_lane_cmb);
                Combine_CautiousConjNorm(bba_face, bba_face_cmb);
                break;
                
            case Yager:
                Combine_Yager(bba_lane, bba_lane_cmb);
                Combine_Yager(bba_face, bba_face_cmb);
                break;
                
            case DuboisPrade:
                Combine_DuboisPrade(bba_lane, bba_lane_cmb);
                Combine_DuboisPrade(bba_face, bba_face_cmb);
                break;
                
            case DuboisPradeNonDistinct:
                Combine_DPNonDistinct(bba_lane, bba_lane_cmb);
                Combine_DPNonDistinct(bba_face, bba_face_cmb);
                break;
                
            default:
                break;
        }//end switch
        
        //! Optimization of the 1st level 
        //! Consider Dubois&Prade's Rule instead of Yager's rule 
//        double conflict_lane = bba_lane_cmb.at<double>(0, 0);
//        double conflict_face = bba_face_cmb.at<double>(0, 0);
//        conflict.push_back(conflict_lane);
//        conflict.push_back(conflict_face);
//        
//        bba_lane_cmb.at<double>(0, (int)pow(2.0, nbr_hyp)-1) += conflict_lane;
//        bba_face_cmb.at<double>(0, (int)pow(2.0, nbr_hyp)-1) += conflict_face;
//        bba_lane_cmb.at<double>(0, 0) = 0;
//        bba_face_cmb.at<double>(0, 0) = 0;
        
//        PrintMat(bba_face_cmb);
//        PrintMat(bba_lane_cmb);
        
/************************************************/
//! Lane and Face Fusion in the 2nd Level
/************************************************/
        const int nbr_LT = 2;
        cv::Mat bba_LT(nbr_LT, (int)pow(2.0, nbr_hyp), CV_64F);
        bba_lane_cmb.copyTo(bba_LT.row(0));
        bba_face_cmb.copyTo(bba_LT.row(1));
        
        //! Choose the Combination Rule
        cv::Mat bba_LT_cmb(1, (int)pow(2.0, nbr_hyp), CV_64F);
        switch (tagRule_LT2) {
            case Dempster:
                Combine_Dempster(bba_LT, bba_LT_cmb);
                break;
            case Conjunctive:
                Combine_Conj(bba_LT, bba_LT_cmb);
                break;
            case Disjunctive:
                Combine_Disj(bba_LT, bba_LT_cmb);
                break;   
            case CautiousConj:
                Combine_CautiousConj(bba_LT, bba_LT_cmb);
                break;  
            case CautiousConjNorm:
                Combine_CautiousConjNorm(bba_LT, bba_LT_cmb);
                break;  
            case Yager:
                Combine_Yager(bba_LT, bba_LT_cmb);
                break;  
            case DuboisPrade:
                Combine_DuboisPrade(bba_LT, bba_LT_cmb);
                break;   
            default:
                break;
        }//end switch
        
        //! Make Decision 
        std::vector<double> results;
        MakeDecision_ORD(bba_LT_cmb, tagDecision, results);
        
        
        //! Conflict in Long-Term
        conflict.push_back(bba_LT_cmb.at<double>(0,0)); //empty set
        
        //! 60-s Moving Weighted Mean Algorithm
        resultsSeq.push_back(results);
        double sumTime = 0;
        for(int i = 0; i < (int)resultsSeq.size(); i++)
        {
            sumTime += resultsSeq[i].back();//intervalTime
        }
        while (sumTime > SAMPLING_TIME) {
            sumTime -= resultsSeq.front().back();
            resultsSeq.pop_front();
        }
        
        drowsyLevel = 0; //reset 
        for(int i = 0; i < (int)resultsSeq.size(); i++)
        {
            drowsyLevel += resultsSeq[i].at(5);
        }
        drowsyLevel /= (double)resultsSeq.size();
        
        //! Show debug info
        if(DEBUG_SHOW) {
            char *text = new char[50];
            
            sprintf(text, "LongTerm >> L: %.3f, P: %.3f, B: %.3f", mLATSD, mPERCLOS, mBLINK);
            cv::putText(debugShow, text, cv::Point(0, 40), cv::FONT_HERSHEY_SIMPLEX, 0.3, CV_RGB(0,255,0));
            
            for(int i = 0; i < (int)bba_LT.rows; i++) {
                for(int j = 0; j < (int)bba_LT.cols; j++) {
                    sprintf(text, "%.3f", bba_LT.at<double>(i,j));
                    cv::putText(debugShow, text, cv::Point(j*30, i*20+50), cv::FONT_HERSHEY_SIMPLEX, 0.3, CV_RGB(0,255,0));
                }
            }
            
            sprintf(text, "Combined Results: ND, SD, MD, ED, VD");
            cv::putText(debugShow, text, cv::Point(0, 110), cv::FONT_HERSHEY_SIMPLEX, 0.3, CV_RGB(0,255,0));
            
            for(int j = 0; j < (int)bba_LT_cmb.cols; j++) {
                sprintf(text, "%.3f", bba_LT_cmb.at<double>(0,j));
                cv::putText(debugShow, text, cv::Point(j*30, 130), cv::FONT_HERSHEY_SIMPLEX, 0.7, CV_RGB(0,255,0));
            }
        
            sprintf(text, "DrowsyLevel: %.3f", drowsyLevel);
            cv::putText(debugShow, text, cv::Point(0, 150), cv::FONT_HERSHEY_SIMPLEX, 0.3, CV_RGB(0,255,0));
            
            delete text;
        }
        
/************************************************/
//! Short-term Variables in Multi-ORD
/************************************************/
        const int nbr_lane_ST = 1;//lateralOffset
        const int nbr_face_ST = 1;//MicroSleep
        const int nbr_hyp_ST = 2; //??? Consider impared driving {D, ~D}
        
        cv::Mat bba_lane_ST = cv::Mat::zeros(nbr_lane_ST, (int)pow(2.0, nbr_hyp_ST), CV_64F);
        cv::Mat bba_face_ST = cv::Mat::zeros(nbr_face_ST, (int)pow(2.0, nbr_hyp_ST), CV_64F);
        
        double mLateralOffset, mMicroSleep;
        Model_MultiORD_ST(laneFeatures, faceFeatures, Model_ST, 
                          bba_lane_ST, bba_face_ST, mLateralOffset, mMicroSleep);
       
        const int nbr_ST = 2;
        cv::Mat bba_ST = cv::Mat::zeros(nbr_ST, (int)pow(2.0, nbr_hyp_ST), CV_64F);
        bba_lane_ST.copyTo(bba_ST.row(0));
        bba_face_ST.copyTo(bba_ST.row(1));
        
        //! Choose the Combination Rule
        cv::Mat bba_ST_cmb(1, (int)pow(2.0, nbr_hyp_ST), CV_64F);
        switch (tagRule_ST) {
            case Dempster:
                Combine_Dempster(bba_ST, bba_ST_cmb);
                break;
            case Conjunctive:
                Combine_Conj(bba_ST, bba_ST_cmb);
                break;
            case Disjunctive:
                Combine_Disj(bba_ST, bba_ST_cmb);
                break;
            case CautiousConj:
                Combine_CautiousConj(bba_ST, bba_ST_cmb);
                break;
            case CautiousConjNorm:
                Combine_CautiousConjNorm(bba_ST, bba_ST_cmb);
                break;
            case Yager:
                Combine_Yager(bba_ST, bba_ST_cmb);
                break; 
            case DuboisPrade:
                Combine_DuboisPrade(bba_ST, bba_ST_cmb);
                break;  
            default:
                break;
        }//end switch
        
//        PrintMat(bba_ST_cmb);
        
        //! Results
        results_ST.clear();
        results_ST.push_back(bba_ST_cmb.at<double>(0,1));//Non Impaired
        results_ST.push_back(bba_ST_cmb.at<double>(0,2));//Impaired
        results_ST.push_back(bba_ST_cmb.at<double>(0,3));//ALL
//        MakeDecision_ORD(bba_ST_cmb, tagDecision, results_ST);
        
        conflict.push_back(bba_ST_cmb.at<double>(0,0));
        
        if(DEBUG_SHOW) {
            char *text = new char[50];
            sprintf(text, "ShortTerm >> LO: %.3f, MS: %.3f", mLateralOffset, mMicroSleep);
            cv::putText(debugShow, text, cv::Point(0, 170), cv::FONT_HERSHEY_SIMPLEX, 0.3, CV_RGB(0,255,0));
            
            for(int i = 0; i < (int)bba_ST.rows; i++) {
                for(int j = 0; j < (int)bba_ST.cols; j++) {
                    sprintf(text, "%.3f", bba_ST.at<double>(i,j));
                    cv::putText(debugShow, text, cv::Point(j*30, i*20+180), cv::FONT_HERSHEY_SIMPLEX, 0.3, CV_RGB(0,255,0));
                }
            }
            
            for(int j = 0; j < (int)bba_ST_cmb.cols; j++) {
                sprintf(text, "%.3f", bba_ST_cmb.at<double>(0,j));
                cv::putText(debugShow, text, cv::Point(j*30, 260), cv::FONT_HERSHEY_SIMPLEX, 0.3, CV_RGB(0,255,0));
            }
            delete text;
        }
    }//end FusionCenter_MultiORD
    
    
    
    

    void FusionCenter_ORD_TEF(const LaneFeature *laneFeatures,
                               const FaceFeature *faceFeatures,
                               const int &tagRule, const int &tagDecision,
                               std::deque<double> *conflict,
                               std::vector<double> &CUSUM, std::vector<double> &model,
                               cv::Mat bba_cmb_TEF, cv::Mat bba_cmb, cv::Mat bba_predict,
                               cv::Mat bba_cmb_lane, cv::Mat bba_cmb_face,
                               double &result)
    {
        const int nbr_hyp = 5; //ND, SD, MD, VD, ED
        
        //! Choose Modeling Method
        std::vector<cv::Mat> bba_lane, bba_face;//5x4xn
        double interval = 0.025;
        Model_ORD_TEF_Lane(laneFeatures, interval, bba_lane);
        Model_ORD_TEF_Face(faceFeatures, interval, bba_face);
        
        int laneParamsNum = (int)bba_lane.size();
        int faceParamsNum = (int)bba_face.size();
        
//        std::cout << "BBA_lane" << std::endl;
//        for(int i = 0; i < bba_lane.size(); i++)
//            PrintMat(bba_lane.at(i));
        
//        std::cout << "BBA_face" << std::endl;
//        for(int i = 0; i < bba_face.size(); i++)
//          PrintMat(bba_face.at(i));
        
        ///! Combination in Multi-level Framework
        switch (tagRule) {
            case CautiousConjNorm_Conjunctive:
                //!CautiousConjNorm for Lane Features
                if (laneParamsNum != 1) {
                    for (int i = 0; i < nbr_hyp; i++) {
                        cv::Mat mass_lane(laneParamsNum, 4, CV_64F, cv::Scalar::all(0));
                        for (int j = 0; j < laneParamsNum; j++) {
                            bba_lane.at(j).row(i).copyTo(mass_lane.row(j));
                        }
                        cv::Mat bba_cmb_lane_temp(1, 4, CV_64F);
                        Combine_CautiousConjNorm(mass_lane, bba_cmb_lane_temp);
                        bba_cmb_lane_temp.copyTo(bba_cmb_lane.row(i));
                    } 
                } else {
                    bba_lane.at(0).copyTo(bba_cmb_lane);
                }
//                std::cout << "Combination Lane" << std::endl;
//                PrintMat(bba_cmb_lane);
                
                //!CautiousConjNorm for Face Features
                if (faceParamsNum != 1) {
                    for (int i = 0; i < nbr_hyp; i++) {
                        cv::Mat mass_face(faceParamsNum, 4, CV_64F, cv::Scalar::all(0));
                        for (int j = 0; j < faceParamsNum; j++) {
                            bba_face.at(j).row(i).copyTo(mass_face.row(j));
                        }
                        cv::Mat bba_cmb_face_temp(1, 4, CV_64F);
                        Combine_CautiousConjNorm(mass_face, bba_cmb_face_temp);
                        bba_cmb_face_temp.copyTo(bba_cmb_face.row(i));
                    } 
                } else {
                    bba_face.at(0).copyTo(bba_cmb_face);
                }
//                std::cout << "Combination Face" << std::endl;
//                PrintMat(bba_cmb_face);

                //!Conjunctive Rule for bba_cmb
                for (int i = 0; i < nbr_hyp; i++) {
                    cv::Mat bba_multiSrc(2, 4, CV_64F);
                    bba_cmb_lane.row(i).copyTo(bba_multiSrc.row(0));
                    bba_cmb_face.row(i).copyTo(bba_multiSrc.row(1));
                    
                    cv::Mat bba_cmb_temp(1, 4, CV_64F);
                    Combine_Conj(bba_multiSrc, bba_cmb_temp);
                    bba_cmb_temp.copyTo(bba_cmb.row(i));
                }
//                std::cout << "Measure BBA" << std::endl;
//                PrintMat(bba_cmb);
                break;
            default:
                break;
        }
        
        
        ///! TEF model
        double gamma = 0.9, stop_th = 3; //belief mass, stop threshold
        for (int i = 0; i < nbr_hyp; i++) {
            //!Building the model(True/False), based on Time t
            if (bba_cmb.at<double>(i,0) > 0) {
                gamma = 1 - bba_cmb.at<double>(i,0);
            } else {
                gamma = 0.9;
            }
            cv::Mat model_true = (cv::Mat_<double>(1,4)<< 0, gamma, 0, 1-gamma);
            cv::Mat model_fault = (cv::Mat_<double>(1,4)<< 0, 0, gamma, 1-gamma);
            
            //!Detection of model change
            //!keep model = 1, change model = 0;
            //!Calculate the predicted BBA, based on Time t-1
            if ((int)CUSUM.size() != 5 ) {
                std::cout << "Error: CUSUM Vector Wrong!" << std::endl;
                abort();
            }
            cv::Mat bba_model(2, 4, CV_64F);//For combination
            if(CUSUM.at(i) < stop_th) {
                //!keep model
                if(model.at(i) == 1) {
                    //!True model
                    model_true.copyTo(bba_model.row(0));
                    bba_cmb_TEF.row(i).copyTo(bba_model.row(1));//Results of Time: t-1
                } else {
                    //!False model
                    model_fault.copyTo(bba_model.row(0));
                    bba_cmb_TEF.row(i).copyTo(bba_model.row(1));//Results of Time: t-1
                }
            } 
            else {
                //!change model
                CUSUM.at(i) = 0;
                if(model.at(i) == 1) {
                    //!change to False model
                    model_fault.copyTo(bba_model.row(0));
                    bba_cmb_TEF.row(i).copyTo(bba_model.row(1));//Results of Time: t-1
                } else {
                    //!change to True model
                    model_true.copyTo(bba_model.row(0));
                    bba_cmb_TEF.row(i).copyTo(bba_model.row(1));//Results of Time: t-1
                }
                model.at(i) = 1 - model.at(i);//Change model
            }
            //!Building the predicted BBA
            cv::Mat bba_predict_temp(1, 4, CV_64F);
            Combine_Disj(bba_model, bba_predict_temp);
            bba_predict_temp.copyTo(bba_predict.row(i));
        }
        
        
        ///! Fusion of Prediction and Meausrement
        //std::cout << "Fusion of Prediction and Measurement" << std::endl;
        double lambda = 0.9;//fader coping with low conflict
        cv::Mat mass_fusion_temp(2, 4, CV_64F);
        cv::Mat bba_fusion(1, 4, CV_64F);
        for(int i = 0; i < nbr_hyp; i++ ) {
            /* 
             * Still Some Problem with bba_cmb
             * because of the conflict bba_cmb(0)
             */
            bba_predict.row(i).copyTo(mass_fusion_temp.row(0));//Predict of Time: t;
            bba_cmb.row(i).copyTo(mass_fusion_temp.row(1)); //Measure of Time: t;
            Combine_Conj(mass_fusion_temp, bba_fusion);
//            std::cout << "BBA_fusion "<<std::endl;
//            PrintMat(bba_fusion);
            
            //!CUSUM process
            const double windowSize = 5;
            while ((int)conflict[i].size() >= windowSize)
                conflict[i].pop_front();
            conflict[i].push_back(bba_fusion.at<double>(0));//epsilon
            
            CUSUM.at(i) = lambda * CUSUM.at(i) + bba_fusion.at<double>(0);
            
            //!Update the final bba_cmb_TEF
            bba_fusion.copyTo(bba_cmb_TEF.row(i));//bba_cmb_TEF: 5x4;
            
            //!TEF output
            double delta, sum = 0;
            for(int j = 0; j < (int)conflict[i].size(); j++ ) {
                sum += conflict[i].at(j);
            }
            delta = 1/windowSize * sum;
            
            if (bba_fusion.at<double>(0) > delta) {
                bba_predict.row(i).copyTo(bba_cmb_TEF.row(i));
            } //!Otherwise, keep the fusion results

            
            //!Redistribution rule
            if (model.at(i) == 1) {
                //! True model
                bba_cmb_TEF.at<double>(i,1) = bba_cmb_TEF.at<double>(i,1) + bba_cmb_TEF.at<double>(i,0);
                bba_cmb_TEF.at<double>(i,3) = bba_cmb_TEF.at<double>(i,3) + bba_cmb_TEF.at<double>(i,2);
                bba_cmb_TEF.at<double>(i,0) = 0;
                bba_cmb_TEF.at<double>(i,2) = 0;
            } else {
                //! False model
                bba_cmb_TEF.at<double>(i,2) = bba_cmb_TEF.at<double>(i,2) + bba_cmb_TEF.at<double>(i,0);
                bba_cmb_TEF.at<double>(i,3) = bba_cmb_TEF.at<double>(i,3) + bba_cmb_TEF.at<double>(i,1);
                bba_cmb_TEF.at<double>(i,0) = 0;
                bba_cmb_TEF.at<double>(i,1) = 0;
            }
        }
        
        //!Final BBA
        //!ND
        std::cout   << " ND_emp: "  << bba_cmb_TEF.at<double>(0,0)
        << " ND_T: "    << bba_cmb_TEF.at<double>(0,1) 
        << " ND_F: "    << bba_cmb_TEF.at<double>(0,2) 
        << " ND_TF: "   << bba_cmb_TEF.at<double>(0,3) << std::endl; 
        //!SD
        std::cout   << " SD_emp: "  << bba_cmb_TEF.at<double>(1,0) 
        << " SD_T: "    << bba_cmb_TEF.at<double>(1,1) 
        << " SD_F: "    << bba_cmb_TEF.at<double>(1,2) 
        << " SD_TF: "   << bba_cmb_TEF.at<double>(1,3) << std::endl;
        //!MD
        std::cout   << " MD_emp: "  << bba_cmb_TEF.at<double>(2,0) 
        << " MD_T: "    << bba_cmb_TEF.at<double>(2,1)
        << " MD_F: "    << bba_cmb_TEF.at<double>(2,2)
        << " MD_TF: "   << bba_cmb_TEF.at<double>(2,3) << std::endl;
        //!VD
        std::cout   << " VD_emp: "  << bba_cmb_TEF.at<double>(3,0) 
        << " VD_T: "    << bba_cmb_TEF.at<double>(3,1)
        << " VD_F: "    << bba_cmb_TEF.at<double>(3,2)
        << " VD_TF: "   << bba_cmb_TEF.at<double>(3,3) << std::endl;
        //!ED
        std::cout   << " ED_emp: "  << bba_cmb_TEF.at<double>(4,0)
        << " ED_T: "    << bba_cmb_TEF.at<double>(4,1)
        << " ED_F: "    << bba_cmb_TEF.at<double>(4,2)
        << " ED_TF: "   << bba_cmb_TEF.at<double>(4,3) << std::endl;
        
        //!Measure BBA
        //!ND
        std::cout   << " (M)ND_emp: "  << bba_cmb.at<double>(0,0)
        << " (M)ND_T: "    << bba_cmb.at<double>(0,1) 
        << " (M)ND_F: "    << bba_cmb.at<double>(0,2) 
        << " (M)ND_TF: "   << bba_cmb.at<double>(0,3) << std::endl; 
        //!SD
        std::cout   << " (M)SD_emp: "  << bba_cmb.at<double>(1,0) 
        << " (M)SD_T: "    << bba_cmb.at<double>(1,1) 
        << " (M)SD_F: "    << bba_cmb.at<double>(1,2) 
        << " (M)SD_TF: "   << bba_cmb.at<double>(1,3) << std::endl;
        //!MD
        std::cout   << " (M)MD_emp: "  << bba_cmb.at<double>(2,0) 
        << " (M)MD_T: "    << bba_cmb.at<double>(2,1)
        << " (M)MD_F: "    << bba_cmb.at<double>(2,2)
        << " (M)MD_TF: "   << bba_cmb.at<double>(2,3) << std::endl;
        //!VD
        std::cout   << " (M)VD_emp: "  << bba_cmb.at<double>(3,0) 
        << " (M)VD_T: "    << bba_cmb.at<double>(3,1)
        << " (M)VD_F: "    << bba_cmb.at<double>(3,2)
        << " (M)VD_TF: "   << bba_cmb.at<double>(3,3) << std::endl;
        //!ED
        std::cout   << " (M)ED_emp: "  << bba_cmb.at<double>(4,0)
        << " (M)ED_T: "    << bba_cmb.at<double>(4,1)
        << " (M)ED_F: "    << bba_cmb.at<double>(4,2)
        << " (M)ED_TF: "   << bba_cmb.at<double>(4,3) << std::endl;
        
        //!Predicted BBA
        //!ND
        std::cout   << " (P)ND_emp: "  << bba_predict.at<double>(0,0)
        << " (P)ND_T: "    << bba_predict.at<double>(0,1) 
        << " (P)ND_F: "    << bba_predict.at<double>(0,2) 
        << " (P)ND_TF: "   << bba_predict.at<double>(0,3) << std::endl; 
        //!SD
        std::cout   << " (P)SD_emp: "  << bba_predict.at<double>(1,0) 
        << " (P)SD_T: "    << bba_predict.at<double>(1,1) 
        << " (P)SD_F: "    << bba_predict.at<double>(1,2) 
        << " (P)SD_TF: "   << bba_predict.at<double>(1,3) << std::endl;
        //!MD
        std::cout   << " (P)MD_emp: "  << bba_predict.at<double>(2,0) 
        << " (P)MD_T: "    << bba_predict.at<double>(2,1)
        << " (P)MD_F: "    << bba_predict.at<double>(2,2)
        << " (P)MD_TF: "   << bba_predict.at<double>(2,3) << std::endl;
        //!VD
        std::cout   << " (P)VD_emp: "  << bba_predict.at<double>(3,0) 
        << " (P)VD_T: "    << bba_predict.at<double>(3,1)
        << " (P)VD_F: "    << bba_predict.at<double>(3,2)
        << " (P)VD_TF: "   << bba_predict.at<double>(3,3) << std::endl;
        //!ED
        std::cout   << " ED_emp: "  << bba_predict.at<double>(4,0)
        << " ED_T: "    << bba_predict.at<double>(4,1)
        << " ED_F: "    << bba_predict.at<double>(4,2)
        << " ED_TF: "   << bba_predict.at<double>(4,3) << std::endl;
        
        //! Make Decision
        std::cout   << " DrowsyLevel: " << result << std::endl;
    }
    
    
    const double scale_ND = 0;//0~1
    const double scale_SD = 0.25;//1~3
    const double scale_MD = 0.5;//3~5
    const double scale_VD = 0.75;//5~7
    const double scale_ED = 1;//7~8
    double scale[5] = {scale_ND, scale_SD, scale_MD, scale_VD, scale_ED};
    
	int EMPTY = 0, ND = 1, SD = 2, ND_SD = 3, MD = 4, ND_MD = 5, SD_MD = 6, ND_SD_MD = 7;
    int VD = 8, ND_VD = 9, SD_VD = 10, ND_SD_VD = 11, MD_VD = 12, ND_MD_VD = 13, SD_MD_VD = 14, ND_SD_MD_VD = 15;
    int ED = 16, ND_ED = 17, SD_ED = 18, ND_SD_ED = 19, MD_ED = 20, ND_MD_ED = 21, SD_MD_ED = 22, ND_SD_MD_ED = 23;
	int VD_ED = 24, ND_VD_ED = 25, SD_VD_ED = 26, ND_SD_VD_ED = 27, MD_VD_ED = 28, ND_MD_VD_ED = 29, SD_MD_VD_ED = 30, ALL = 31;
    void MakeDecision_ORD(const cv::Mat &bba_cmb, const int &tagDecision, std::vector<double> &results)
    {
        int nbr_elem = (int)log2((double)bba_cmb.cols);
        cv::Mat data(1, nbr_elem, CV_64F, cv::Scalar::all(0));
        //PrintMat(bba_cmb);
        
        switch (tagDecision) {
            case BetP:
                //ND
                data.at<double>(0, 0) = bba_cmb.at<double>(0, ND) + 
										bba_cmb.at<double>(0, ND_SD)/2.0 + bba_cmb.at<double>(0, ND_MD)/2.0 + bba_cmb.at<double>(0, ND_VD)/2.0 + bba_cmb.at<double>(0,ND_ED)/2.0 + //4 two elements
                                        bba_cmb.at<double>(0, ND_SD_MD)/3.0 + bba_cmb.at<double>(0, ND_SD_VD)/3.0 + bba_cmb.at<double>(0, ND_SD_ED)/3.0 + bba_cmb.at<double>(0, ND_MD_VD)/3.0 + bba_cmb.at<double>(0, ND_MD_ED)/3.0 + bba_cmb.at<double>(0, ND_VD_ED)/3.0 +// 6 three elements
                                        bba_cmb.at<double>(0, ND_SD_MD_VD)/4.0 + bba_cmb.at<double>(0, ND_SD_MD_ED)/4.0 + bba_cmb.at<double>(0, ND_SD_VD_ED)/4.0 + bba_cmb.at<double>(0, ND_MD_VD_ED)/4.0 + // 4 four elements
                                        bba_cmb.at<double>(0, ALL)/5.0;
                //SD
                data.at<double>(0, 1) = bba_cmb.at<double>(0, SD) +
                                        bba_cmb.at<double>(0, ND_SD)/2.0 + bba_cmb.at<double>(0, SD_MD)/2.0 + bba_cmb.at<double>(0, SD_VD)/2.0 + bba_cmb.at<double>(0, SD_ED)/2.0 + //4 two elements
                                        bba_cmb.at<double>(0, ND_SD_MD)/3.0 + bba_cmb.at<double>(0, ND_SD_VD)/3.0 + bba_cmb.at<double>(0, ND_SD_ED)/3.0 + bba_cmb.at<double>(0, SD_MD_VD)/3.0 + bba_cmb.at<double>(0, SD_MD_ED)/3.0 + bba_cmb.at<double>(0, SD_VD_ED)/3.0  + // 6 three elements
                                        bba_cmb.at<double>(0, ND_SD_MD_VD)/4.0 + bba_cmb.at<double>(0, ND_SD_MD_ED)/4.0 + bba_cmb.at<double>(0, ND_SD_VD_ED)/4.0 + bba_cmb.at<double>(0, SD_MD_VD_ED)/4.0 + //4 four elements
                                        bba_cmb.at<double>(0, ALL)/5.0;
                //MD
                data.at<double>(0, 2) = bba_cmb.at<double>(0, MD) +
										bba_cmb.at<double>(0, ND_MD)/2.0 + bba_cmb.at<double>(0, SD_MD)/2.0 + bba_cmb.at<double>(0, MD_VD)/2.0 + bba_cmb.at<double>(0, MD_ED)/2.0 + //4 two elements
                                        bba_cmb.at<double>(0, ND_SD_MD)/3.0 + bba_cmb.at<double>(0, ND_MD_VD)/3.0 + bba_cmb.at<double>(0, ND_MD_ED)/3.0 + bba_cmb.at<double>(0, SD_MD_VD)/3.0 + bba_cmb.at<double>(0, SD_MD_ED)/3.0 + bba_cmb.at<double>(0, MD_VD_ED)/3.0 + //6 three elements
                                        bba_cmb.at<double>(0, ND_SD_MD_VD)/4.0 + bba_cmb.at<double>(0, ND_SD_MD_ED)/4.0 + bba_cmb.at<double>(0, SD_MD_VD_ED)/4.0 + bba_cmb.at<double>(0, ND_MD_VD_ED)/4.0 + //4 four elements
                                        bba_cmb.at<double>(0, ALL)/5.0;
                //VD
                data.at<double>(0, 3) = bba_cmb.at<double>(0, VD) + 
                                        bba_cmb.at<double>(0, ND_VD)/2.0 + bba_cmb.at<double>(0, SD_VD)/2.0 + bba_cmb.at<double>(0, MD_VD)/2.0 + bba_cmb.at<double>(0, VD_ED)/2.0 + //4 two elements
                                        bba_cmb.at<double>(0, ND_SD_VD)/3.0 + bba_cmb.at<double>(0, ND_MD_VD)/3.0 + bba_cmb.at<double>(0, SD_MD_VD)/3.0 + bba_cmb.at<double>(0, ND_VD_ED)/3.0 + bba_cmb.at<double>(0, SD_VD_ED)/3.0 + bba_cmb.at<double>(0, MD_VD_ED)/3.0 + //6 three elements
                                        bba_cmb.at<double>(0, ND_SD_MD_VD)/4.0 + bba_cmb.at<double>(0, ND_SD_VD_ED)/4.0 + bba_cmb.at<double>(0, SD_MD_VD_ED)/4.0 + bba_cmb.at<double>(0, ND_MD_VD_ED)/4.0 + //4 four elements
                                        bba_cmb.at<double>(0, ALL)/5.0; 
                //ED
                data.at<double>(0, 4) = bba_cmb.at<double>(0, ED) + 
										bba_cmb.at<double>(0, ND_ED)/2.0 + bba_cmb.at<double>(0, SD_ED)/2.0 + bba_cmb.at<double>(0, MD_ED)/2.0 + bba_cmb.at<double>(0, VD_ED)/2.0 + //4 two elements
                                        bba_cmb.at<double>(0, ND_SD_ED)/3.0 + bba_cmb.at<double>(0, ND_MD_ED)/3.0 + bba_cmb.at<double>(0, ND_VD_ED)/3.0 + bba_cmb.at<double>(0, SD_MD_ED)/3.0 + bba_cmb.at<double>(0, SD_VD_ED)/3.0 + bba_cmb.at<double>(0, MD_VD_ED)/3.0 + //6 three elements
                                        bba_cmb.at<double>(0, ND_SD_MD_ED)/4.0 + bba_cmb.at<double>(0, ND_SD_VD_ED)/4.0 + bba_cmb.at<double>(0, SD_MD_VD_ED)/4.0 + bba_cmb.at<double>(0, ND_MD_VD_ED)/4.0 + //4 four elements
                                        bba_cmb.at<double>(0, ALL)/5.0;
                break;
                
            case Belief:
                break;
            case Plausibility:
                break;
            default:
                break;
        }
        //PrintMat(data);
        
        for(int i = 0; i < nbr_elem; i++){
            results.push_back(data.at<double>(0,i));//ND,SD,MD,VD,ED
        }
        
        double DrowsyLevel = 0;
        //! Mean based on BetP
        if(tagDecision == BetP) {
            for(size_t i = 0; i < results.size(); i++) {
                DrowsyLevel += results.at(i) * scale[i];
            }
            results.push_back(DrowsyLevel);
        }
    }//MakeDecision_ORD
    
    // Combine any two indicators as unit
    void MakeDecision_Single_ORD(const cv::Mat bba_cmb[], const int &nbr_unit_LT,
                                 std::vector<double> &results)
    {
        int nbr_elem = (int)log2((double)bba_cmb[0].cols);
        cv::Mat bba(nbr_unit_LT, nbr_elem, CV_64F, cv::Scalar::all(0)); //1 x 5
        
        // Get BetP, now only 5 elements
        for(int i = 0; i < nbr_unit_LT; i++){
            //ND
            bba.at<double>(i, 0) = bba_cmb[i].at<double>(0, ND) + bba_cmb[i].at<double>(0, ND_SD)/2.0 
            + bba_cmb[i].at<double>(0, ND_MD)/2.0 + bba_cmb[i].at<double>(0, ND_VD)/2.0 
            + bba_cmb[i].at<double>(0,ND_ED)/2.0 //4 two elements
            + bba_cmb[i].at<double>(0, ND_SD_MD)/3.0 + bba_cmb[i].at<double>(0, ND_SD_VD)/3.0 
            + bba_cmb[i].at<double>(0, ND_SD_ED)/3.0 + bba_cmb[i].at<double>(0, ND_MD_VD)/3.0 
            + bba_cmb[i].at<double>(0, ND_MD_ED)/3.0 + bba_cmb[i].at<double>(0, ND_VD_ED)/3.0 // 6 three elements
            + bba_cmb[i].at<double>(0, ND_SD_MD_VD)/4.0 + bba_cmb[i].at<double>(0, ND_SD_MD_ED)/4.0 
            + bba_cmb[i].at<double>(0, ND_SD_VD_ED)/4.0 + bba_cmb[i].at<double>(0, ND_MD_VD_ED)/4.0 // 4 four elements
            + bba_cmb[i].at<double>(0, ALL)/5.0;
            
            //SD
            bba.at<double>(i, 1) = bba_cmb[i].at<double>(0, SD) + bba_cmb[i].at<double>(0, ND_SD)/2.0 
            + bba_cmb[i].at<double>(0, SD_MD)/2.0 + bba_cmb[i].at<double>(0, SD_VD)/2.0 
            + bba_cmb[i].at<double>(0, SD_ED)/2.0 //4 two elements
            + bba_cmb[i].at<double>(0, ND_SD_MD)/3.0 + bba_cmb[i].at<double>(0, ND_SD_VD)/3.0 
            + bba_cmb[i].at<double>(0, ND_SD_ED)/3.0 + bba_cmb[i].at<double>(0, SD_MD_VD)/3.0 
            + bba_cmb[i].at<double>(0, SD_MD_ED)/3.0 + bba_cmb[i].at<double>(0, SD_VD_ED)/3.0 // 6 three elements
            + bba_cmb[i].at<double>(0, ND_SD_MD_VD)/4.0 + bba_cmb[i].at<double>(0, ND_SD_MD_ED)/4.0 
            + bba_cmb[i].at<double>(0, ND_SD_VD_ED)/4.0 + bba_cmb[i].at<double>(0, SD_MD_VD_ED)/4.0 //4 four elements
            + bba_cmb[i].at<double>(0, ALL)/5.0;
            
            //MD
            bba.at<double>(i, 2) = bba_cmb[i].at<double>(0, MD) + bba_cmb[i].at<double>(0, ND_MD)/2.0 
            + bba_cmb[i].at<double>(0, SD_MD)/2.0 + bba_cmb[i].at<double>(0, MD_VD)/2.0 
            + bba_cmb[i].at<double>(0, MD_ED)/2.0 //4 two elements
            + bba_cmb[i].at<double>(0, ND_SD_MD)/3.0 + bba_cmb[i].at<double>(0, ND_MD_VD)/3.0 
            + bba_cmb[i].at<double>(0, ND_MD_ED)/3.0 + bba_cmb[i].at<double>(0, SD_MD_VD)/3.0 
            + bba_cmb[i].at<double>(0, SD_MD_ED)/3.0 + bba_cmb[i].at<double>(0, MD_VD_ED)/3.0 //6 three elements
            + bba_cmb[i].at<double>(0, ND_SD_MD_VD)/4.0 + bba_cmb[i].at<double>(0, ND_SD_MD_ED)/4.0 
            + bba_cmb[i].at<double>(0, SD_MD_VD_ED)/4.0 + bba_cmb[i].at<double>(0, ND_MD_VD_ED)/4.0 //4 four elements
            + bba_cmb[i].at<double>(0, ALL)/5.0;
            
            //VD
            bba.at<double>(i, 3) = bba_cmb[i].at<double>(0, VD) + bba_cmb[i].at<double>(0, ND_VD)/2.0 
            + bba_cmb[i].at<double>(0, SD_VD)/2.0 + bba_cmb[i].at<double>(0, MD_VD)/2.0 
            + bba_cmb[i].at<double>(0, VD_ED)/2.0 //4 two elements
            + bba_cmb[i].at<double>(0, ND_SD_VD)/3.0 + bba_cmb[i].at<double>(0, ND_MD_VD)/3.0 
            + bba_cmb[i].at<double>(0, SD_MD_VD)/3.0 + bba_cmb[i].at<double>(0, ND_VD_ED)/3.0 
            + bba_cmb[i].at<double>(0, SD_VD_ED)/3.0 + bba_cmb[i].at<double>(0, MD_VD_ED)/3.0 //6 three elements
            + bba_cmb[i].at<double>(0, ND_SD_MD_VD)/4.0 + bba_cmb[i].at<double>(0, ND_SD_VD_ED)/4.0 
            + bba_cmb[i].at<double>(0, SD_MD_VD_ED)/4.0 + bba_cmb[i].at<double>(0, ND_MD_VD_ED)/4.0 //4 four elements
            + bba_cmb[i].at<double>(0, ALL)/5.0; 
            
            //ED
            bba.at<double>(i, 4) = bba_cmb[i].at<double>(0, ED) + bba_cmb[i].at<double>(0, ND_ED)/2.0 
            + bba_cmb[i].at<double>(0, SD_ED)/2.0 + bba_cmb[i].at<double>(0, MD_ED)/2.0 
            + bba_cmb[i].at<double>(0, VD_ED)/2.0 //4 two elements
            + bba_cmb[i].at<double>(0, ND_SD_ED)/3.0 + bba_cmb[i].at<double>(0, ND_MD_ED)/3.0 
            + bba_cmb[i].at<double>(0, ND_VD_ED)/3.0 + bba_cmb[i].at<double>(0, SD_MD_ED)/3.0 
            + bba_cmb[i].at<double>(0, SD_VD_ED)/3.0 + bba_cmb[i].at<double>(0, MD_VD_ED)/3.0 //6 three elements
            + bba_cmb[i].at<double>(0, ND_SD_MD_ED)/4.0 + bba_cmb[i].at<double>(0, ND_SD_VD_ED)/4.0 
            + bba_cmb[i].at<double>(0, SD_MD_VD_ED)/4.0 + bba_cmb[i].at<double>(0, ND_MD_VD_ED)/4.0 //4 four elements
            + bba_cmb[i].at<double>(0, ALL)/5.0;
        }         
        
        cv::Mat result(1, nbr_elem, CV_64F, cv::Scalar::all(0));
        for(int i = 0; i < nbr_unit_LT; i++){
            result += bba.row(i);
        }
        // Mean weighted 
        for(int i =0; i < nbr_elem; i++) {
            results.push_back(result.at<double>(0,i)/(float)nbr_unit_LT);
        }
        // Final result
        double DrowsyLevel = 0;
        for(size_t i = 0; i < results.size(); i++) {
            DrowsyLevel += results.at(i) * scale[i];
        }
        results.push_back(DrowsyLevel);
        
    }//MakeDecision_Single_ORD
    
    
    
    
    
    void DecodeMsg(char* text, std::deque<double> &params)
    {
        char *temp = strtok(text, ", ");
        while (temp != NULL) {
            params.push_back(atof(temp));
            temp = strtok(NULL, ", ");
        }
    }//end DecodeMsg

}
