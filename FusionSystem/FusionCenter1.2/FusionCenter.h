//
//  FusionCenter.h
//  FusionCenter1.2
//
//  Created by XUANPENG LI on 11/11/13.
//  Copyright (c) 2013 __MyCompanyName__. All rights reserved.
//

#ifndef FusionCenter_Process_h
#define FusionCenter_Process_h

#include "ModelFeatures.h"
#include "TransferFunctions.h"
#include "CombineRules.h"
#include "Tools.h"

#include <vector>
#include <deque>
#include <opencv2/opencv.hpp>

enum CmbRule {  
    Dempster,
    Conjunctive,
    Disjunctive,
    CautiousConj,
    CautiousConjNorm,
    Yager,
    DuboisPrade,
    DuboisPradeNonDistinct
};

enum MultiCmbRule {
    CautiousConjNorm_Conjunctive    
};

enum Decision {
    BetP,
    Belief,
    Plausibility
}; 

namespace FusionCenter {
    void FusionCenter_SingleORD(const LaneFeature *laneFeatures,
                              const FaceFeature *faceFeatures,
                              const float &laneThreshold,
                              const float &faceThreshold,
                              const int &tagRule_LT, const int &tagRule_ST,
                              const int &tagDecision,
                              const double &passTime,
                              std::deque<std::vector<double> > &resultsSeq,
                              double &drowsyLevel, std::vector<double> &results,
                              std::vector<double> &conflict, cv::Mat &debugShow, 
                              double &mLATSD, double &mPERCLOS, double &mBLINK);
    
    
    void FusionCenter_MultiFusion(const LaneFeature *laneFeatures, 
                                  const FaceFeature *faceFeatures, 
                                  const int &tagRule_LT1, const int &tagRule_LT2,
                                  const int &tagDecision_LT,
                                  const int &tagRule_ST, 
                                  const double &intervalTime, std::vector<double> &results,
                                  double &mLATSD, double &mPERCLOS, double &mBLINK,
                                  double &T, double &TuF, double &F, cv::Mat &debugShow);
    
    
    void FusionCenter_MultiORD(const LaneFeature *laneFeatures, 
                               const FaceFeature *faceFeatures, 
                               const int &tagRule_LT1, const int &tagRule_LT2, 
                               const int &tagRule_ST, const int &tagDecision,
                               const double &intervalTime,
                               std::deque<std::vector<double> > &resultsSeq, 
                               double &drowsyLevel, 
                               std::vector<double> &results_ST, 
                               std::vector<double> &conflict, cv::Mat &debugShow, 
                               double &mLATSD, double &mPERCLOS, double &mBLINK);
    
    
    void FusionCenter_ORD_TEF(const LaneFeature *laneFeatures,
                              const FaceFeature *faceFeatures,
                              const int &tagRule, const int &tagDecision,
                              std::deque<double> *conflict,
                              std::vector<double> &CUSUM, std::vector<double> &model,
                              cv::Mat bba_cmb_TEF, cv::Mat bba_cmb, cv::Mat bba_predict, 
                              cv::Mat bba_cmb_lane, cv::Mat bba_cmb_face,
                              double &result);
    
    void MakeDecision_ORD(const cv::Mat &bba_cmb, 
                          const int &tagDecision, 
                          std::vector<double> &results);
    
    void MakeDecision_Single_ORD(const cv::Mat bba_cmb[], 
                                 const int &nbr_unit_LT,
                                 std::vector<double> &results);
    
    void DecodeMsg(char* text, std::deque<double> &params);
    
    
}

#endif
