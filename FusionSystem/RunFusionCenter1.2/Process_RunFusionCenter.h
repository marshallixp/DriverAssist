//
//  Process_RunFusionCenter.h
//  RunFusionCenter1.2
//
//  Created by XUANPENG LI on 11/11/13.
//  Copyright (c) 2013 __MyCompanyName__. All rights reserved.
//

#ifndef RunFusionCenter_Process_h
#define RunFusionCenter_Process_h

#include "../FusionCenter/ModelFeatures.h"
#include "../FusionCenter/FusionCenter.h"

#include <fstream>
#include <iterator>
#include <vector>
#include <deque>
#include <iomanip>
#include <opencv2/opencv.hpp>

namespace RunFusionCenter {
    const int NUM_FEATURE = 9;
    const int NUM_RESULT = 18;
    const int NUM_MULTI = 15;
    const int NUM_TEF = 22;
    
    const std::string featureName[NUM_FEATURE] =
    {
        "frame", "CCP", "LATSD",
        "EyeSize", "PERCLOS", "BLINK", "MST",
        "interTime", "pastTime"
    };
    
    const std::string resultName[NUM_RESULT] = 
    {
        "frame",
        "ND", "SD", "MD", "VD", "ED",
        "CurrentDL", "InterTime", "DrowsySeq",
        "ID_ST", "NID_ST", "ALLID_ST", 
        "mLATSD", "mPERCLOS", "mBLINK", "pastTime",
        "Conflict0", "Conflict1"
    };
    
    const std::string resultMultiName[NUM_MULTI] = 
    {
        "frame",
        "ND", "SD", "MD", "VD", "ED", 
        "DL", "InterTime"
        "mLATSD", "mPERCLOS", "mBLINK",
        "T", "TuF", "F","pastTime"
    };
    
    const std::string resultTEFName[NUM_TEF] = 
    {
        "frame",
        "ND_Emp", "ND_T", "ND_F", "ND_TF",
        "SD_Emp", "SD_T", "SD_F", "SD_TF",
        "MD_Emp", "MD_T", "MD_F", "MD_TF",
        "VD_Emp", "VD_T", "VD_F", "VD_TF",
        "ED_Emp", "ED_T", "ED_F", "ED_TF",
        "DrowsyLevel"
    };
    
    void InitRecordData(std::ofstream &file, const char* fileName, const std::string *strName, const int &elemNum);
    
    void RecordFeature(std::ofstream &file, const int &idx,
                       FusionCenter::LaneFeature *laneParams,
                       FusionCenter::FaceFeature *faceParams,
                       const double &intervalTime,
                       const double &pastTime);
    
    void RecordResult(std::ofstream &file, 
                      const int &idx, const std::vector<double> &results, 
                      const double &intervalTime,
                      const double &drowsyLevel,
                      const std::vector<double> &results_ST, const std::vector<double> &conflict, 
                      const double &mLATSD, const double &mPERCLOS, const double &mBLINK, 
                      const double &pastTime);
    
    void RecordMultiResult(std::ofstream &file, const int &idx, 
                           std::vector<double> &results,
                           const double &intervalTime,
                           const double &mLATSD, const double &mPERCLOS, const double &mBLINK,
                           const double &T, const double &TuF, const double &F, 
                           const double &pastTime);
    
    void RecordTEFResult(std::ofstream &file, const int &idx, const cv::Mat &results);
}


#endif
