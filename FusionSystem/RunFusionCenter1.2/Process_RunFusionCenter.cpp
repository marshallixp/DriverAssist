//
//  Process_RunFusionCenter.cpp
//  RunFusionCenter1.2
//
//  Created by XUANPENG LI on 11/11/13.
//  Copyright (c) 2013 __MyCompanyName__. All rights reserved.
//
#include "Process_RunFusionCenter.h"
#include <iostream>

namespace RunFusionCenter {
    void InitRecordData(std::ofstream &file, const char* fileName, const std::string *strName, const int &elemNum)
    {
        file.open(fileName);
        
        for(int i = 0; i < elemNum; i++ )
        {
            file << strName[i] <<  std::setiosflags(std::ios::fixed) << std::setw(15);
        }
        file << std::endl;
    }//end InitRecordData
    
    void RecordFeature(std::ofstream &file, const int &idx,
                       FusionCenter::LaneFeature *laneParams,
                       FusionCenter::FaceFeature *faceParams,
                       const double &intervalTime,
                       const double &pastTime)
    {
        file << idx;
        file << std::setiosflags(std::ios::fixed) << std::setprecision(5) << std::setw(15) << laneParams->lateralOffset;
        file << std::setiosflags(std::ios::fixed) << std::setprecision(5) << std::setw(15) << laneParams->LATSD;
        file << std::setiosflags(std::ios::fixed) << std::setprecision(5) << std::setw(15) << faceParams->Height;
        file << std::setiosflags(std::ios::fixed) << std::setprecision(5) << std::setw(15) << faceParams->PERCLOS;
        file << std::setiosflags(std::ios::fixed) << std::setprecision(5) << std::setw(15) << faceParams->BLINK;
        file << std::setiosflags(std::ios::fixed) << std::setprecision(5) << std::setw(15) << faceParams->MICROSLEEP;
        file << std::setiosflags(std::ios::fixed) << std::setprecision(5) << std::setw(15) << intervalTime;
        file << std::setiosflags(std::ios::fixed) << std::setprecision(5) << std::setw(15) << pastTime;
        file << std::endl;
    }
    
    void RecordResult(std::ofstream &file, const int &idx, 
                      const std::vector<double> &results, 
                      const double &intervalTime,
                      const double &drowsyLevel, 
                      const std::vector<double> &results_ST, const std::vector<double> &conflict, 
                      const double &mLATSD, const double &mPERCLOS, const double &mBLINK,
                      const double &pastTime)
    {
        file << idx;
        //! Long-term 
        for (int i = 0; i < (int)results.size(); i++) {
            file << std::setiosflags(std::ios::fixed) << std::setprecision(5) << std::setw(15) << results[i];
        }
        file << std::setiosflags(std::ios::fixed) << std::setprecision(5) << std::setw(15) << intervalTime;
        
        file << std::setiosflags(std::ios::fixed) << std::setprecision(5) << std::setw(15) << drowsyLevel;
        
        //! Short-term
        for (int j = 0; j < (int)results_ST.size(); j++) {
            file << std::setiosflags(std::ios::fixed) << std::setprecision(5) << std::setw(15) << results_ST[j];
        }
        
        //! Mass Record
        file << std::setiosflags(std::ios::fixed) << std::setprecision(5) << std::setw(15) << mLATSD;
        file << std::setiosflags(std::ios::fixed) << std::setprecision(5) << std::setw(15) << mPERCLOS;
        file << std::setiosflags(std::ios::fixed) << std::setprecision(5) << std::setw(15) << mBLINK;
        file << std::setiosflags(std::ios::fixed) << std::setprecision(3) << std::setw(15) << pastTime;
        
        
        //! Conflict
        for(int k = 0; k < (int)conflict.size(); k++) {
        file << std::setiosflags(std::ios::fixed) << std::setprecision(5) << std::setw(15) << conflict[k];
        }
        
        file << std::endl;
    }//end Record Result
    
    void RecordMultiResult(std::ofstream &file, const int &idx, 
                           std::vector<double> &results,
                           const double &intervalTime,
                           const double &mLATSD, const double &mPERCLOS, const double &mBLINK,
                           const double &T, const double &TuF, const double &F, 
                           const double &pastTime)
    {
        file << idx;
        for(int i=0; i<results.size(); i++)
        {
            file << std::setiosflags(std::ios::fixed) << std::setprecision(5) << std::setw(15) << results[i];
        }
        file << std::setiosflags(std::ios::fixed) << std::setprecision(5) << std::setw(15) << intervalTime;
        
        file << std::setiosflags(std::ios::fixed) << std::setprecision(5) << std::setw(15) << mLATSD;
        file << std::setiosflags(std::ios::fixed) << std::setprecision(5) << std::setw(15) << mPERCLOS;
        file << std::setiosflags(std::ios::fixed) << std::setprecision(5) << std::setw(15) << mBLINK;
        
        file << std::setiosflags(std::ios::fixed) << std::setprecision(5) << std::setw(15) << T;
        file << std::setiosflags(std::ios::fixed) << std::setprecision(5) << std::setw(15) << TuF;
        file << std::setiosflags(std::ios::fixed) << std::setprecision(5) << std::setw(15) << F;
        file << std::setiosflags(std::ios::fixed) << std::setprecision(5) << std::setw(15) << pastTime;
        file << std::endl;
    }
    
    void RecordTEFResult(std::ofstream &file, const int &idx, const cv::Mat &results)
    {
        file << idx;
        for(int i = 0; i < 5; i++)
        {
            for(int j = 0; j < 4; j++)
                file << std::setiosflags(std::ios::fixed) << std::setprecision(5) << std::setw(15) << results.at<double>(i,j);
        }
        file << std::endl;
    }//end RecordTEFResult

}//end RunFusionCenter