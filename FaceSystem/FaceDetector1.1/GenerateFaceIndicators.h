//
//  GenerateFaceIndicators.h
//  FaceDetector1.1
//
//  Created by Xuanpeng Li on 05/13.
//  Copyright (c) 2012 ESIEE-AMIENS. All rights reserved.
//

#ifndef FaceDetector_GenerateFaceIndicators_h
#define FaceDetector_GenerateFaceIndicators_h

#include <vector>
#include <deque>
#include <iostream>
#include <opencv2/opencv.hpp>

namespace FaceDetector{
    typedef struct _InfoPERCLOS {
        double  time;       //intervalTime
        int     eyePERCLOS; //flag of eyes closed or not
        double  winTimePERCLOS;//sum of window time of PERCLOS
        double  timePERCLOS;//sum of PERCLOS time
    }InfoPERCLOS;
    
    typedef struct _InfoBLINK {
        double time;
        int     eyeBLINK;
        double  winTimeBLINK;  //sum of window time 
    }InfoBLINK;
    
    typedef struct _FaceFeature {
        int frame;
        double Height;
        double lastHeight;
        double Height_Baseline;
        double PERCLOS;
        double MICROSLEEPTime;
        double MICROSLEEP;
        double BLINK;
        double BLINK_Baseline;
    }FaceFeature;
    
    bool SortBigger(const double &value1, const double &value2);
    bool SortSmaller(const double &value1, const double &value2);
    
    void GetFaceBaseline(const int &sampleIdx,
                         std::vector<double> &eyesHeightVec,
                         std::vector<double> &eyesHeightBaselineVec,
                         FaceFeature &faceFeatures);
    
    void GenerateFaceIndicators(const int &sampleIdx,
                                std::vector<double> &eyesHeightVec, 
                                FaceFeature &faceFeatures,
                                std::deque<InfoPERCLOS> &PERCLOSDeque,
                                std::deque<InfoBLINK> &BLINKDeque,
                                const double &intervalTime);

} 

#endif
