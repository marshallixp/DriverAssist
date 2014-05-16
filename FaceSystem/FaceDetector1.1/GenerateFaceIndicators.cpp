//
//  GenerateFaceIndicators.cpp
//  FaceDetector1.1
//
//  Created by Xuanpeng Li on 05/13.
//  Copyright (c) 2012 ESIEE-AMIENS. All rights reserved.
//
#include "GenerateFaceIndicators.h"


/// Sampling Interval
extern const double TIME_PERCLOS_WINDOW;   
extern const double TIME_BLINK_WINDOW;  
/// Threshold
extern const double THRESHOLD_PERCLOS;
extern const double THRESHOLD_CLOSURE; 
/// Baseline
extern const int TIME_BASELINE;

namespace FaceDetector{
    /// small to big
    bool SortBigger(const double &value1, const double &value2) {
        return value1 < value2;
    }
    
    /// big to small
    bool SortSmaller(const double &value1, const double &value2) {
        return value1 > value2;
    }
    
    /***********************************************************************************/
    // This function acquires the baselines in first few time.
    /***********************************************************************************/
    void GetFaceBaseline(const int &sampleIdx,
                         std::vector<double> &eyesHeightVec,
                         std::vector<double> &eyesHeightBaselineVec,
                         FaceFeature &faceFeatures)
    { 
        faceFeatures.frame = sampleIdx;
        
        double eyesHeight_Baseline = 0, eyesHeight = 0, lastEyesHeight = faceFeatures.lastHeight;
        /****************************/
        /* Baseline of eye height   */
        /****************************/
        if (!eyesHeightVec.empty()) {
            sort(eyesHeightVec.begin(), eyesHeightVec.end(), SortBigger);
            eyesHeight = eyesHeightVec.front();
        }
        else {
            eyesHeight = lastEyesHeight;
        }
        eyesHeightBaselineVec.push_back(eyesHeight);
        sort(eyesHeightBaselineVec.begin(), eyesHeightBaselineVec.end(), SortBigger);
        eyesHeight_Baseline = eyesHeightBaselineVec.at(cvRound((double)eyesHeightBaselineVec.size()/2));//median of eyesHeightVec
    
        
        //!Update the faceFeatures struct
        faceFeatures.Height = eyesHeight;
        faceFeatures.lastHeight = eyesHeight;
        faceFeatures.Height_Baseline = eyesHeight_Baseline;
        faceFeatures.PERCLOS = 0;
        faceFeatures.MICROSLEEP = 0;
        faceFeatures.MICROSLEEPTime = 0;
        faceFeatures.BLINK = 0;
        faceFeatures.BLINK_Baseline = 0;
    }
    
    
    
    /***********************************************************************************/
    //  This function generates the mass of face related based on baselines. 
    //  \param: microsleep_baseline is according to defined statistics value (not used), 
    //  but different between individuals
    /***********************************************************************************/
    void GenerateFaceIndicators(const int &sampleIdx,
                                std::vector<double> &eyesHeightVec, 
                                FaceFeature &faceFeatures,
                                std::deque<InfoPERCLOS> &PERCLOSDeque, 
                                std::deque<InfoBLINK> &BLINKDeque, 
                                const double &intervalTime)
    {
        double eyesHeight = 0, lastEyesHeight = faceFeatures.lastHeight;
        double PERCLOS = 0, MICROSLEEP = 0, MICROSLEEPTime = faceFeatures.MICROSLEEPTime, BLINK = 0;
        
        double eyesHeight_b = faceFeatures.Height_Baseline;
        //Find the min size between two eyes.
        if (!eyesHeightVec.empty()) {
            sort(eyesHeightVec.begin(), eyesHeightVec.end(), SortBigger);
            eyesHeight = eyesHeightVec.front();
            if(eyesHeight > 2*eyesHeight_b )
                eyesHeight = 2*eyesHeight_b;
        }
        else {
            //Negative, if can NOT detect eyes correctly
            eyesHeight = lastEyesHeight;
        }

/*********************************************/
/*  PERCLOS: Percentage of eye closure       */
/*********************************************/   
        InfoPERCLOS perclosInfo;
        perclosInfo.time = intervalTime;
        if(!PERCLOSDeque.empty()){
            /// Calculate the number of frames in fixed time
            if (eyesHeight < faceFeatures.Height_Baseline * THRESHOLD_PERCLOS) {
                perclosInfo.eyePERCLOS = 1; // eye closed
                perclosInfo.timePERCLOS = PERCLOSDeque.back().timePERCLOS +intervalTime;
            }
            else {
                perclosInfo.eyePERCLOS = 0;
                perclosInfo.timePERCLOS = PERCLOSDeque.back().timePERCLOS;
                
            }
            perclosInfo.winTimePERCLOS = PERCLOSDeque.back().winTimePERCLOS + intervalTime;

            //! Only focus on the fixed time interval.
            while(perclosInfo.winTimePERCLOS > TIME_PERCLOS_WINDOW)
            {
                perclosInfo.winTimePERCLOS -= PERCLOSDeque.front().time;
                if(PERCLOSDeque.front().eyePERCLOS == 1)
                {
                    perclosInfo.timePERCLOS -= PERCLOSDeque.front().time;
                }
                PERCLOSDeque.pop_front();
            }
            
            //cout << "time PERCLOS: " << perclosInfo.timePERCLOS << endl;
        }
        else 
        {// The first frames without any PERCLOSInfo
            if (eyesHeight < faceFeatures.Height_Baseline * THRESHOLD_PERCLOS) {
                perclosInfo.eyePERCLOS = 1;
                perclosInfo.timePERCLOS = intervalTime;
            }
            else {
                perclosInfo.eyePERCLOS = 0;
                perclosInfo.timePERCLOS = 0;
            }
            perclosInfo.winTimePERCLOS = intervalTime;
        }
        PERCLOSDeque.push_back(perclosInfo);
        
        //! PERCLOS
        if(perclosInfo.winTimePERCLOS < TIME_PERCLOS_WINDOW / 2)
            PERCLOS = 0;//In first time interval too high value
        else 
            PERCLOS = perclosInfo.timePERCLOS / perclosInfo.winTimePERCLOS;
        
/************************************************/
/*  Statistics of Continuous Eye Closure        */
/*  MICROSLEEP: ND: 0~0.5                       */
/*              SD: 0.5~1                       */
/*              MD: 1~2                         */
/*              VD: 2~4                         */
/*              ED: 4~                          */
/************************************************/
        if (eyesHeight < faceFeatures.Height_Baseline * THRESHOLD_CLOSURE) {
            MICROSLEEPTime += intervalTime;
        } 
        else {
            MICROSLEEPTime -= intervalTime;
            MICROSLEEPTime = MICROSLEEPTime > 0 ? MICROSLEEPTime : 0;
        }
        
        //! When MICROSLEEPTime not equal to 0, Update the MICROSLEEP
        if (MICROSLEEPTime < 0.5)
            MICROSLEEP = 0.25 * MICROSLEEPTime;  //Alert
        else if (MICROSLEEPTime >= 0.5 && MICROSLEEPTime < 1)
            MICROSLEEP = 0.5 * MICROSLEEPTime - 0.125;  //Slightly Drowsy
        else if (MICROSLEEPTime >= 1 && MICROSLEEPTime < 2)
            MICROSLEEP = 0.25 * MICROSLEEPTime + 0.125;  //Moderately Drowsy
        else if(MICROSLEEPTime >= 2 && MICROSLEEPTime < 4)
            MICROSLEEP = 0.125 * MICROSLEEPTime + 0.375;  //Very Drowsy
        else if(MICROSLEEPTime >= 4 && MICROSLEEPTime < 5)
            MICROSLEEP = 0.125 * MICROSLEEPTime + 0.375;
        else
            MICROSLEEP = 1;    //Extremely Drowsy
        
/************************************************/
//! BLINK: Blink Frequency Statistics 
/************************************************/
        InfoBLINK blinkInfo;
        blinkInfo.time = intervalTime;
        if(!BLINKDeque.empty()) {
            /// Calculate the number of frames in fixed time
            if (eyesHeight < faceFeatures.Height_Baseline * THRESHOLD_CLOSURE) {
                blinkInfo.eyeBLINK = 1;
            }
            else {
                blinkInfo.eyeBLINK = 0;
            }
            blinkInfo.winTimeBLINK = BLINKDeque.back().winTimeBLINK + intervalTime;
            
            //! Only focus on the fixed time interval.
            while(blinkInfo.winTimeBLINK  > TIME_BLINK_WINDOW)
            {
                blinkInfo.winTimeBLINK  -= BLINKDeque.front().time;
                BLINKDeque.pop_front();
            }
        }
        else {
            if (eyesHeight < faceFeatures.Height_Baseline * THRESHOLD_CLOSURE) {
                blinkInfo.eyeBLINK = 1;
            }
            else {
                blinkInfo.eyeBLINK = 0;
            }
            blinkInfo.winTimeBLINK = intervalTime;
        }
        BLINKDeque.push_back(blinkInfo);

        //! Calculate the BLINK number 
        int flagBLINK = 0, numBLINK = 0;//start with open eyes
        for (std::deque<InfoBLINK>::const_iterator iter = BLINKDeque.begin(); iter != BLINKDeque.end(); ++iter)
        {
            if (iter->eyeBLINK == 1) {   
                //! closed eyes at first should change flag = 1.
                if (flagBLINK == 0) {
                    flagBLINK = 1;
                    numBLINK ++;
                }
            }
            else {  
                //! open eyes will change flag = 0.
                if (flagBLINK == 1) {
                    flagBLINK = 0;
                } 
            }// end if
        }// end for
        
        BLINK = numBLINK;
//        BLINK = (double)numBLINK / (double)BLINKDeque.size();
//        BLINK = (BLINK - faceFeatures.BLINK_Baseline) / faceFeatures.BLINK_Baseline;
//        BLINK = BLINK < 0 ? 0 : BLINK; 
//        BLINK = BLINK > 1 ? 1 : BLINK; 
        
        
        //! Update the faceFeatures
        faceFeatures.frame  = sampleIdx;
        faceFeatures.Height = eyesHeight;
        faceFeatures.lastHeight = eyesHeight;
        faceFeatures.PERCLOS = PERCLOS;
        faceFeatures.MICROSLEEPTime = MICROSLEEPTime;
        faceFeatures.MICROSLEEP = MICROSLEEP;
        faceFeatures.BLINK = BLINK;
        
    }///GenerateFaceIndicators
    
} ///namespace FaceDetector
