//
//  ModelFeatures.h
//  FusionCenter1.2
//
//  Created by XUANPENG LI on 11/11/13.
//  Copyright (c) 2013 __MyCompanyName__. All rights reserved.
//

#ifndef FusionCenter_ModelFeatures_h
#define FusionCenter_ModelFeatures_h

#include <deque>
#include <opencv2/opencv.hpp>

namespace FusionCenter {
    typedef struct _LaneFeature{
        int frame;
        double lateralOffset;       //Current lateral offset
        double LATSD;               //Standard deviation of lateral offset
        double LATSD_Baseline;      //Baseline of LATSD
        double LATMEAN;             //Average of lateral offset by EWMA
        double LATMEAN_Baseline;    //Baseline of LATMEAN
        double LANEDEV;             //standard deviation of LATMEAN by EWVAR
        double LANEDEV_Baseline;    //Baseline of LATDEV
        double LANEX;               //Fraction of lane exits
        //double LANEX_Baseline;      //Baseline of LANEX;
        double TLC;                 //Simple time of lane crossing
        int    TLC_2s;              //Number of times that TLC fall below 0.5s
        double TLCF_2s;
        int    TLC_halfs;           //Number of times that TLC fall below 2s
        double TLCF_halfs;
        double TLC_min;             //Global minimum of TLC over a given time interval
        double TOT;                 //Time on task
    } LaneFeature;
    
    typedef struct _FaceFeature {
        int	   frame;
        double Height;
        double lastHeight;
        double Height_Baseline;
        double PERCLOS;
        double MICROSLEEPTime;
        double MICROSLEEP;
        double BLINK;
        double BLINK_Baseline;
    }FaceFeature;
    
    enum ModelType{
        CONFIDENCE,
        TRIANGULAR,
        GAUSSIAN
    };
    
    //! Using TEF model, every state has their own set {T,F}
    void Model_ORD_TEF_Lane(const LaneFeature *laneFeatures,
                            const double e, std::vector<cv::Mat> &bba);
    
    void Model_ORD_TEF_Face(const FaceFeature *faceFeatures,
                            const double e, std::vector<cv::Mat> &bba);
    
    //! Using the concept of Membership function of Fuzzy set 
    double getTriangularMass(const double &x, const double &a1, const double &a2);
    double getGaussianMass(const double &x, const double &k, const double &a);
    
    
    void Model_ORD_Lane(const LaneFeature *laneFeatures, 
                        const int &massType, const int &nbr_hyp, 
                        cv::Mat &bba_lane, double &mLATSD);
    
    void Model_ORD_Face(const FaceFeature *faceFeatures, 
                        const int &massType, const int &nbr_hyp, 
                        cv::Mat &bba_face, double &mPERCLOS, double &mBLINK);
    
    void Model_MultiORD_ST(const LaneFeature *laneFeatures, const FaceFeature *faceFeatures,
                           const int &massType, cv::Mat &bba_lane, cv::Mat &bba_face,
                           double &mLateralOffset, double &mMicroSleep);
    
    void Model_SingleORD_ST(const LaneFeature *laneFeatures, const FaceFeature *faceFeatures,
                            const float &laneThreshold, const float &faceThreshold, 
                            cv::Mat &bba_lane, cv::Mat &bba_face,
                            double &mLateralOffset, double &mMicroSleep);
}

#endif
