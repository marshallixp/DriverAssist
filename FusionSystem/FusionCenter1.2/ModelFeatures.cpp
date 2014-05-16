//
//  ModelFeatures.cpp
//  FusionCenter1.2
//
//  Created by XUANPENG LI on 11/11/13.
//  Copyright (c) 2013 __MyCompanyName__. All rights reserved.
//
#include "Tools.h"
#include "ModelFeatures.h"
#include <iostream>
#include <cmath>
using namespace DST;
namespace FusionCenter {
    //! Position of each states
    const int ND = 1, SD = 2, MD = 4, VD = 8, ED = 16;
    const int ND_SD = 3, SD_MD = 6, MD_VD = 12, VD_ED = 24, ALL = 31;
   
    const double k = 100; //GAUSSIAN Parameters
 
/********************************************************/
//! TEF model build each state with Omega = {T, F};
//! 2^Omega = {Empty, T, F, {T, F}};
// \param e: the region between the states.
// \param bba: { bba_ND; bba_SD; bba_MD; bba_VD; bba_ED }
/********************************************************/
    void Model_ORD_TEF_Lane(const LaneFeature *laneFeatures, 
                            const double e, std::vector<cv::Mat> &bba)
    {
        std::vector<double> params;
        params.push_back(laneFeatures->LATSD);
        params.push_back(laneFeatures->LANEX);
        params.push_back(laneFeatures->TLCF_2s);
        params.push_back(laneFeatures->TLCF_halfs);
        
        int nbr_src = (int)params.size();
        for (int i = 0; i < nbr_src; i++)
        {
            cv::Mat bba_ND(1,4,CV_64F, cv::Scalar::all(0));
            cv::Mat bba_SD(1,4,CV_64F, cv::Scalar::all(0));
            cv::Mat bba_MD(1,4,CV_64F, cv::Scalar::all(0));
            cv::Mat bba_VD(1,4,CV_64F, cv::Scalar::all(0));
            cv::Mat bba_ED(1,4,CV_64F, cv::Scalar::all(0));
            
            double param = params.at(i);
            if (param < 0.125 - e) {
                bba_ND.at<double>(1) = 1; //[0 1 0 0] 
                bba_SD.at<double>(3) = 1; //[0 0 0 1]
                bba_MD.at<double>(3) = 1; //[0 0 0 1]
                bba_VD.at<double>(3) = 1; //[0 0 0 1]
                bba_ED.at<double>(3) = 1; //[0 0 0 1]
            } else if( param >= 0.125 - e && param < 0.125) {
                bba_ND.at<double>(1) = (param - 0.125)/(-e);
                bba_ND.at<double>(3) = 1 - (param - 0.125)/(-e);
                bba_SD.at<double>(3) = 1;
                bba_MD.at<double>(3) = 1;
                bba_VD.at<double>(3) = 1;
                bba_ED.at<double>(3) = 1;
            } else if( param >= 0.125 && param < 0.125 + e) {
                bba_ND.at<double>(2) = 1 - (param - 0.125 - e)/(-e);
                bba_ND.at<double>(3) = (param - 0.125 - e)/(-e);
                bba_SD.at<double>(1) = (param - 0.125)/e;
                bba_SD.at<double>(3) = 1 - (param - 0.125)/e;
                bba_MD.at<double>(3) = 1;
                bba_VD.at<double>(3) = 1;
                bba_ED.at<double>(3) = 1;
            } else if(param >= 0.125 + e && param < 0.375 - e) {
                bba_ND.at<double>(2) = 1;
                bba_SD.at<double>(1) = 1;
                bba_MD.at<double>(3) = 1;
                bba_VD.at<double>(3) = 1;
                bba_ED.at<double>(3) = 1;
            } else if(param >= 0.375 - e && param < 0.375) {
                bba_ND.at<double>(2) = 1;
                bba_SD.at<double>(1) = (param - 0.375)/(-e);
                bba_SD.at<double>(3) = 1 - (param - 0.375)/(-e);
                bba_MD.at<double>(3) = 1;
                bba_VD.at<double>(3) = 1;
                bba_ED.at<double>(3) = 1;
            } else if(param >= 0.375 && param < 0.375 + e) {
                bba_ND.at<double>(2) = 1;
                bba_SD.at<double>(2) = 1 - (param - 0.375 -e)/(-e);
                bba_SD.at<double>(3) = (param - 0.375 -e)/(-e);
                bba_MD.at<double>(1) = (param - 0.375)/e;
                bba_MD.at<double>(3) = 1 - (param - 0.375)/e;
                bba_VD.at<double>(3) = 1;
                bba_ED.at<double>(3) = 1;
            } else if(param >= 0.375 + e && param < 0.625 - e) {
                bba_ND.at<double>(2) = 1;
                bba_SD.at<double>(2) = 1;
                bba_MD.at<double>(1) = 1;
                bba_VD.at<double>(3) = 1;
                bba_ED.at<double>(3) = 1;
            } else if(param >= 0.625 - e && param < 0.625) {
                bba_ND.at<double>(2) = 1;
                bba_SD.at<double>(2) = 1;
                bba_MD.at<double>(1) = (param - 0.625)/(-e);
                bba_MD.at<double>(3) = 1 - (param - 0.625)/(-e);
                bba_VD.at<double>(3) = 1;
                bba_ED.at<double>(3) = 1;
            } else if(param >= 0.625 && param < 0.625 + e) {
                bba_ND.at<double>(2) = 1;
                bba_SD.at<double>(2) = 1;
                bba_MD.at<double>(2) = 1 - (param - 0.625 -e)/(-e);
                bba_MD.at<double>(3) = (param - 0.625 -e)/(-e);
                bba_VD.at<double>(1) = (param - 0.625)/e;
                bba_VD.at<double>(3) = 1 - (param - 0.625)/e;
                bba_ED.at<double>(3) = 1;
            } else if(param >= 0.625 + e && param < 0.875 - e) {
                bba_ND.at<double>(2) = 1;
                bba_SD.at<double>(2) = 1;
                bba_MD.at<double>(2) = 1;
                bba_VD.at<double>(1) = 1;
                bba_ED.at<double>(3) = 1;
            } else if(param >= 0.875 - e && param < 0.875) {
                bba_ND.at<double>(2) = 1;
                bba_SD.at<double>(2) = 1;
                bba_MD.at<double>(2) = 1;
                bba_VD.at<double>(1) = (param - 0.875)/(-e);
                bba_VD.at<double>(3) = 1 - (param - 0.875)/(-e);
                bba_ED.at<double>(1) = (param - 0.875)/e;
                bba_ED.at<double>(3) = 1 - (param - 0.875)/e;
            } else if(param >= 0.875 && param < 0.875 + e) {
                bba_ND.at<double>(2) = 1;
                bba_SD.at<double>(2) = 1;
                bba_MD.at<double>(2) = 1;
                bba_VD.at<double>(2) = 1 - (param - 0.875 - e)/(-e);
                bba_VD.at<double>(3) = (param - 0.875 - e)/(-e);
                bba_ED.at<double>(2) = (param - 0.875)/e;
                bba_ED.at<double>(3) = 1 - (param -0.875)/e; 
            } else if(param >= 0.875 + e && param <= 1) {
                bba_ND.at<double>(2) = 1;
                bba_SD.at<double>(2) = 1;
                bba_MD.at<double>(2) = 1;
                bba_VD.at<double>(2) = 1;
                bba_ED.at<double>(1) = 1;
            } else {
                std::cout << "ERROR: Lane Feature Modeling!" << std::endl;
            }
            
            cv::Mat bba_temp(5, 4, CV_64F);
            bba_ND.copyTo(bba_temp.row(0));
            bba_SD.copyTo(bba_temp.row(1));
            bba_MD.copyTo(bba_temp.row(2));
            bba_VD.copyTo(bba_temp.row(3));
            bba_ED.copyTo(bba_temp.row(4));
    
//            std::cout << "Modeling Lane" << std::endl;
//            PrintMat(bba_temp);
            bba.push_back(bba_temp);
        }
    }
    
    void Model_ORD_TEF_Face(const FaceFeature *faceFeatures, 
                            const double e, std::vector<cv::Mat> &bba)
    {
        std::vector<double> params;
        params.push_back(faceFeatures->PERCLOS);
        params.push_back(faceFeatures->MICROSLEEP);
        params.push_back(faceFeatures->BLINK);
        
        int nbr_src = (int)params.size();
        for (int i = 0; i < nbr_src; i++)
        {
            cv::Mat bba_ND(1,4,CV_64F, cv::Scalar::all(0));
            cv::Mat bba_SD(1,4,CV_64F, cv::Scalar::all(0));
            cv::Mat bba_MD(1,4,CV_64F, cv::Scalar::all(0));
            cv::Mat bba_VD(1,4,CV_64F, cv::Scalar::all(0));
            cv::Mat bba_ED(1,4,CV_64F, cv::Scalar::all(0));
            
            double param = params.at(i);
            if (param < 0.125 - e) {
                bba_ND.at<double>(1) = 1; //[0 1 0 0] 
                bba_SD.at<double>(2) = 1; //[0 0 1 0]
                bba_MD.at<double>(2) = 1; //[0 0 1 0]
                bba_VD.at<double>(2) = 1; //[0 0 1 0]
                bba_ED.at<double>(2) = 1; //[0 0 1 0]
            } else if( param >= 0.125 - e && param < 0.125) {
                bba_ND.at<double>(1) = (param - 0.125)/(-e);
                bba_ND.at<double>(3) = 1 - (param - 0.125)/(-e);
                bba_SD.at<double>(2) = 1 - (param - 0.125 + e)/e;
                bba_SD.at<double>(3) = (param - 0.125 + e)/e;
                bba_MD.at<double>(2) = 1;
                bba_VD.at<double>(2) = 1;
                bba_ED.at<double>(2) = 1;
            } else if( param >= 0.125 && param < 0.125 + e) {
                bba_ND.at<double>(2) = 1 - (param - 0.125 - e)/(-e);
                bba_ND.at<double>(3) = (param - 0.125 - e)/(-e);
                bba_SD.at<double>(1) = (param - 0.125)/e;
                bba_SD.at<double>(3) = 1 - (param - 0.125)/e;
                bba_MD.at<double>(2) = 1;
                bba_VD.at<double>(2) = 1;
                bba_ED.at<double>(2) = 1;
            } else if(param >= 0.125 + e && param < 0.375 - e) {
                bba_ND.at<double>(2) = 1;
                bba_SD.at<double>(1) = 1;
                bba_MD.at<double>(2) = 1;
                bba_VD.at<double>(2) = 1;
                bba_ED.at<double>(2) = 1;
            } else if(param >= 0.375 - e && param < 0.375) {
                bba_ND.at<double>(2) = 1;
                bba_SD.at<double>(1) = (param - 0.375)/(-e);
                bba_SD.at<double>(3) = 1 - (param - 0.375)/(-e);
                bba_MD.at<double>(2) = 1 - (param - 0.375 + e)/e;
                bba_MD.at<double>(3) = (param - 0.375 + e)/e;
                bba_VD.at<double>(2) = 1;
                bba_ED.at<double>(2) = 1;
            } else if(param >= 0.375 && param < 0.375 + e) {
                bba_ND.at<double>(2) = 1;
                bba_SD.at<double>(2) = 1 - (param - 0.375 -e)/(-e);
                bba_SD.at<double>(3) = (param - 0.375 - e)/(-e);
                bba_MD.at<double>(1) = (param - 0.375)/e;
                bba_MD.at<double>(3) = 1 - (param - 0.375)/e;
                bba_VD.at<double>(2) = 1;
                bba_ED.at<double>(2) = 1;
            } else if(param >= 0.375 + e && param < 0.625 - e) {
                bba_ND.at<double>(2) = 1;
                bba_SD.at<double>(2) = 1;
                bba_MD.at<double>(1) = 1;
                bba_VD.at<double>(2) = 1;
                bba_ED.at<double>(2) = 1;
            } else if(param >= 0.625 - e && param < 0.625) {
                bba_ND.at<double>(2) = 1;
                bba_SD.at<double>(2) = 1;
                bba_MD.at<double>(1) = (param - 0.625)/(-e);
                bba_MD.at<double>(3) = 1 - (param - 0.625)/(-e);
                bba_VD.at<double>(2) = 1 - (param - 0.625 + e)/e;
                bba_VD.at<double>(3) = (param - 0.625 + e)/e;
                bba_ED.at<double>(2) = 1;
            } else if(param >= 0.625 && param < 0.625 + e) {
                bba_ND.at<double>(2) = 1;
                bba_SD.at<double>(2) = 1;
                bba_MD.at<double>(2) = 1 - (param - 0.625 - e)/(-e);
                bba_MD.at<double>(3) = (param - 0.625 - e)/(-e);
                bba_VD.at<double>(1) = (param - 0.625)/e;
                bba_VD.at<double>(3) = 1 - (param - 0.625)/e;
                bba_ED.at<double>(2) = 1;
            } else if(param >= 0.625 + e && param < 0.875 - e) {
                bba_ND.at<double>(2) = 1;
                bba_SD.at<double>(2) = 1;
                bba_MD.at<double>(2) = 1;
                bba_VD.at<double>(1) = 1;
                bba_ED.at<double>(2) = 1;
            } else if(param >= 0.875 - e && param < 0.875) {
                bba_ND.at<double>(2) = 1;
                bba_SD.at<double>(2) = 1;
                bba_MD.at<double>(2) = 1;
                bba_VD.at<double>(1) = (param - 0.875)/(-e);
                bba_VD.at<double>(3) = 1 - (param - 0.875)/(-e);
                bba_ED.at<double>(2) = 1 - (param - 0.875 + e)/e;
                bba_ED.at<double>(3) = (param - 0.875 + e)/e;
            } else if(param >= 0.875 && param < 0.875 + e) {
                bba_ND.at<double>(2) = 1;
                bba_SD.at<double>(2) = 1;
                bba_MD.at<double>(2) = 1;
                bba_VD.at<double>(2) = 1 - (param - 0.875 - e)/(-e);
                bba_VD.at<double>(3) = (param - 0.875 - e)/(-e);
                bba_ED.at<double>(2) = (param - 0.875)/e;
                bba_ED.at<double>(3) = 1 - (param-0.875)/e;
            } else if(param >= 0.875 + e && param <= 1) {
                bba_ND.at<double>(2) = 1;
                bba_SD.at<double>(2) = 1;
                bba_MD.at<double>(2) = 1;
                bba_VD.at<double>(2) = 1;
                bba_ED.at<double>(1) = 1;
            } else {
                std::cout << "ERROR: Face Feature Modeling!" << std::endl;
            }
            
//            std::cout << "ND,SD,MD,VD,ED" << std::endl;
//            PrintMat(bba_ND);
//            PrintMat(bba_SD);
//            PrintMat(bba_MD);
//            PrintMat(bba_VD);
//            PrintMat(bba_ED);


            cv::Mat bba_temp(5, 4, CV_64F, cv::Scalar::all(0));
            bba_ND.copyTo(bba_temp.row(0));
            bba_SD.copyTo(bba_temp.row(1));
            bba_MD.copyTo(bba_temp.row(2));
            bba_VD.copyTo(bba_temp.row(3));
            bba_ED.copyTo(bba_temp.row(4));
            
//            std::cout << "Modeling Face" << std::endl;
//            PrintMat(bba_temp);
            bba.push_back(bba_temp);
        }
    }
    
/********************************************************/
//! Refer to the Membership Functions
/********************************************************/
    double getTriangularMass(const double &x, const double &a1, const double &a2)
    {
        assert(a1 != a2);
        double mass;
        if(x >= a1 && x <= a2)
            mass = (x - a2) / (a1 - a2);
        else 
            mass = 0;
        return mass;
    }
    
    double getGaussianMass(const double &x, const double &k, const double &a)
    {
        return exp( -k * pow((x-a),2) ); 
    }
    
    
    //! Generate each feature corresponding to drowsiness
    //! All data sampled in a 60-s interval window
    void Model_ORD_Lane(const LaneFeature *laneFeatures, const int &massType, const int &nbr_hyp, cv::Mat &bba_lane, double &mLATSD)
    {
        double LATSD_Baseline = laneFeatures->LATSD_Baseline; //!the Median Value //0.3 before ??
        double LATSD = laneFeatures->LATSD - LATSD_Baseline;
		LATSD = LATSD < 0 ? 0 : LATSD;
        mLATSD = LATSD;//Record the mass value
        
        double rel_LATSD = 0.71, rel_LANEX = 0.58, rel_TLCF_2s = 0.66,
        rel_TLCF_halfs = 0.57;//CONSTANT
        
        switch (massType)
        {
/*****************************************************************/
//! LATSD relative to DROWSY LEVEL
//! Here, we referred to the paper(Ingre 2006)
//! It presents the LATSD relative to KSS(1--9)
//! 0.19(KSS 1), 0.25(KSS 5) 0.28(KSS 7), 036(KSS 8), 0.47(KSS 9)
//! Thus, to ORD, ND:0--0.25, SD:0.25--0.28, MD:0.28--0.36
//! VD: 0.36--0.47, ED: 0.47--
/*****************************************************************/
            case CONFIDENCE:
                if(LATSD < 0.25) {                                    
                    bba_lane.at<double>(0, ND) = rel_LATSD;      //ND
                } else if(LATSD >= 0.25 && LATSD < 0.28) { 
                    bba_lane.at<double>(0, SD) = rel_LATSD;      //SD
                } else if(LATSD >= 0.28 && LATSD < 0.36) {              
                    bba_lane.at<double>(0, MD) = rel_LATSD;      //MD
                } else if(LATSD >= 0.36 && LATSD < 0.47) { 
                    bba_lane.at<double>(0, VD) = rel_LATSD;      //VD
                } else if(LATSD >= 0.47) { 
                    bba_lane.at<double>(0, ED) = rel_LATSD;      //ED
                }
                bba_lane.at<double>(0, ALL) = 1 - rel_LATSD;     //{ND,SD,MD,VD,ED}
                break;
                
/****************************************************************/
//! Triangular Mass Function, e.g. assign {ND} and {ND,SD} or {ND,SD,MD,VD,ED}
/****************************************************************/
            case TRIANGULAR:
                if(LATSD >= 0 && LATSD < 0.125) {//down
                    bba_lane.at<double>(0, ND) = getTriangularMass(LATSD, 0, 0.125);//ND
//                    bba_lane.at<double>(0, ND_SD) = 1 - getTriangularMass(LATSD, 0, 0.125);// {ND,SD} 
                    bba_lane.at<double>(0, ALL) = 1 - getTriangularMass(LATSD, 0, 0.125);//{ND,SD,MD,VD,ED}
                } 
                else if( LATSD >= 0.125 && LATSD < 0.25) {//up
                    bba_lane.at<double>(0, SD) = getTriangularMass(LATSD, 0.25, 0.125);//SD
//                    bba_lane.at<double>(0, ND_SD) = 1 - getTriangularMass(LATSD, 0.25, 0.125);//{ND,SD}
                    bba_lane.at<double>(0, ALL) = 1 - getTriangularMass(LATSD, 0.25, 0.125);//{ND,SD,MD,VD,ED}
                } 
                else if( LATSD >= 0.25 && LATSD < 0.375) {//down
                    bba_lane.at<double>(0, SD) = getTriangularMass(LATSD, 0.25, 0.375);//SD
//                    bba_lane.at<double>(0, SD_MD) = 1 - getTriangularMass(LATSD, 0.25, 0.375);//{SD,MD}
                    bba_lane.at<double>(0, ALL) = 1 - getTriangularMass(LATSD, 0.25, 0.375);//{ND,SD,MD,VD,ED}
                } 
                else if(LATSD >= 0.375 && LATSD < 0.5) {//up
                    bba_lane.at<double>(0, MD) = getTriangularMass(LATSD, 0.5, 0.375);//MD
//                    bba_lane.at<double>(0, SD_MD) = 1- getTriangularMass(LATSD, 0.5, 0.375);//{SD,MD}
                    bba_lane.at<double>(0, ALL) = 1- getTriangularMass(LATSD, 0.5, 0.375);//{ND,SD,MD,VD,ED}
                } 
                else if(LATSD >= 0.5 && LATSD < 0.625) {//down
                    bba_lane.at<double>(0, MD) = getTriangularMass(LATSD, 0.5, 0.625);//MD
//                    bba_lane.at<double>(0, MD_VD) = 1- getTriangularMass(LATSD, 0.5, 0.625);//{MD,VD}
                    bba_lane.at<double>(0, ALL) = 1- getTriangularMass(LATSD, 0.5, 0.625);//{ND,SD,MD,VD,ED}
                } 
                else if(LATSD >= 0.625 && LATSD < 0.75) {//up
                    bba_lane.at<double>(0, VD) = getTriangularMass(LATSD, 0.75, 0.625);//VD
//                    bba_lane.at<double>(0, MD_VD) = 1 - getTriangularMass(LATSD, 0.75, 0.625);//{MD,VD}
                    bba_lane.at<double>(0, ALL) = 1 - getTriangularMass(LATSD, 0.75, 0.625);//{ND,SD,MD,VD,ED}
                } 
                else if(LATSD >= 0.75 && LATSD < 0.875) {//down
                    bba_lane.at<double>(0, VD) = getTriangularMass(LATSD, 0.75, 0.875);//VD
//                    bba_lane.at<double>(0, VD_ED) = 1 - getTriangularMass(LATSD, 0.75, 0.875);//{VD,ED}
                    bba_lane.at<double>(0, ALL) = 1 - getTriangularMass(LATSD, 0.75, 0.875);//{ND,SD,MD,VD,ED}
                } 
                else if(LATSD >= 0.875 && LATSD <= 1) {//up
                    bba_lane.at<double>(0, ED) = getTriangularMass(LATSD, 1, 0.875);//ED
//                    bba_lane.at<double>(0, VD_ED) = 1 - getTriangularMass(LATSD, 1, 0.875);//{VD,ED};
                    bba_lane.at<double>(0, ALL) = 1 - getTriangularMass(LATSD, 1, 0.875);//{ND,SD,MD,VD,ED}
                } 
                else {
                    std::cerr << "#ERROR: LATSD doesn't have the right number" << std::endl;
                }
                break;
        
/****************************************************************/
//! Gaussian Mass Function, e.g. assign {ND}, {SD} and {ND,SD}
/****************************************************************/
            case GAUSSIAN:
                if(LATSD >= 0 && LATSD < 0.25) {
                    bba_lane.at<double>(0, ND) = getGaussianMass(LATSD, k, 0);//ND
                    bba_lane.at<double>(0, SD) = getGaussianMass(LATSD, k, 0.25);//SD
//                    bba_lane.at<double>(0, ND_SD) = 1 - getGaussianMass(LATSD, k, 0) - getGaussianMass(LATSD, k, 0.25); //{ND,SD}
                    bba_lane.at<double>(0, ALL) = 1 - getGaussianMass(LATSD, k, 0) - getGaussianMass(LATSD, k, 0.25);//{ND,SD,MD,VD,ED}
                } 
                else if(LATSD >= 0.25 && LATSD < 0.5) {
                    bba_lane.at<double>(0, SD) = getGaussianMass(LATSD, k, 0.25);//SD
                    bba_lane.at<double>(0, MD) = getGaussianMass(LATSD, k, 0.5);//MD
//                    bba_lane.at<double>(0, SD_MD) = 1 - getGaussianMass(LATSD, k, 0.25) - getGaussianMass(LATSD, k, 0.5);//{SD,MD}
                    bba_lane.at<double>(0, ALL) = 1 - getGaussianMass(LATSD, k, 0.25) - getGaussianMass(LATSD, k, 0.5);//{ND,SD,MD,VD,ED}
                } 
                else if(LATSD >= 0.5 && LATSD < 0.75) {
                    bba_lane.at<double>(0, MD) = getGaussianMass(LATSD, k, 0.5);//MD
                    bba_lane.at<double>(0, VD) = getGaussianMass(LATSD, k, 0.75);//VD
//                    bba_lane.at<double>(0, MD_VD) = 1 - getGaussianMass(LATSD, k, 0.5) - getGaussianMass(LATSD, k, 0.75);//{MD,VD}
                    bba_lane.at<double>(0, ALL) = 1 - getGaussianMass(LATSD, k, 0.5) - getGaussianMass(LATSD, k, 0.75); //{ND,SD,MD,VD,ED}
                } 
                else if(LATSD >= 0.75 && LATSD <= 1) {
                    bba_lane.at<double>(0, VD) = getGaussianMass(LATSD, k, 0.75);//VD
                    bba_lane.at<double>(0, ED) = getGaussianMass(LATSD, k, 1);//ED
//                    bba_lane.at<double>(0, VD_ED) = 1 - getGaussianMass(LATSD, k, 0.75) - getGaussianMass(LATSD, k, 1);//{VD,ED}
                    bba_lane.at<double>(0, ALL) = 1 - getGaussianMass(LATSD, k, 0.75) - getGaussianMass(LATSD, k, 1); //{ND,SD,MD,VD,ED}
                } 
                else {
                    std::cerr << "#ERROR: LATSD doesn't have the right number" << std::endl;
                }
                break;
                
            default:
                break;
        }//end switch
    }//end Model_ORD_Lane
    
    
    void Model_ORD_Face(const FaceFeature *faceFeatures, const int &massType,const int &nbr_hyp, cv::Mat &bba_face, double &mPERCLOS, double &mBLINK)
    {
        double PERCLOS_Baseline = 0.03; // Minimum in series
        double PERCLOS = faceFeatures->PERCLOS - PERCLOS_Baseline;
        double PERCLOS_MAX = 0.444 - PERCLOS_Baseline;
		PERCLOS = PERCLOS < 0 ? 0 : PERCLOS; 
        mPERCLOS = PERCLOS;

        double BLINK_Baseline = 12;
        double BLINK = faceFeatures->BLINK - BLINK_Baseline;
        double BLINK_MAX = 30;
		BLINK = BLINK < 0 ? 0 : BLINK;
		BLINK = BLINK /BLINK_MAX;
		BLINK = BLINK > 1 ? 1 : BLINK;
        mBLINK = BLINK;
        
        
        double rel_PERCLOS = 0.8, rel_BLINK = 0.8;
        switch (massType)
        {
            case CONFIDENCE:
/*****************************************************************/
//! PERCLOS relative to DROWSY LEVEL
/*****************************************************************/
                if(PERCLOS < PERCLOS_MAX/8.0) {                                             
                    bba_face.at<double>(0, ND) = rel_PERCLOS; //ND
                    bba_face.at<double>(0, ND_SD) = 1 - rel_PERCLOS; //ND_SD
                } 
                else if(PERCLOS >= PERCLOS_MAX/8.0 && PERCLOS < PERCLOS_MAX*2.0/8.0) {      
                    bba_face.at<double>(0, SD) = rel_PERCLOS; //SD
                    bba_face.at<double>(0, ND_SD) = 1 - rel_PERCLOS; //ND_SD
                } 
                else if(PERCLOS >= PERCLOS_MAX*2.0/8.0 && PERCLOS < PERCLOS_MAX*3.0/8.0) {
                    bba_face.at<double>(0, SD) = rel_PERCLOS; //SD
                    bba_face.at<double>(0, SD_MD) = 1 - rel_PERCLOS; //SD_MD
                }
                else if(PERCLOS >= PERCLOS_MAX*3.0/8.0 && PERCLOS < PERCLOS_MAX*4.0/8.0) {
                    bba_face.at<double>(0, MD) = rel_PERCLOS; //SD
                    bba_face.at<double>(0, SD_MD) = 1 - rel_PERCLOS; //SD_MD
                }
                else if(PERCLOS >= PERCLOS_MAX*4.0/8.0 && PERCLOS < PERCLOS_MAX*5.0/8.0) {
                    bba_face.at<double>(0, MD) = rel_PERCLOS; //SD
                    bba_face.at<double>(0, MD_VD) = 1 - rel_PERCLOS; //MD_VD
                }
                else if(PERCLOS >= PERCLOS_MAX*5.0/8.0 && PERCLOS < PERCLOS_MAX*6.0/8.0) {  
                    bba_face.at<double>(0, VD) = rel_PERCLOS; //VD
                    bba_face.at<double>(0, MD_VD) = 1 - rel_PERCLOS; //MD_VD
                } 
                else if(PERCLOS >= PERCLOS_MAX*6.0/8.0 && PERCLOS < PERCLOS_MAX*7.0/8.0) {  
                    bba_face.at<double>(0, VD) = rel_PERCLOS; //VD
                    bba_face.at<double>(0, VD_ED) = 1 - rel_PERCLOS; //VD_ED
                } 
                else if (PERCLOS >= PERCLOS_MAX*7.0/8.0) {                                  
                    bba_face.at<double>(0, ED) = rel_PERCLOS; //ED
                    bba_face.at<double>(0, VD_ED) = 1 - rel_PERCLOS; //VD_ED
                }
                //bba.at<double>(1, ALL) = 1 - rel_PERCLOS;
        
/*****************************************************************/
//! BLINK relative to DROWSY LEVEL
/*****************************************************************/
                if(BLINK < 0.125) {                                              
                    bba_face.at<double>(1, ND) = rel_BLINK; //ND
                    bba_face.at<double>(1, ND_SD) = 1 - rel_BLINK;   //ND_SD
                }  
                else if(BLINK >= 0.125 && BLINK < 0.25) {        
                    bba_face.at<double>(1, SD) = rel_BLINK; //SD
                    bba_face.at<double>(1, ND_SD) = 1 - rel_BLINK;   //ND_SD
                }
                else if(BLINK >= 0.25 && BLINK < 0.375) {        
                    bba_face.at<double>(1, SD) = rel_BLINK; //SD
                    bba_face.at<double>(1, SD_MD) = 1 - rel_BLINK;   //SD_MD
                } 
                else if(BLINK >= 0.375 && BLINK < 0.5) {    
                    bba_face.at<double>(1, MD) = rel_BLINK; //MD
                    bba_face.at<double>(1, SD_MD) = 1 - rel_BLINK;   //SD_MD
                } 
                else if(BLINK >= 0.5 && BLINK < 0.625) {    
                    bba_face.at<double>(1, MD) = rel_BLINK; //MD
                    bba_face.at<double>(1, MD_VD) = 1 - rel_BLINK;   //MD_VD
                }
                else if(BLINK >= 0.625 && BLINK < 0.75) {    
                    bba_face.at<double>(1, VD) = rel_BLINK; //VD
                    bba_face.at<double>(1, MD_VD) = 1 - rel_BLINK;   //MD_VD
                } 
                else if(BLINK >= 0.75 && BLINK < 0.875) {    
                    bba_face.at<double>(1, VD) = rel_BLINK; //VD
                    bba_face.at<double>(1, VD_ED) = 1 - rel_BLINK;   //VD_ED
                }
                else if(BLINK >= 0.875) {                                 
                    bba_face.at<double>(1, ED) = rel_BLINK; //ED
                    bba_face.at<double>(1, VD_ED) = 1 - rel_BLINK;   //VD_ED
                }
                //bba_face.at<double>(1, ALL) = 1 - rel_BLINK;
                break;
                
            case TRIANGULAR:
                //! PERCLOS
                if(PERCLOS >= 0 && PERCLOS < 0.125) {//down
                    bba_face.at<double>(0, ND) = getTriangularMass(PERCLOS, 0, 0.125);//ND
                    bba_face.at<double>(0, ND_SD) = 1 - getTriangularMass(PERCLOS, 0, 0.125);// {ND,SD} 
                } 
                else if( PERCLOS >= 0.125 && PERCLOS < 0.25) {//up
                    bba_face.at<double>(0, SD) = getTriangularMass(PERCLOS, 0.25, 0.125);//SD
                    bba_face.at<double>(0, ND_SD) = 1 - getTriangularMass(PERCLOS, 0.25, 0.125);//{ND,SD}
                } 
                else if( PERCLOS >= 0.25 && PERCLOS < 0.375) {//down
                    bba_face.at<double>(0, SD) = getTriangularMass(PERCLOS, 0.25, 0.375);//SD
                    bba_face.at<double>(0, SD_MD) = 1 - getTriangularMass(PERCLOS, 0.25, 0.375);//{SD,MD}
                } 
                else if(PERCLOS >= 0.375 && PERCLOS < 0.5) {//up
                    bba_face.at<double>(0, MD) = getTriangularMass(PERCLOS, 0.5, 0.375);//MD
                    bba_face.at<double>(0, SD_MD) = 1- getTriangularMass(PERCLOS, 0.5, 0.375);//{SD,MD}
                } 
                else if(PERCLOS >= 0.5 && PERCLOS < 0.625) {//down
                    bba_face.at<double>(0, MD) = getTriangularMass(PERCLOS, 0.5, 0.625);//MD
                    bba_face.at<double>(0, MD_VD) = 1- getTriangularMass(PERCLOS, 0.5, 0.625);//{MD,VD}
                } 
                else if(PERCLOS >= 0.625 && PERCLOS < 0.75) {//up
                    bba_face.at<double>(0, VD) = getTriangularMass(PERCLOS, 0.75, 0.625);//VD
                    bba_face.at<double>(0, MD_VD) = 1 - getTriangularMass(PERCLOS, 0.75, 0.625);//{MD,VD}
                } 
                else if(PERCLOS >= 0.75 && PERCLOS < 0.875) {//down
                    bba_face.at<double>(0, VD) = getTriangularMass(PERCLOS, 0.75, 0.875);//VD
                    bba_face.at<double>(0, VD_ED) = 1 - getTriangularMass(PERCLOS, 0.75, 0.875);//{VD,ED}
                } 
                else if(PERCLOS >= 0.875 && PERCLOS <= 1) {//up
                    bba_face.at<double>(0, ED) = getTriangularMass(PERCLOS, 1, 0.875);//ED
                    bba_face.at<double>(0, VD_ED) = 1 - getTriangularMass(PERCLOS, 1, 0.875);//{VD,ED};
                } 
                else {
                    std::cerr << "#ERROR: PERCLOS doesn't have the right number" << std::endl;
                }
               
                //!BLINK
                if(BLINK >= 0 && BLINK < 0.125) {//down
                    bba_face.at<double>(1, ND) = getTriangularMass(BLINK, 0, 0.125);//ND
                    bba_face.at<double>(1, ND_SD) = 1 - getTriangularMass(BLINK, 0, 0.125);// {ND,SD} 
                } 
                else if( BLINK >= 0.125 && BLINK < 0.25) {//up
                    bba_face.at<double>(1, SD) = getTriangularMass(BLINK, 0.25, 0.125);//SD
                    bba_face.at<double>(1, ND_SD) = 1 - getTriangularMass(BLINK, 0.25, 0.125);//{ND,SD}
                } 
                else if( BLINK >= 0.25 && BLINK < 0.375) {//down
                    bba_face.at<double>(1, SD) = getTriangularMass(BLINK, 0.25, 0.375);//SD
                    bba_face.at<double>(1, SD_MD) = 1 - getTriangularMass(BLINK, 0.25, 0.375);//{SD,MD}
                } 
                else if( BLINK >= 0.375 && BLINK < 0.5) {//up
                    bba_face.at<double>(1, MD) = getTriangularMass(BLINK, 0.5, 0.375);//MD
                    bba_face.at<double>(1, SD_MD) = 1- getTriangularMass(BLINK, 0.5, 0.375);//{SD,MD}
                } 
                else if( BLINK >= 0.5 && BLINK < 0.625) {//down
                    bba_face.at<double>(1, MD) = getTriangularMass(BLINK, 0.5, 0.625);//MD
                    bba_face.at<double>(1, MD_VD) = 1- getTriangularMass(BLINK, 0.5, 0.625);//{MD,VD}
                } 
                else if( BLINK >= 0.625 && BLINK < 0.75) {//up
                    bba_face.at<double>(1, VD) = getTriangularMass(BLINK, 0.75, 0.625);//VD
                    bba_face.at<double>(1, MD_VD) = 1 - getTriangularMass(BLINK, 0.75, 0.625);//{MD,VD}
                } 
                else if( BLINK >= 0.75 && BLINK < 0.875) {//down
                    bba_face.at<double>(1, VD) = getTriangularMass(BLINK, 0.75, 0.875);//VD
                    bba_face.at<double>(1, VD_ED) = 1 - getTriangularMass(BLINK, 0.75, 0.875);//{VD,ED}
                } 
                else if( BLINK >= 0.875 && BLINK <= 1) {//up
                    bba_face.at<double>(1, ED) = getTriangularMass(BLINK, 1, 0.875);//ED
                    bba_face.at<double>(1, VD_ED) = 1 - getTriangularMass(BLINK, 1, 0.875);//{VD,ED};
                } 
                else {
                    std::cerr << "#ERROR: BLINK doesn't have the right number" << std::endl;
                }
                break;
                
            case GAUSSIAN:
                //! PERCLOS
                if(PERCLOS >= 0 && PERCLOS < 0.25) {//down
                    bba_face.at<double>(0, ND) = getGaussianMass(PERCLOS, k, 0);     //ND
                    bba_face.at<double>(0, SD) = getGaussianMass(PERCLOS, k, 0.25);  //SD
                    bba_face.at<double>(0, ND_SD) = 1 - getGaussianMass(PERCLOS, k, 0) - getGaussianMass(PERCLOS, k, 0.25);   //{ND,SD}
                } 
                else if(PERCLOS >= 0.25 && PERCLOS < 0.5) {
                    bba_face.at<double>(0, SD) = getGaussianMass(PERCLOS, k, 0.25);  //SD
                    bba_face.at<double>(0, MD) = getGaussianMass(PERCLOS, k, 0.5);   //MD
                    bba_face.at<double>(0, SD_MD) = 1 - getGaussianMass(PERCLOS, k, 0.25) - getGaussianMass(PERCLOS, k, 0.5);    //{SD,MD}
                    
                } 
                else if(PERCLOS >= 0.5 && PERCLOS < 0.75) {
                    bba_face.at<double>(0, MD) = getGaussianMass(PERCLOS, k, 0.5);   //MD
                    bba_face.at<double>(0, VD) = getGaussianMass(PERCLOS, k, 0.75);  //VD
                    bba_face.at<double>(0, MD_VD) = 1 - getGaussianMass(PERCLOS, k, 0.5) - getGaussianMass(PERCLOS, k, 0.75);   //{MD,VD}
                } 
                else if(PERCLOS >= 0.75 && PERCLOS <= 1) {
                    bba_face.at<double>(0, VD) = getGaussianMass(PERCLOS, k, 0.75);  //VD
                    bba_face.at<double>(0, ED) = getGaussianMass(PERCLOS, k, 1);     //ED
                    bba_face.at<double>(0, VD_ED) = 1 - getGaussianMass(PERCLOS, k, 0.75) - getGaussianMass(PERCLOS, k, 1);      //{VD,ED}
                } 
                else {
                    std::cerr << "#ERROR: PERCLOS doesn't have the right number" << std::endl;
                }
                break;
                
                //!BLINK
                if(BLINK >= 0 && BLINK < 0.25) {//down
                    bba_face.at<double>(1, ND) = getGaussianMass(BLINK, k, 0);     //ND
                    bba_face.at<double>(1, SD) = getGaussianMass(BLINK, k, 0.25);  //SD
                    bba_face.at<double>(1, ND_SD) = 1 - getGaussianMass(BLINK, k, 0) - getGaussianMass(BLINK, k, 0.25);          //{ND,SD}
                } 
                else if(BLINK >= 0.25 && BLINK < 0.5) {
                    bba_face.at<double>(1, SD) = getGaussianMass(BLINK, k, 0.25);  //SD
                    bba_face.at<double>(1, MD) = getGaussianMass(BLINK, k, 0.5);   //MD
                    bba_face.at<double>(1, SD_MD) = 1 - getGaussianMass(BLINK, k, 0.25) - getGaussianMass(BLINK, k, 0.5);   //{SD,MD}
                    
                } 
                else if(BLINK >= 0.5 && BLINK < 0.75) {
                    bba_face.at<double>(1, MD) = getGaussianMass(BLINK, k, 0.5);   //MD
                    bba_face.at<double>(1, VD) = getGaussianMass(BLINK, k, 0.75);  //VD
                    bba_face.at<double>(1, MD_VD) = 1 - getGaussianMass(BLINK, k, 0.5) - getGaussianMass(PERCLOS, k, 0.75); //{MD,VD}
                } 
                else if(BLINK >= 0.75 && BLINK <= 1) {
                    bba_face.at<double>(1, VD) = getGaussianMass(BLINK, k, 0.75);  //VD
                    bba_face.at<double>(1, ED) = getGaussianMass(BLINK, k, 1);     //ED
                    bba_face.at<double>(1, VD_ED) = 1 - getGaussianMass(BLINK, k, 0.75) - getGaussianMass(BLINK, k, 1);      //{VD,ED}
                } 
                else {
                    std::cerr << "#ERROR: BLINK doesn't have the right number" << std::endl;
                }
                break;
        }//end switch
        
    }//end Model_ORD_Face
    
    
    //! Impared Driving 
    //! The model's framework is {-I, I, IGN}, different from the other one. 
    //! Each feature corresponding to drowsiness
    //! All data sampled in a 60-s interval window
    void Model_MultiORD_ST(const LaneFeature *laneFeatures, const FaceFeature *faceFeatures,
                           const int &massType, cv::Mat &bba_lane, cv::Mat &bba_face,
                           double &mLateralOffset, double &mMicroSleep)
    {
        /*****************************************************************/
        //! Lane Indicators
        /*****************************************************************/
        double LO = std::abs(laneFeatures->lateralOffset);
        mLateralOffset = LO;
        const double rel_LO = 0.5;
        
        switch (massType)
        {
            
            case CONFIDENCE:
                if(LO < 1) {                                    
                    bba_lane.at<double>(0, 1) = rel_LO;      //Non-Impaired
                } else { 
                    bba_lane.at<double>(0, 2) = rel_LO;      //Impaired
                }
                bba_lane.at<double>(0, 3) = 1 - rel_LO;     //Ignorance
                break;
                
            /****************************************************************/
            //! Triangular Mass Function, e.g. assign {ND} and {ND,SD} or {ND,SD,MD,VD,ED}
            /****************************************************************/
            case TRIANGULAR:
                if(LO < 1) {
                    bba_lane.at<double>(0, 1) = getTriangularMass(LO, 0, 0.125);//ND
//                    bba_lane.at<double>(0, ND_SD) = 1 - getTriangularMass(LO, 0, 0.125);// {ND,SD} 
                    bba_lane.at<double>(0, 3) = 1 - getTriangularMass(LO, 0, 0.125);//{ND,SD,MD,VD,ED}
                }
                else {//up
                    bba_lane.at<double>(0, 2) = getTriangularMass(LO, 1, 0.875);//ED
//                    bba_lane.at<double>(0, VD_ED) = 1 - getTriangularMass(LO, 1, 0.875);//{VD,ED};
                    bba_lane.at<double>(0, 3) = 1 - getTriangularMass(LO, 1, 0.875);//{ND,SD,MD,VD,ED}
                } 
                break;
                            
            default:
                break;
        }//end switch
        
        /*****************************************************************/
        //! Face Indicators
        /*****************************************************************/
        double MICROSLEEP = faceFeatures->MICROSLEEP;   
        mMicroSleep = MICROSLEEP;
        const double rel_MICROSLEEP = 0.5;
        
        switch (massType)
        {
            case CONFIDENCE:
                /*****************************************************************/
                //! MICROSLEEP relative to DROWSY LEVEL
                /*****************************************************************/
                if(MICROSLEEP <= 0.375) {                                             
                    bba_face.at<double>(0, 1) = rel_MICROSLEEP;     //Non-Impaired
                } else {      
                    bba_face.at<double>(0, 2) = rel_MICROSLEEP;     //Impaired
                } 
                bba_face.at<double>(0, 3) = 1 - rel_MICROSLEEP; 
                break;
                
            case TRIANGULAR:
                if( MICROSLEEP <= 0.375 ) {//down
                    bba_face.at<double>(0, 1) = getTriangularMass(MICROSLEEP, 0, 0.125);//ND
                    bba_face.at<double>(0, 3) = 1 - getTriangularMass(MICROSLEEP, 0, 0.125);// {ND,SD} 
                }
                else {//up
                    bba_face.at<double>(0, 2) = getTriangularMass(MICROSLEEP, 1, 0.875);//ED
                    bba_face.at<double>(0, 3) = 1 - getTriangularMass(MICROSLEEP, 1, 0.875);//{VD,ED};
                } 
                break;
                
            default:
                break;
        }//end switch
        
    }//end Model_MultiORD_ST
    
    void Model_SingleORD_ST(const LaneFeature *laneFeatures, const FaceFeature *faceFeatures,
                            const float &laneThreshold, const float &faceThreshold, 
                            cv::Mat &bba_lane, cv::Mat &bba_face,
                            double &mLateralOffset, double &mMicroSleep)
    {
        /*****************************************************************/
        //! Lane Indicators
        /*****************************************************************/
        double ccp = std::abs(laneFeatures->lateralOffset);
        mLateralOffset = ccp;
        const double rel_ccp = 0.8;
    
        if(ccp <= laneThreshold) {                                    
            bba_lane.at<double>(0, 1) = 0;      //Impaired
            bba_lane.at<double>(0, 2) = rel_ccp*(1-ccp/laneThreshold); //Non-Impaired
        } else { 
            bba_lane.at<double>(0, 1) = rel_ccp*(ccp-laneThreshold)/(1-laneThreshold); //Impaired
            bba_lane.at<double>(0, 2) = 0;      //Non-Impaired
        }
        bba_lane.at<double>(0, 3) = 1 - bba_lane.at<double>(0, 1) - bba_lane.at<double>(0, 2);//Ignorance
                      
        /*****************************************************************/
        //! Face Indicators
        /*****************************************************************/
        double MST = faceFeatures->MICROSLEEP;   
        mMicroSleep = MST;
        const double rel_MST = 0.8;
        
        if(MST <= faceThreshold) {                                             
            bba_face.at<double>(0, 1) = 0;     //Impaired : time > 1s (faceThreshold: 0.375)
            bba_face.at<double>(0, 2) = rel_MST*(1-MST/faceThreshold); 
        } else {      
            bba_face.at<double>(0, 1) = rel_MST*(MST-faceThreshold)/(1-faceThreshold); 
            bba_face.at<double>(0, 2) = 0;     //Non-Impaired
        } 
        bba_face.at<double>(0, 3) = 1 - bba_face.at<double>(0,1) - bba_face.at<double>(0,2); 
    }
    
}//namespace
