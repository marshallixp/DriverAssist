//
//  main_RunFusionCenter.cpp
//  RunFusionCenter1.2
//
//  Created by XUANPENG LI on 11/11/13.
//  Copyright (c) 2013 ESIEE-Amiens. All rights reserved.
//
#include "main_RunFusionCenter.h"
#include "Process_RunFusionCenter.h"
//! IPC
#include <sys/types.h>
#include <sys/msg.h>
#include <sys/ipc.h>
#include <errno.h>

#ifdef __cplusplus

extern enum CmbRule tagCmbRule; 
extern enum Decision tagDecision;

#ifdef __APPLE__
//! ORD RECORD
extern const char   FILE_FEATURE[]      = "/Users/xuanpengli/DriveAssist/Sim_Record/sim_feature_fusion.txt";
extern const char   FILE_RESULT_ORD[]   = "/Users/xuanpengli/DriveAssist/Sim_Record/sim_result_fusion.txt";
//! TEF RECORD 
extern const char   FILE_RESULT_TEF[]   = "/Users/xuanpengli/DriveAssist/Sim_Record/sim_result_fusion.txt";
extern const char   FILE_MEASURE[]      = "/Users/xuanpengli/DriveAssist/Sim_Record/sim_measure_fusion.txt";
extern const char   FILE_PREDICT[]      = "/Users/xuanpengli/DriveAssist/Sim_Record/sim_predict_fusion.txt";
extern const char   FILE_LANE_CMB[]     = "/Users/xuanpengli/DriveAssist/Sim_Record/sim_lane_cmb_fusion.txt";
extern const char   FILE_FACE_CMB[]     = "/Users/xuanpengli/DriveAssist/Sim_Record/sim_face_cmb_fusion.txt";    
extern const char   KEY_PATH[]          = "/Users/xuanpengli/DriveAssist/key.txt";
#elif defined __linux
//! ORD RECORD
extern const char   FILE_FEATURE[]      = "/home/lixp/DriveAssist/Sim_Record/sim_feature_fusion.txt";
extern const char   FILE_RESULT_ORD[]   = "/home/lixp/DriveAssist/Sim_Record/sim_result_fusion.txt";
//! TEF RECORD 
extern const char   FILE_RESULT_TEF[]   = "/home/lixp/DriveAssist/Sim_Record/sim_result_fusion.txt";
extern const char   FILE_MEASURE[]      = "/home/lixp/DriveAssist/Sim_Record/sim_measure_fusiontxt";
extern const char   FILE_PREDICT[]      = "/home/lixp/DriveAssist/Sim_Record/sim_predict_fusion.txt";
extern const char   FILE_LANE_CMB[]     = "/home/lixp/DriveAssist/Sim_Record/sim_lane_cmb_fusion.txt";
extern const char   FILE_FACE_CMB[]     = "/home/lixp/DriveAssist/Sim_Record/sim_face_cmb_fusion.txt";    
extern const char   KEY_PATH[]          = "/home/lixp/DriveAssist/key.txt";
#endif


extern const double SAMPLING_TIME       = 60; //sec

///Run application
extern const int    DEBUG_SHOW          = 1;

namespace RunFusionCenter {
    int CombineType     = Single_ORD;
    int TagRule_LT1     = Conjunctive;
    int TagRule_LT2     = DuboisPrade;
    int TagRule_ST      = Dempster;
    int TagDecision_LT  = BetP;
    int TagDecision     = BetP;
    double ccpThreshold_ST = 0.8;
    double mstThreshold_ST = 0.375; 

    char key;
    double delay = 1;
    
    int Process (int argc, const char * argv[])
    {
        int DATA_RECORD = atoi(argv[1]);
        std::ofstream resultFile, measureFile, predictFile, laneCmbFile, faceCmbFile, featureFile;
		std::ofstream resultFileORD;
        
        InitRecordData(featureFile, FILE_FEATURE, featureName, NUM_FEATURE);
        
        if (DATA_RECORD) {
            switch (CombineType) {
                case Single_ORD:
                    InitRecordData(resultFileORD, FILE_RESULT_ORD, resultName, NUM_RESULT);
                    break;
                    
                case Multi_Fusion:
                    InitRecordData(resultFileORD, FILE_RESULT_ORD, resultMultiName, NUM_MULTI);
                    break;
                    
                case Single_ORD_TEF:
                    InitRecordData(resultFile, FILE_RESULT_TEF, resultTEFName, NUM_TEF);
                    InitRecordData(measureFile, FILE_MEASURE, resultTEFName, NUM_TEF);
                    InitRecordData(predictFile, FILE_PREDICT, resultTEFName, NUM_TEF);
                    InitRecordData(laneCmbFile, FILE_LANE_CMB, resultTEFName, NUM_TEF);
                    InitRecordData(faceCmbFile, FILE_FACE_CMB, resultTEFName, NUM_TEF);
                    break;
                    
                case Multi_ORD:
                    InitRecordData(resultFileORD, FILE_RESULT_ORD, resultName, NUM_RESULT);
                    break;
                    
                default:
                    break;
            }
        }

        //! IPC
        key_t ipckey;
        struct { 
            long type; 
            char text[1024];
        } mymsg;
        /* Generate the ipc key */
        ipckey = ftok(KEY_PATH, 'a');
        if(ipckey == -1){
            printf("Key Error: %s\n", strerror(errno));
            exit(1);
        }
        
        int mq_id;
        mq_id = msgget(ipckey, 0);
        if (mq_id == -1)
        {
            mq_id = msgget(ipckey, IPC_CREAT| IPC_EXCL | 0666);
            printf("FusionCenter creates a new MQ %d\n", mq_id);
        }
        else {
            mq_id = msgget(ipckey, IPC_EXCL | 0666);
            printf("FusionCenter uses an existed MQ %d\n", mq_id);
        }
        if(mq_id==-1)  
        {  
            printf("Create Failed. Error: %s\n", strerror(errno));  
            exit(1);  
        }
        
        //! Parameters for fusion
        int laneFlag = 0, faceFlag = 0;
        std::deque<double> params;
        
        FusionCenter::LaneFeature *laneParams = new FusionCenter::LaneFeature[sizeof(FusionCenter::LaneFeature)];
        FusionCenter::FaceFeature *faceParams = new FusionCenter::FaceFeature[sizeof(FusionCenter::FaceFeature)];
        
        //! ORD Results
        std::vector<double> results;
        std::deque<std::vector<double> > resultsSeq;
        std::vector<double> results_ST;
        //! ORD_TEF
        std::vector<double> CUSUM(5,0);
        std::deque<double> conflictDeque[5];
        double model_init[] = {1, 0, 0, 0, 0};
        std::vector<double> model(model_init, model_init+5);
        
        //! Init state of BBA
        cv::Mat bba_cmb_TEF = (cv::Mat_<double>(5,4) << 
                               0, 1, 0, 0,
                               0, 0, 1, 0,
                               0, 0, 1, 0,
                               0, 0, 1, 0,
                               0, 0, 1, 0);
        cv::Mat bba_cmb_lane(5, 4, CV_64F), bba_cmb_face(5, 4, CV_64F);
        cv::Mat bba_cmb(5, 4, CV_64F, cv::Scalar::all(0));
        cv::Mat bba_predict(5, 4, CV_64F, cv::Scalar::all(0));
        
        //! Conflict
        std::vector<double> conflict;
        double drowsyLevel = 0;
        
        //! Record Mass
        double mLATSD = 0;
        double mPERCLOS = 0;
        double mBLINK = 0;
        
        //! Record results of MultiFusion
        double T = 0;
        double TuF = 0;
        double F = 0;
        
        int sampleIdx = 1;
        double intervalTime = 0, lastStartTime = 0;
        
        double initTime = (double)cv::getTickCount();
        
        cv::Mat debugShow = cv::Mat::zeros(300, 1000, CV_64FC3);
        //! Receive the message
        while (1) {
            double startTime = (double)cv::getTickCount();
            
            if(msgrcv(mq_id, &mymsg, sizeof(mymsg), 0, 0) == -1) {
                printf("msgrcv failed!\n"); 
                perror("error");
                exit(1);  
            }
            
            char *text = new char[300];
            // sprintf(text, "Server receives: %s", mymsg.text);
            // cv::putText(debugShow, text, cv::Point(0, 10), cv::FONT_HERSHEY_SIMPLEX, 0.3, CV_RGB(0,255,0));
            
            //! Decode the message queue
            FusionCenter::DecodeMsg(mymsg.text, params);
            
            if( (int)params.size() > 0){
                //! Update the params if receive the same kind data.
                //! Fuse the params if receive both data.
                if ((int)params.at(0) == 0) {
                    //std::cout << "Lane Data Received..." << std::endl;
                    laneFlag = 1;
                    
                    //! Construct the LaneFeature struct
                    params.pop_front();
                    laneParams->frame               = cvRound(params[0]);
                    laneParams->lateralOffset       = params[1];
                    laneParams->LATSD               = params[2];
                    laneParams->LATSD_Baseline      = params[3];
                    laneParams->LATMEAN             = params[4];
                    laneParams->LATMEAN_Baseline    = params[5];
                    laneParams->LANEDEV             = params[6];
                    laneParams->LANEDEV_Baseline    = params[7];
                    laneParams->LANEX               = params[8];
                    laneParams->TLC                 = params[9];
                    laneParams->TLC_2s              = cvRound(params[10]);
                    laneParams->TLCF_2s             = params[11];
                    laneParams->TLC_halfs           = cvRound(params[12]);
                    laneParams->TLCF_halfs          = params[13];
                    laneParams->TLC_min             = params[14];
                    laneParams->TOT                 = params[15];
                    params.clear();
                }
                else 
                {
                    //std::cout << "Face Data Received..." << std::endl;
                    faceFlag = 1;
                    
                    //! Construct the FaceFeature struct
                    params.pop_front();
                    faceParams->frame               = cvRound(params[0]);
                    faceParams->Height              = params[1];
                    faceParams->lastHeight          = params[2];
                    faceParams->Height_Baseline     = params[3];
                    faceParams->PERCLOS             = params[4];
                    faceParams->MICROSLEEPTime      = params[5];
                    faceParams->MICROSLEEP          = params[6];
                    faceParams->BLINK               = params[7];
                    faceParams->BLINK_Baseline      = params[8];
                    params.clear();
                }
            } 
            else 
            {
                std::cout << "Error: Data Wrong!" << std::endl;
            }

			intervalTime = (startTime - lastStartTime)/cv::getTickFrequency();
			lastStartTime = startTime;
            double pastTime = 0;
            
            //! Use flag to marker the new data.
            if(laneFlag == 1 && faceFlag == 1) 
            {
                debugShow = cv::Mat::zeros(300, 1000, CV_64FC3);
                
                sprintf(text, "@Fuse the Lane and Face Indicators...");
                cv::putText(debugShow, text, cv::Point(0, 10), cv::FONT_HERSHEY_SIMPLEX, 0.3, CV_RGB(0,255,0));
                
                
   //              // switch (CombineType) {
   //              //     case Single_ORD:
   //              //         //! Long-Term : Dempster
   //              //         //! Short-Term : Dempster
   //              //         FusionCenter::FusionCenter_SingleORD(laneParams, faceParams, ccpThreshold_ST, mstThreshold_ST, Dempster, Dempster, BetP, intervalTime, resultsSeq, drowsyLevel, results_ST, conflict, debugShow, mLATSD, mPERCLOS, mBLINK);
   //              //         break;
                        
   //              //     case Multi_Fusion:
   //              //         //! Long-Term : CautiousConj, DuboisPrade
   //              //         //! Short-Term : Direct Fusion
   //              //         FusionCenter::FusionCenter_MultiFusion(laneParams, faceParams, TagRule_LT1, TagRule_LT2, TagDecision_LT, TagRule_ST, intervalTime, results, mLATSD, mPERCLOS, mBLINK, T, TuF, F, debugShow);
   //              //          break;
                            
   //              //     case  Single_ORD_TEF:
   //              //         FusionCenter::FusionCenter_ORD_TEF(laneParams, faceParams, CautiousConjNorm_Conjunctive, BetP, conflictDeque, CUSUM, model, bba_cmb_TEF, bba_cmb, bba_predict, bba_cmb_lane, bba_cmb_face, drowsyLevel);
   //              //         break;
                        
   //              //     case Multi_ORD:
   //              //         FusionCenter::FusionCenter_MultiORD(laneParams, faceParams, TagRule_LT1, TagRule_LT2, TagRule_ST, TagDecision, intervalTime, resultsSeq, drowsyLevel, results_ST, conflict, debugShow, mLATSD, mPERCLOS, mBLINK);
   //              //         break;
                        
   //              //     default:
   //              //         break; 
   //              // }//end switch
                
                pastTime = ((double)cv::getTickCount() - initTime)/cv::getTickFrequency();
                
                if(DATA_RECORD) {
                    RecordFeature(featureFile, sampleIdx, laneParams, faceParams, intervalTime, pastTime);
                    
   //                  switch (CombineType) {
   //                      case Single_ORD:
   //                          RecordResult(resultFileORD, sampleIdx, resultsSeq.back(), intervalTime, drowsyLevel, results_ST, conflict, mLATSD, mPERCLOS, mBLINK, pastTime);
   //                          break;
                            
   //                      case Multi_Fusion:
   //                          RecordMultiResult(resultFileORD, sampleIdx, results, intervalTime,mLATSD, mPERCLOS, mBLINK, T, TuF, F, pastTime);
   //                          break;
                            
   //                      case Single_ORD_TEF:
   //                          RecordTEFResult(resultFile, sampleIdx, bba_cmb_TEF);
   //                          RecordTEFResult(measureFile, sampleIdx, bba_cmb);
   //                          RecordTEFResult(predictFile, sampleIdx, bba_predict);
   //                          RecordTEFResult(laneCmbFile, sampleIdx, bba_cmb_lane);
   //                          RecordTEFResult(faceCmbFile, sampleIdx, bba_cmb_face);
   //                          break;

   //                      case Multi_ORD:
   //                          RecordResult(resultFileORD, sampleIdx, resultsSeq.back(), intervalTime, drowsyLevel, results_ST, conflict, mLATSD, mPERCLOS, mBLINK, pastTime);
   //                          break;
                            
   //                      default:
   //                          break;
   //                  }
                }
                
                //! Reset the Flag
                laneFlag = 0; 
                faceFlag = 0;
                
                sampleIdx++;
                
                double execTime = ((double)cv::getTickCount() - startTime)/cv::getTickFrequency();
                sprintf(text, "Process: %.2f Hz", 1.0/execTime);
                cv::putText(debugShow, text, cv::Point(0, 20), cv::FONT_HERSHEY_SIMPLEX, 0.3, CV_RGB(0,255,0));
                
                pastTime = ((double)cv::getTickCount() - initTime)/cv::getTickFrequency();
                sprintf(text, "FusionCenter Time: %.2f sec", pastTime);
                cv::putText(debugShow, text, cv::Point(0, debugShow.rows-5), cv::FONT_HERSHEY_SIMPLEX, 0.3, CV_RGB(0,255,0));
            } 
            else 
            {
                sprintf(text, "#Can't fuse, Wait for new data...");
                cv::putText(debugShow, text, cv::Point(0, 30), cv::FONT_HERSHEY_SIMPLEX, 0.3, CV_RGB(0,255,0));
            }
            delete text;
            
            if(DEBUG_SHOW) {
                cv::imshow("debugShow", debugShow);
                cv::moveWindow("debugShow", 40, 10);
                key = cv::waitKey(delay);
                if (key == 'q' || key == 'Q' || 27 == (int)key) //Esc q\Q\key to stop
                    break;
                else if(key == 's' || key == 'S')
                    delay = 0;
                else
                    delay = 1;
            }
            
   //          //! Clear the temp params
   //          params.clear();
            
        }//end while
        
        delete laneParams;
        delete faceParams;
        featureFile.close();
        resultFileORD.close();
        resultFile.close();
        measureFile.close();
        predictFile.close();
        laneCmbFile.close();
        faceCmbFile.close();
        
        

        //! Delete the queue
        if (msgctl(mq_id, IPC_RMID, NULL) < 0) {
            fprintf(stderr, "delete message queue %x error!\n", mq_id);
        } else {
            fprintf(stderr, "delete message queue %x success!\n", mq_id);
        }

        return 0;
    }//end Process

}//RunFusionCenter

#endif //__cplusplus

using RunFusionCenter:: Process;
int main(int argc, const char * argv[])
{
//    cv::Mat m = cv::Mat::zeros(2, 4, CV_64F);
//    m.at<double>(0,1) = 0.5;
//    m.at<double>(0,3) = 0.5;
//    m.at<double>(1,2) = 0.8;
//    m.at<double>(1,3) = 0.2;
//    cv::Mat cmb = cv::Mat::zeros(1, 4, CV_64F);
//    DST::Combine_Conj(m, cmb);
//    DST::PrintMat(cmb);
//    cv::waitKey();
    return Process(argc, argv);
}
