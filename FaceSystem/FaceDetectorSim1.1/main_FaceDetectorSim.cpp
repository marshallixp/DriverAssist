//
//  main.cpp
//  FaceDetectorSim1.1
//
//  Created by XUANPENG LI on 23/09/13.
//  Copyright (c) 2013 ESIEE-Amiens. All rights reserved.
//
#include "main_FaceDetectorSim.h"
#include "Process_FaceDetectorSim.h"
#include <stdexcept>

#ifdef __cplusplus
///Time
extern const double SAMPLING_TIME       = 60;
extern const double SAMPLING_FREQ       = 9.409;
extern const double TIME_BASELINE       = 300; //sec

/// Face feature analysis
extern const double TIME_PERCLOS_WINDOW = 60;
extern const double TIME_BLINK_WINDOW   = 60;
extern const double THRESHOLD_PERCLOS   = 0.8;
extern const double THRESHOLD_CLOSURE   = 0.7;
/// Size of Image
extern const double COEF                = 0.75;
extern const int    WIDTH               = 640;
extern const int    HEIGHT              = 480;
/// Multi-Image Show
extern const int    WIN_COLS            = 3;
extern const int    WIN_ROWS            = 3;

/// Capture images
extern const int    IMAGE_RECORD        = 0; // Record Face images
extern const int    EYES_RECORD         = 0; // Record eyes images

#ifdef __APPLE__
extern const char   FACE_IMG_NAME[] 	= "/Users/xuanpengli/DriveAssist/Capture/face/face_%d.jpg";
extern const char   EYE_COLOR_NAME[]    = "/Users/xuanpengli/DriveAssist/Capture/Eye_color/eye_color_%d_%d.jpg";
extern const char   EYE_BIN_NAME[]      = "/Users/xuanpengli/DriveAssist/Capture/Eye_bin/eye_bin_%d_%d.jpg";
/// Record docs
extern const char   FACE_RECORD_FILE[]  = "/Users/xuanpengli/DriveAssist/Data/FaceFeatures_22-03-2014_13h05m12s.txt";
extern const char   FILE_FACE_FEATURE[] = "/Users/xuanpengli/DriveAssist/Sim_Record/Sim_FaceFeatures_22-03-2014_13h05m12s.txt";
// extern const char   FACE_RECORD_FILE[]  = "/Users/xuanpengli/DriveAssist/Data/FaceFeatures_16-03-2014_16h37m38s.txt";
// extern const char   FILE_FACE_FEATURE[] = "/Users/xuanpengli/DriveAssist/Sim_Record/Sim_FaceFeatures_16-03-2014_16h37m38s.txt";
/// Data Source
extern const char   FACE_RAW_NAME[]     = "/Users/xuanpengli/DriveAssist/Data/FaceRaw_22-03-2014_13h05m12s/face_%d.jpg";
// extern const char   FACE_RAW_NAME[]     = "/Users/xuanpengli/DriveAssist/Data/FaceRaw_16-03-2014_16h37m38s/face_%d.jpg";
// extern const char   FACE_RAW_NAME[]     = "/Users/xuanpengli/DriveAssist/Data/FaceRaw_10-07-2013_18h30m21s/face_%d.jpg";
extern const char   KEY_PATH[]          = "/Users/xuanpengli/DriveAssist/key.txt";


#elif defined __linux
extern const char   FACE_IMG_NAME[] 	= "/home/lixp/Developer/Capture/face/face_%d.jpg";
extern const char   EYE_COLOR_NAME[]    = "/home/lixp/Developer/Capture/Eye_color/eye_color_%d_%d.jpg";
extern const char   EYE_BIN_NAME[]      = "/home/lixp/Developer/Capture/Eye_bin/eye_bin_%d_%d.jpg";
/// Record docs
extern const char   FACE_RECORD_FILE[]  = "/home/lixp/Developer/Data/FaceFeatures_22-03-2014_13h05m12s.txt";
extern const char   FILE_FACE_FEATURE[] = "/home/lixp/Developer/Sim_Record/Sim_FaceFeatures_22-03-2014_13h05m12s.txt";
/// Data Source
extern const char   FACE_RAW_NAME[]     = "/home/lixp/Developer/Data/FaceRaw_22-03-2014_13h05m12s/face_%d.jpg";
extern const char   KEY_PATH[]          = "/home/lixp/Developer/key.txt";
#endif

/// Classifier Cascade 
extern const char   EYE_CASCADE_NAME[]  = "/usr/local/share/OpenCV/haarcascades/haarcascade_eye.xml";
extern const char   FACE_CASCADE_NAME[] = "/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml";

/// Frame   
// extern const int    FRAME_START         = 2028;		//18:34:00 \2028
// extern const int    FRAME_STOP1         = 15580;    //18:57:59
// extern const int    FRAME_RESTART1      = 16712;    //19:00:00
// extern const int    FRAME_STOP2         = 19539;    //19:04:59
// extern const int    FRAME_RESTART2      = 21801;    //19:09:00
// extern const int    FRAME_END           = 23511; 	//19:11:59
// extern const int    FRAME_START         = 1;
// extern const int    FRAME_END           = INFINITY;


namespace FaceDetectorSim {
    int Process(int argc, const char* argv[])
	{
        std::cout << "/*************************************/" << std::endl;
        std::cout << "Input FACE_DETECTOR" << std::endl;
        std::cout << "Input FACE_ANALYZER" << std::endl;
        std::cout << "Input SEND_DATA" << std::endl;
        std::cout << "Input DATA_RECROD" << std::endl;
        std::cout << "Input StartFrame" << std::endl;
        std::cout << "Input EndFrame" << std::endl;
        std::cout << "Input YAW_ANGLE" << std::endl;
        std::cout << "Input PITCH_ANGLE" << std::endl;
        std::cout << "/*************************************/" << std::endl;
        if(argc < 7)
            std::cout << "Not enough parameters" << std::endl;

		/// Run application
		int FACE_DETECTOR   = atoi(argv[1]);
        int FACE_ANALYZER   = atoi(argv[2]);
		int SEND_DATA       = atoi(argv[3]);
		int DATA_RECORD     = atoi(argv[4]); // Record the data
		int StartFrame		= atoi(argv[5]); 
		int EndFrame		= atoi(argv[6]); 
	
        int idx 			= StartFrame;  //index for image sequence
        int sampleIdx 		= 1;
        char faceImg[100];

        double initTime         = (double)cv::getTickCount();
        double intervalTime     = 0;
        double execTime         = 0;  // Execute Time for Each Frame
        double pastTime         = 0;
        double lastStartTime    = (double)cv::getTickCount();
        char key;
        double delay = 1;
        
        std::ofstream faceFeatureFile;
        if (DATA_RECORD){
            InitRecordData(faceFeatureFile, FILE_FACE_FEATURE, faceFeatureName, NUM_FACE);
        }
        
        /* Parameters for Face Detector */
        cv::Mat faceMat;
        cv::Rect faceRoiRect;
        cv::CascadeClassifier faceCascade(FACE_CASCADE_NAME);
        cv::CascadeClassifier eyeCascade(EYE_CASCADE_NAME); 
        std::vector<double> eyesHeightBaselineVec;
        std::vector<double> eyesHeightVec;
        std::vector<cv::Point> eyesPointVec;
        std::deque<FaceDetector::InfoPERCLOS> PERCLOSDeque; 
        std::deque<FaceDetector::InfoBLINK> BLINKDeque;
        FaceDetector::FaceFeature faceFeatures;
        int faceKalmanIndex  = 0;
        std::vector<float> samplingTime;

        cv::Mat hist_camshift;
        cv::MatND faceHist;
        std::vector<FaceDetector::PARTICLE_FACE> particles;
        
        //! Initialize Face Kalman Filter
        int trackKalman = 0;
        cv::KalmanFilter faceKalmanFilter(4,4,0);
        cv::Mat faceKalmanMeasureMat(4, 1, CV_32F, cv::Scalar::all(0));
		
		/* Get mapping of time */
        GetSamplingTime(FACE_RECORD_FILE, samplingTime);
       
		if (FACE_DETECTOR) {
            FaceDetector::InitFaceKalmanFilter(faceKalmanFilter, faceKalmanMeasureMat);       
        } 
        
        
        /* Inter-process communication */
        key_t ipckey;
        struct { 
            long type; 
            char text[1024]; 
        } faceMsg;
        int mq_id;
        if (SEND_DATA) 
        {
            /* Generate the ipc key */
            ipckey = ftok(KEY_PATH, 'a');
            if(ipckey == -1){
                printf("Key Error: %s\n", strerror(errno));
                exit(1);
            }
            
            mq_id = msgget(ipckey, 0);
            if (mq_id == -1) { 
                //MQ doesn't exit
                mq_id = msgget(ipckey, IPC_CREAT | IPC_EXCL | 0666);
                printf("FaceDetector creates a new MQ %d\n", mq_id);
            }
            else {
                //MQ does exit
                mq_id = msgget(ipckey, IPC_EXCL | 0666);
                printf("FaceDetector uses an existed MQ %d\n", mq_id);
            }
            //printf("Lane identifier is %d\n", mq_id);
            if(mq_id == -1) {  
                throw std::logic_error("Can't build pipeline"); 
            }
        }

		double delayTime;
		if(idx == 1)
			delayTime = 0;
		else
			delayTime = samplingTime.at(idx - 2);

        
        /* Entrance of Face Detector */
        if (faceCascade.empty() | eyeCascade.empty())
            abort ();
        else
        {
            while (idx <= EndFrame)
            {
                printf("\nProcess in %d frames: \n", idx);
                double startTime = (double)cv::getTickCount();

                sprintf(faceImg, FACE_RAW_NAME, idx);
                faceMat = cv::imread(faceImg);
                
                /* Face detect and tracking */  
                if (FACE_DETECTOR) {
                    ProcessFaceImage(faceMat, faceCascade, eyeCascade, 
                        faceRoiRect, eyesHeightVec, eyesPointVec, startTime, idx, 
                        trackKalman, faceKalmanFilter, faceKalmanMeasureMat, faceKalmanIndex, 
                        execTime, hist_camshift, faceHist, particles);
                }

                intervalTime = (startTime - lastStartTime)/ cv::getTickFrequency();//get the time between two continuous frames
                lastStartTime = startTime;
                std::cout << "intervalTime: "<< intervalTime << std::endl;

                /**********************************************************/
                //! Generate the relative information  
                //! Don't calculate the first sampling
                //! Calculate the mass after BASELINE_FRAME
                /**********************************************************/
                if (FACE_DETECTOR && FACE_ANALYZER) {
                    if ( pastTime < TIME_BASELINE + delayTime) {
                        /// Get face baseline
                        FaceDetector::GetFaceBaseline(sampleIdx,
                                        eyesHeightVec, 
                                        eyesHeightBaselineVec,
                                        faceFeatures);
                    }
                    else {
                        /// Generate face mass after baseline acquired 
                        FaceDetector::GenerateFaceIndicators(sampleIdx,
                                               eyesHeightVec, 
                                               faceFeatures, 
                                               PERCLOSDeque, 
                                               BLINKDeque, 
                                               intervalTime);
                        
                        //! eyesHeight
                        char *text_eyesHeight = new char[30];
                        sprintf(text_eyesHeight, "Eyes Height: %.2f", faceFeatures.Height);
                        cv::putText(faceMat, text_eyesHeight, cv::Point(0, 70), cv::FONT_HERSHEY_SIMPLEX, 0.3, CV_RGB(0,255,0));
                        delete text_eyesHeight;
                        //! PERCLOS
                        char *text_PERCLOS = new char[30];
                        sprintf(text_PERCLOS, "F1. PERCLOS: %.4f", faceFeatures.PERCLOS);
                        cv::putText(faceMat, text_PERCLOS, cv::Point(0, 80), cv::FONT_HERSHEY_SIMPLEX, 0.3, CV_RGB(0,255,0));
                        delete text_PERCLOS;
                        //! MICROSLEEP
                        char *text_MICROSLEEP = new char[30];
                        sprintf(text_MICROSLEEP, "F2. MICROSLEEP: %.4f", faceFeatures.MICROSLEEP);
                        cv::putText(faceMat, text_MICROSLEEP, cv::Point(0,90), cv::FONT_HERSHEY_SIMPLEX, 0.3, CV_RGB(0,255,0));
                        delete text_MICROSLEEP;
                        //! BLINK
                        char *text_BLINK = new char[30];
                        sprintf(text_BLINK, "F3. BLINK: %.4f", faceFeatures.BLINK);
                        cv::putText(faceMat, text_BLINK, cv::Point(0, 100), cv::FONT_HERSHEY_SIMPLEX, 0.3, CV_RGB(0,255,0));
                        delete text_BLINK;
                        
                        //! Eye states
                        char *text_eye = new char[30];
                        if (BLINKDeque.back().eyeBLINK == 0) {
                            sprintf(text_eye, "%s", "Eyes Open");
                            cv::putText(faceMat, text_eye, cv::Point(180, 80), cv::FONT_HERSHEY_SIMPLEX, 0.8, CV_RGB(0,0,255));
                        } else {
                            sprintf(text_eye, "%s", "Eyes Closure");
                            cv::putText(faceMat, text_eye, cv::Point(180, 80), cv::FONT_HERSHEY_SIMPLEX, 0.8, CV_RGB(255,0,0));
                        }
                        delete text_eye;
                        
                    }//end Generate Face Indicators
                }//end FACE_DETECTOR
            
                if(IMAGE_RECORD){
                    char *faceImg = new char[100]; 
                    sprintf(faceImg, FACE_IMG_NAME, idx);
                    cv::imwrite(faceImg, faceMat);
                    delete faceImg;
                }
                
                /* Send the datas as string to fusion center */
                if(FACE_DETECTOR & SEND_DATA) {
                    char *str = new char[1024];
                    memset(str, 0, 1024);
                    CodeMsg(faceFeatures, str);
    
                    strcpy(faceMsg.text, str);
                    faceMsg.type = 1;
                    //! 0 will cause a block/ IPC_NOWAIT will close the app.
                    if(msgsnd(mq_id, &faceMsg, sizeof(faceMsg), 0) == -1)
                    {  
                        throw std::runtime_error("FaceDetectSim: msgsnd failed!");  
                    }  
                    delete str;
                }

                 
                    
                /* Adjust the interval time within fixed frequency */
				double execFreq;
                do {
                    execFreq = 1.0 / (((double)cv::getTickCount() - startTime)/cv::getTickFrequency());   
                    pastTime = ((double)cv::getTickCount() - initTime)/cv::getTickFrequency() + delayTime;             
				} while ( pastTime < samplingTime.at(idx-1) );
                // }while(execFreq > SAMPLING_FREQ);
                // }while(pastTime < (double)sampleIdx/SAMPLING_FREQ);

                /* Record Face features */
                if (DATA_RECORD) {
                    RecordFaceFeatures(faceFeatureFile, faceFeatures, pastTime);
                }

                char *text = new char[30];
                sprintf(text, "past time: %.2f sec", pastTime);
                cv::putText(faceMat, text, cv::Point(0, faceMat.rows-5), cv::FONT_HERSHEY_SIMPLEX, 0.8, CV_RGB(0,255,0));

                sprintf(text, "Adjusted Freq: %.2f Hz", execFreq);
                cv::putText(faceMat, text, cv::Point(0, 60), cv::FONT_HERSHEY_SIMPLEX, 0.8, CV_RGB(0, 255, 0));
                delete text;

                cv::imshow("Face System", faceMat);
                cv::moveWindow("Face System", 790, 410);
                key = cv::waitKey(delay);
                if (key == 'q' || key == 'Q' || 27 == (int)key) //Esc q\Q\key to stop
                    break;
                else if(key == 's' || key == 'S')
                    delay = 0;
                else
                    delay = 1;
                    
                /* Update the sampling index */
                sampleIdx++;
                idx++;

                // if(idx == FRAME_STOP1)
                //     idx = FRAME_RESTART1;
                // if(idx == FRAME_STOP2)
                //     idx = FRAME_RESTART2;
            }//end while
        }//end if
            
        faceFeatureFile.close();
        cv::destroyAllWindows();
        
        return 0;
    }//end Process
}//FaceDetectorSim

#endif //__cplusplus

using FaceDetectorSim:: Process;
int main(int argc, const char * argv[])
{
    return Process(argc, argv);
}
