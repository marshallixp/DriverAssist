//
//  main.cpp
//  FaceRecorder1.1
//
//  Created by Xuanpeng Li on 10/13.
//  Copyright (c) 2012 ESIEE-AMIENS. All rights reserved.
//  
//  Tip:    argv[1]: FACE_IMG_RECORD
//          argv[2]: DATA_RECORD
//          argv[3]: SAMPLING_FREQ
//          argv[4]: compression_params
//          argv[5]: ProcessFaceImage
// 

#include "main_FaceRecorder.h"
#include "Process_FaceRecorder.h"   
#include <stdexcept>

#ifdef __cplusplus

/// Face feature analysis
extern const double     TIME_PERCLOS_WINDOW     = 60;   //PERCLOS window(secs)
extern const double     TIME_BLINK_WINDOW       = 60;   //BLINK window(secs)
extern const double     THRESHOLD_PERCLOS       = 0.8;  //PERCLOS
extern const double     THRESHOLD_CLOSURE       = 0.7;  //MICROSLEEP
/// Size of Image
extern const double     COEF                    = 0.75;
extern const int        WIDTH                   = 640;
extern const int        HEIGHT                  = 480;
/// Multi-Image Show
extern const int        WIN_COLS                = 3;
extern const int        WIN_ROWS                = 3;

/// Debug Markers
extern const int        COLOR_EYES_RECORD       = 0;    // Record detected eyes images
extern const int        COLOR_EYES_TRACK_RECORD = 0;	// Record tracked eyes images
extern const int        EYES_RECORD             = 0;    // Record binary eyes images
/// Frame
extern const int        FRAME_START             = 1; 
extern const int        FRAME_END               = INFINITY;

#ifdef __APPLE__
/// Record images
extern const char       FACE_PATH_NAME[]        = "/Users/xuanpengli/DriveAssist/Data/FaceRaw_%02d-%02d-%d_%02dh%02dm%02ds";
extern const char       FACE_IMG_NAME[]         = "/face_%d.jpg";
extern const char       EYE_COLOR_NAME[]        = "/Users/xuanpengli/DriveAssist/Capture/Eye_raw/eyeRaw_%d_%d.jpg";
extern const char       EYE_BIN_NAME[]          = "/Users/xuanpengli/DriveAssist/Capture/Eye_bin/eyeBin_%d_%d.jpg";
/// Record docs
extern const char       FILE_FACE_FEATURE[]     = "/Users/xuanpengli/DriveAssist/Data/FaceFeatures_%02d-%02d-%d_%02dh%02dm%02ds.txt";
/// Classifier cascade 
extern const char       EYE_CASCADE_NAME[]      = "/usr/local/share/OpenCV/haarcascades/haarcascade_eye.xml";
extern const char       FACE_CASCADE_NAME[]     = "/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml";
extern const char       KEY_PATH[]              = "/Users/xuanpengli/DriveAssist/key.txt";

#elif defined __linux
/// Record images
extern const char       FACE_PATH_NAME[]        = "/home/lixp/Developer/Data/FaceRaw_%02d-%02d-%d_%02dh%02dm%02ds";
extern const char       FACE_IMG_NAME[]         = "/face_%d.jpg";
extern const char       EYE_COLOR_NAME[]        = "/home/lixp/Developer/Capture/Eye_raw/eyeRaw_%d_%d.jpg";
extern const char       EYE_BIN_NAME[]          = "/home/lixp/Developer/Capture/Eye_bin/eyeBin_%d_%d.jpg";
/// Record docs
extern const char       FILE_FACE_FEATURE[]     = "/home/lixp/Developer/Record/FaceFeatures_%02d-%02d-%d_%02dh%02dm%02ds.txt";
/// Classifier cascade 
extern const char       EYE_CASCADE_NAME[]      = "/usr/local/share/OpenCV/haarcascades/haarcascade_eye.xml";
extern const char       FACE_CASCADE_NAME[]     = "/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml";
extern const char       KEY_PATH[]              = "/home/lixp/Developer/key.txt";
#endif

namespace FaceRecorder{
	int Process(int argc, const char* argv[])
	{
        std::cout << "/*************************************/" << std::endl;
        std::cout << "Input FACE_IMG_RECORD" << std::endl;
        std::cout << "Input FACE_DETECTOR" << std::endl;
        std::cout << "Input DATA_RECORD" << std::endl;
        std::cout << "Input SEND_DATA" << std::endl;
        std::cout << "Input SAMPLING_FREQ" << std::endl;
        std::cout << "Input COMPRESSION_RATE" << std::endl;
        std::cout << "Input TIME_BASELINE" << std::endl;
        std::cout << "Input FACE_ANALYSER" << std::endl;
        std::cout << "/*************************************/" << std::endl;
        if(argc < 9)
            std::cout << "Not enough parameters" << std::endl;

        int FACE_IMG_RECORD     = atoi(argv[1]); //Record the raw face image
        int FACE_DETECTOR       = atoi(argv[2]);
        int DATA_RECORD         = atoi(argv[3]); //Record the results of analysis
        int SEND_DATA           = atoi(argv[4]);
        int SAMPLING_FREQ       = atoi(argv[5]); //Sample Frequency
        int COMPRESSION_RATE    = atoi(argv[6]);
        double TIME_BASELINE    = atof(argv[7]);
        int FACE_ANALYSER       = atoi(argv[8]);

        char key;
        int delay = 1;
        
        // Parameters for All
        int idx                 = FRAME_START; //index for image sequence
        int sampleIdx           = 1;
        
        
        //! Get the current time
        time_t t;
        time(&t);
        tm *tmTime = localtime(&t);
        char currentTime[50];
        sprintf(currentTime,"%02d-%02d-%d_%02dh%02dm%02ds",tmTime->tm_mday,tmTime->tm_mon+1,tmTime->tm_year+1900, tmTime->tm_hour,tmTime->tm_min, tmTime->tm_sec);  
        std::cout << "Current Time: " << currentTime << std::endl;
        
        //! Create a new doc for recording the images
        char * faceRawImgPath = new char[100];
        DIR *pDir;
        if (FACE_IMG_RECORD) {
            do { 
                sprintf(faceRawImgPath, FACE_PATH_NAME, tmTime->tm_mday,tmTime->tm_mon+1,tmTime->tm_year+1900, tmTime->tm_hour,tmTime->tm_min, tmTime->tm_sec);
                pDir = opendir(faceRawImgPath);
            } while (pDir != NULL);
            mkdir(faceRawImgPath, S_IRWXU);
        }
        
        //! Init the feature record file
        std::ofstream faceFeatureFile;
        char *faceFeaturePath = new char[100];
        if (DATA_RECORD){
            sprintf(faceFeaturePath, FILE_FACE_FEATURE, tmTime->tm_mday,tmTime->tm_mon+1,tmTime->tm_year+1900, tmTime->tm_hour,tmTime->tm_min, tmTime->tm_sec);
            
            InitRecordData(faceFeatureFile, faceFeaturePath, faceFeatureName, NUM_FACE_FEATURES);
        }
       
        
        /* Parameters for Face Detector */
        cv::VideoCapture faceCap(0); //Open the default camera
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
        int faceKalmanIndex = 0;
        
        /* Initialize Face Kalman Filter */
        int trackKalman = 0;
        cv::KalmanFilter faceKalmanFilter(4,4,0);
        cv::Mat faceKalmanMeasureMat(4, 1, CV_32F, cv::Scalar::all(0));
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
        
        /* Entrance of Face Detector */
        if (! faceCap.isOpened() | faceCascade.empty() | eyeCascade.empty())
            abort();
    
        // Start Time in the process
        double initTime         = (double)cv::getTickCount();
        double intervalTime     = 0;
        double execTime         = 0;
        double pastTime         = 0;
        double lastStartTime    = (double)cv::getTickCount();
        while(idx < FRAME_END)
        {   
            double startTime = (double)cv::getTickCount();

            faceCap >> faceMat;

               
            //! Reduce the size of raw image
            cv::resize(faceMat, faceMat, cv::Size(cvRound(WIDTH * COEF), cvRound(HEIGHT * COEF)), cv::INTER_AREA);
            
            // Record the resized raw image
            if (FACE_IMG_RECORD) 
            {
                std::vector<int> compression_params;
                compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);
                compression_params.push_back(COMPRESSION_RATE);//Image Compressed Size
                
                char * rawFaceName = new char[100];
                char * tempName = new char[100];
                sprintf(rawFaceName, FACE_IMG_NAME, idx);
                strcpy(tempName, faceRawImgPath);
                strcat(tempName, rawFaceName);
                cv::imwrite(tempName, faceMat, compression_params);
                delete rawFaceName;
                delete tempName;
            }

            /* Face Detector starts */       
            if(FACE_DETECTOR)
            {
                ProcessFaceImage(faceMat, faceCascade, eyeCascade, faceRoiRect, eyesHeightVec, eyesPointVec, idx, 
                trackKalman, faceKalmanFilter, faceKalmanMeasureMat, faceKalmanIndex, execTime);
            }
            
            /* Calculate the running time for every sampling */  
            pastTime = ((double)cv::getTickCount() - initTime)/cv::getTickFrequency();
            char *text_pastTime = new char[50];
            sprintf(text_pastTime, "FaceDetector Time: %.2f sec", pastTime);
            cv::putText(faceMat, text_pastTime, cv::Point(0, faceMat.rows-5), cv::FONT_HERSHEY_SIMPLEX, 0.3, CV_RGB(0,255,0));
            delete text_pastTime;
            
            intervalTime = (startTime - lastStartTime)/ cv::getTickFrequency();//get the time between two continuous frames
            lastStartTime = startTime;

            /* Generate the relative information */
            if (FACE_DETECTOR && FACE_ANALYSER) {
                if ( pastTime < TIME_BASELINE) {
                    /// Get face baseline
                    FaceDetector::GetFaceBaseline(sampleIdx, eyesHeightVec, 
                                                  eyesHeightBaselineVec, faceFeatures);
                }
                else {
                    /// Generate face mass after baseline acquired 
                    FaceDetector::GenerateFaceIndicators(sampleIdx, eyesHeightVec, 
                                                   faceFeatures, PERCLOSDeque,
                                                   BLINKDeque, intervalTime);
                    //! eyesHeight
                    char *text_eyesHeight = new char[50];
                    sprintf(text_eyesHeight, "Eyes Height: %.2f", faceFeatures.Height);
                    cv::putText(faceMat, text_eyesHeight, cv::Point(0, 70), cv::FONT_HERSHEY_SIMPLEX, 0.3, CV_RGB(0,255,0));
                    delete text_eyesHeight;
                    //! PERCLOS
                    char *text_PERCLOS = new char[50];
                    sprintf(text_PERCLOS, "F1. PERCLOS: %.4f", faceFeatures.PERCLOS);
                    cv::putText(faceMat, text_PERCLOS, cv::Point(0, 80), cv::FONT_HERSHEY_SIMPLEX, 0.3, CV_RGB(0,255,0));
                    delete text_PERCLOS;
                    //! MICROSLEEP
                    char *text_MICROSLEEP = new char[50];
                    sprintf(text_MICROSLEEP, "F2. MICROSLEEP: %.4f", faceFeatures.MICROSLEEP);
                    cv::putText(faceMat, text_MICROSLEEP, cv::Point(0, 90), cv::FONT_HERSHEY_SIMPLEX, 0.3, CV_RGB(0,255,0));
                    delete text_MICROSLEEP;
                    //! BLINK
                    char *text_BLINK = new char[50];
                    sprintf(text_BLINK, "F3. BLINK: %.4f", faceFeatures.BLINK);
                    cv::putText(faceMat, text_BLINK, cv::Point(0, 100), cv::FONT_HERSHEY_SIMPLEX, 0.3, CV_RGB(0,255,0));
                    delete text_BLINK;
                    
                    //! Eye states
                    char *text_eye = new char[50];
                    if (BLINKDeque.back().eyeBLINK == 0) {
                        sprintf(text_eye, "%s", "Eyes Open");
                        cv::putText(faceMat, text_eye, cv::Point(180, 30), cv::FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(0,255,0));
                    } else {
                        sprintf(text_eye, "%s", "Eyes Closure");
                        cv::putText(faceMat, text_eye, cv::Point(180, 30), cv::FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(255,0,0));
                    }
                    delete text_eye;
                }//end GenerateFaceIndicators
            }//end Face_DETECTOR
            
    
            /* Record Face Features */
            if (DATA_RECORD) {
                faceFeatures.frame = idx;
                RecordFaceFeatures(faceFeatureFile, faceFeatures, execTime, pastTime);
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
                      throw std::runtime_error("FaceRecorder: msgsnd failed!"); 
                }  
                delete str;
            }
            
            /* Adjust the interval time within fixed frequency */
            double execFreq;
            do {
                // cv::waitKey(1);
                execFreq = 1.0 / (((double)cv::getTickCount() - startTime)/cv::getTickFrequency());                    
            } while ( execFreq >= SAMPLING_FREQ );


            char *text = new char[30];
            sprintf(text, "Process: %.2f Hz", execFreq);
            cv::putText(faceMat, text, cv::Point(0, 60), cv::FONT_HERSHEY_SIMPLEX, 0.8, CV_RGB(0, 255, 0));
            delete text;

            cv::imshow("Face", faceMat);
            key = cv::waitKey(delay);
            if (key == 'q' || key == 'Q' || 27 == (int)key) //Esc q\Q\key to stop
                break;
            else if(key == 's' || key == 'S')
                delay = 0;
            else
                delay = 1;

            sampleIdx++;
            idx++;
        }// while loop end
        
        faceFeatureFile.close();
        delete faceRawImgPath;
        delete faceFeaturePath;
        cv::destroyAllWindows(); 
        
        return 0;
    }
}//FaceRecorder

#endif //_cplusplus

using FaceRecorder:: Process;
int main (int argc, const char * argv[])
{
    return Process(argc, argv);
}
