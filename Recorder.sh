# This file runs the lane and face sim the database

# Author: Li Xuanpeng <li_xuanpeng@esiee-amiens.fr>
# Date: 07/10/2013

# suffix

# options
options1="1 0 1 0 10 50 0 0.1 600 0"; #LANE_IMG_RECORD, LANE_DETECTOR, DATA_RECORD, SEND_DATA, Sampling Freq, Image Quality, YAW_ANGLE, PITCH_ANGLE, Time_BaseLine, LANE_ANALYSER
options2="1 0 1 0 10 50 600 0"; #FACE_IMG_RECORD, FACE_DETECTOR, DATA_RECORD, SEND_DATA, Sampling Freq, Image Quality, Time_BaseLine, FACE_ANALYSER

# command
command1="./LaneSystem/LaneRecorder64 $options1"
echo $command1

command2="./FaceSystem/FaceRecorder64 $options2"
echo $command2

#command3="./RunFusionCenter"
#echo $command3

#run
$command2 & $command1
