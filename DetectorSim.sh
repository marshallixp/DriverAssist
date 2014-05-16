# This file runs the lane and face sim the database

# Author: Li Xuanpeng <li_xuanpeng@esiee-amiens.fr>
# Date: 07/10/2013

# Database
# 10-07-2013_18h30m21s: 1840 ~ 18000(lane), 2028 ~ 20093(face)
# 16-03-2014_16h37m38s: 1 ~ 15015(lane), 75 ~ 15555(face)
# 22-03-2014_13h05m12s: 1 ~ 18468(lane), 11 ~ 18533(face)

# command
#LANE_DETECTOR, LANE_ANALYZER, SEND_DATA, DATA_RECORD, StartFrame, EndFrame, YAW_ANGLE, PITCH_ANGLE
command1="./LaneSystem/LaneSystem64 1 1 0 0 1 18468 0 0.1"
echo $command1

#FACE_DETECTOR, FACE_ANALYZER, SEND_DATA, DATA_RECORD, StartFrame, EndFrame
command2="./FaceSystem/FaceSystem64 1 1 0 0 11 18533" 
echo $command2

# command3="./FusionSystem/FusionSystem64 1"
# echo $command3

#run
$command1 #& 
# $command2 #& $command3
