# Quanser Self-Driving Car Competition - University of Ottawa Team

## Project Overview 
This repository contains the code developed for Stage 1 and 2 of the 2025 Quanser American Control Conference Self-Driving Car Student Competition.

Our system uses a combination of computer vision techniques and PD control to navigate lanes, detect traffic lights, and handle intersections. 

Due to time and resource constraints, some trade-offs and limitations exist, which are detailed below. 

## Stage 1 

## Software Setup

The docker container and the required dependencies were installed by following the instructions from the repository provided by Quanser at https://github.com/quanser/ACC-Competition-2025/blob/main/Software_Guides/ACC%20Software%20Setup%20Instructions.md

After that, ensure if you could navigate to the directory path of ACC_Development/Development/ros2/src in file explorer. Please open up three terminals and enter the following commands.

- **First Terminal**

    - cd /home/$USER/Documents/ACC_Development/docker/virtual_qcar2
    - sudo docker run --rm -it --network host --name virtual-qcar2 quanser/acc2025-virtual-qcar2 bash
    - cd /home/qcar2_scripts/python
    - Note that the QLabs application must be be opened at this point. 
    - python3 Base_Scenarios_Python/Setup_Real_Scenario.py
    - Ensure the message 'Starting Traffic Light Sequence' pops up before executing the commands in the subsequent terminals.

- **Second Terminal**

    - cd /home/$USER/Documents/ACC_Development/isaac_ros_common
    - ./scripts/run_dev.sh  /home/$USER/Documents/ACC_Development/Development
    - colcon build 
    - install/setup.bash
    - ros2 launch qcar2_nodes qcar2_virtual_launch.py

- **Third Terminal**

    - Copy and paste the first two commands from the second terminal in the third terminal. The third terminal is attached to the Isaac docker container running in the second terminal.
    - Navigate to the source subdirectory of the ros2 folder by entering 'cd ros2/src'.
    - Enter the 'ls' command to enusre that you're currently in the directory in which you can only see the 'qcar2_interfaces' and 'qcar2_nodes' packages.
    - Create a new folder by entering the command 'ros2 pkg create --build-type ament_cmake mycar' in the terminal. This will automatically create an empty 'src' and an empty 'include' subdirectory. In addition, the command will automatically create a CMAKE.txt file and a package.xml file. You can compare those files to the ones that the team has uploaded, if necessary.
    - Navigate to the 'src' subdirectory of the newly created 'mycar' folder by entering 'cd src'. Once there, copy and paste the 'setup.py' and the 'uottawa_qcar2_code.py' python scripts. The files are already executable. But to be on safe side, you may enter 'chmod +x setup.py' and 'chmod +x uottawa_qcar2_code.py' to allow user permissions.
    - Navigate back in the path by entering 'cd ..' four times.
    - colcon build 
    - install/setup.bash
    - Go to the path of the python script in the terminal by entering 'cd ros2/src/mycar/src'.
    - Execute the python script by entering: 'python3 uottawa_qcar2_code.py'.

- **Important Notes**
 
     - Please don't restart the simulation if the car doesn't drive properly at the roundabout. That error occurs because our lane detection algorithm isn't perfect.
     - The yield sign does get detected by the car and the corresponding message would pop up in the terminal. However, this depends on the angular view of the depth camera after the car turns right at the the first traffic light. 
     - It should be noted that the frame rate was set to 30 fps to ensure the camera always sends the best quality images at high speed. This contributed to a low CPS rate of only around 10. Due to low CPS, the algorithm may not work as expected multiple times. On our end, the simulation worked after around 7 failed trials. Please use the video as a reference, if necessary. Occasionally, you will observe the following behaviours while validating our algorithm:
     - - Car stopping at the stop sign a bit early or a bit far from the mark due to depth camera glitching. In other words, the car is not stopping near the mark the way it is in the video.
     - - Car not stopping at the stop sign as programmed.
     - - Car going straight after the stop sign instead of turning left.
     - - Car turning right a bit early at the first traffic light. In other words, the right turn is not smooth and the car may be seen being driven on white area.
     - - Car rotating in circles at the beginning before the first yield sign.
     - - Car trying to turn left at the second traffic light.
     - In all above cases, the simulation must be restarted. Alternatively, if the car's behaviour deviates from the behaviour shown in the video, then the simulation must be restarted.
---

## Stage 2

## Software Setup

Instructions were followed according the repository below:
https://github.com/quanser/ACC-Competition-2025/blob/stage2/Software_Guides/physical_qcar2_software_setup_instructions.md

- **First Terminal**

-- The commands from the above repository were executed in the native installation of Nvidia Jetson motherboard in one terminal.

- **Second Terminal**

-- The commands from the same repository are executed in the second terminal to open up the Isaac docker container. 
-- Execute the script to detect signs by running python3 stop_and_yield.py in the docker container.

## Third Terminal

Execute the script to detect road cone by running python3 road_cone_detector.py in the docker container.

## Fourth Terminal

Execute the script to detect obstacles by running python3 object_detector_lidar.py in the docker container.

## Fifth Terminal

Execute the script to activet LEDs by running python3 led_controller.py in the docker container.

## Sixth Terminal

Execute the script to drive the car by running python3 'ride_1.py', 'ride_2.py', or 'ride_3.py' in the docker container.

Note that running the alternative traffic light script is optional.


## Accomplishments

We have successfully implemented the following features:
- **Lane Detection**
- **Traffic Light Detection**
- **Stop Sign Detection**
- **Pick-up and Drop-off Functionality**

---

## Tuning the Parameters of PD Controller
  - Car starts the first trip using:
    - **kp = 0.3**, **kd = 0.15**
  - After passing the stop sign:
    - **kp = 0.45**, **kd = 0.1**
  - When turning right at a red light:
    - The turn is controlled using a `while` loop.
  - Approaching the roundabout:
    - PD values are adjusted to **kp = 0.47**, **kd = 0.15** for smoother navigation.

---

## Repository Structure

- `uottawa_qcar2_code.py`
        -  Lane, stop sign, yield sign, and traffic light detection; passenger pickup and dropoff; PD steering control.
- `README.md` — Project overview and notes.
- `final_video_submission.mp4` — A video file that may be used for reference if car's behavior gets affected by low CPS.

---

## Acknowledgement

We thank Quanser and the organizing team for providing this opportunity to learn and innovate with autonomous vehicles.

---

## Contact

For questions or collaboration:

- **Professor**: Mohamed Atia
- **University**: Carleton University
- **Department**: Systems and Computer Engineering
- **Email**: mohamed.atia@carleton.ca

  

- **Team Name**: Gee-Gees
- **University**: University of Ottawa
  
- **Team Captain**: Mohammad Ahsan Akhlaque
- **Email**: aakhl097@uottawa.ca

- **Team Member**: Kishita Pakhrani
- **Email**: kpakhran@uottawa.ca

- **Team Member**: Jonathan Horton
- **Email**: jhort062@uottawa.ca

---
