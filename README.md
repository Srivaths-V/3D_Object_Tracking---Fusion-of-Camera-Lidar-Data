# Sensor Fusion Program - 2nd Project 3D Object Tracking

In the previous project 2D feature tracking we used keypoint detectors, descriptors, and implemented methods to match them between successive images. Also, you know how to detect objects in an image using the YOLO deep-learning framework. And finally, you know how to associate regions in a camera image with Lidar points in 3D space. 

## Project description
Camera & Lidar for 3D Object Tracking

<img src="images/course_code_structure.png" width="779" height="414" />

## Project workflow
These are the four major tasks: 

1. First, I developed a way to match 3D objects over time by using keypoint correspondences. 
2. Second, I computed the TTC based on Lidar measurements. 
3. Then combined point cloud data from Lidar, used computer vision, and deep learning to track the moving vehicle in front of the self-driving car and estimated the time to collision (TTC).
4. Object (vehicle) detected using YOLO Neural network through bounding boxes.
5. Object tracking through matching keypoints from 2D feature tracking and bounding boxes across image frames, simultaneously associating regions in images with Lidar points in 3D space.
7. Then, I computed the TTC based on those matches. 
8. And lastly, I conducted various tests with the framework. My goal is to identify the most suitable detector/descriptor combination for TTC estimation and also to search for problems that can lead to faulty measurements by the camera or Lidar sensor. 

## Project Results

## Dependencies for Running Locally
* cmake >= 2.8
  * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1 (Linux, Mac), 3.81 (Windows)
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* OpenCV >= 4.1
  * This must be compiled from source using the `-D OPENCV_ENABLE_NONFREE=ON` cmake flag for testing the SIFT and SURF detectors.
  * The OpenCV 4.1.0 source code can be found [here](https://github.com/opencv/opencv/tree/4.1.0)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools](https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)

## Basic Build Instructions

1. Clone this repo.
2. Make a build directory in the top level project directory: `mkdir build && cd build`
3. Compile: `cmake .. && make`
4. Run it: `./3D_object_tracking`.
