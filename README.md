<script type="text/javascript" async
  src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-MML-AM_CHTML">
</script>


# Advanced-Vision-Systems-Camera-Calibration-Neural-Network-Deployment
This repository presents, wherein camera functionalities have been intricately integrated with neural network deployment on Jetson. From intricate camera calibrations to robust object detection using TensorRT, the intricacies of advanced computer vision methodologies have been meticulously explored and implemented in f1 tenth car.


## Overview
# Vision Lab Project: Autonomous Navigation

## Overview

# Vision Lab Project: Autonomous Navigation

## Overview

The Vision Lab project is an amalgamation of computer vision techniques and neural network models aimed at autonomous navigation. This README provides a deep dive into the technical details of the project.

## Camera Access on Linux

### Interface

Used v4l2 (Video for Linux 2) API, a collection of device drivers and API layers for video capture on Linux.

### Code

Integrated with OpenCV's VideoCapture class to fetch RGB frames from `/dev/video2` at a resolution of 960x540 pixels and a frequency of 60Hz.

## Camera Calibration & Distance Measurement

### Calibration

Deployed a two-step corner detection and refinement process using `findChessboardCorners` and `cornerSubPix`. Leveraged these points with `calibrateCamera` to derive the intrinsic matrix \( K \).

### Height Estimation

Using the similar triangles property, the height \( H \) of the camera, relative to the car frame, was estimated with the formula:
\[ H = \frac{f \times \text{realHeight} \times \text{imageHeight}}{\text{objectHeightInImage} \times \text{realDistance}} \]

### Distance Algorithm

Employed the pinhole camera model, where the distance \( D \) to an object is given by:
\[ D = \frac{f \times \text{realHeight}}{\text{objectHeightInImage}} \]

## Lane Detection

### Pre-processing

Converted images to HSV color space for effective color-based segmentation.

### Detection

Used OpenCV's `findContours` after color thresholding to detect lane markings. The contours were then filtered based on aspect ratio and area to isolate lane markings.

## Object Detection Network Training & Deployment

### Architecture

Utilized a YOLO (You Only Look Once) model, which divides an image into an \( S \times S \) grid. Each grid cell predicts bounding boxes and confidence scores for these boxes.

### Training

The loss function for the YOLO model is a combination of coordinate loss, objectness loss, and classification loss:
\[ \text{Loss} = \lambda_{\text{coord}} \sum_{i=0}^{S^2} \sum_{j=0}^{B} 1_{ij}^{\text{obj}} \left[ (x_i - \hat{x}_i)^2 + (y_i - \hat{y}_i)^2 \right] \]

### Conversion

The trained PyTorch model was exported to the ONNX (Open Neural Network Exchange) format, which was then optimized and converted to a TensorRT engine for efficient deployment.

## Integration

### Pipeline

A sequence of operations was established: image capture → lane detection → object recognition → distance calculation. This pipeline was executed for each frame, ensuring a seamless perception system for navigation.

## Conclusion

This comprehensive system, housed on GitHub, exemplifies the seamless merger of traditional computer vision and modern deep learning techniques for real-time autonomous navigation tasks.
