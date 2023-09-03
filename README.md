# Advanced Vision Systems for Autonomous Cars: Integrating Classical Computer Vision with Deep Learning Techniques

This repository presents, wherein camera functionalities have been intricately integrated with neural network deployment on Jetson. From intricate camera calibrations to robust object detection using TensorRT, the intricacies of advanced computer vision methodologies have been meticulously explored and implemented in f1 tenth car. 
## Overview


## Vision Lab Project

The Vision Lab project is an amalgamation of computer vision techniques and neural network models aimed at autonomous navigation. Here's a deep dive into the technical details:

## Camera Access on Linux:
- **Interface**: Used v4l2 (Video for Linux 2) API, a collection of device drivers and API layers for video capture on Linux.
- **Code**: Integrated with OpenCV's `VideoCapture` class to fetch RGB frames from `/dev/video2` at a resolution of 960×540 pixels and a frequency of 60Hz.

## Camera Calibration & Distance Measurement:
- **Calibration**: Deployed a two-step corner detection and refinement process using `findChessboardCorners` and `cornerSubPix`. Leveraged these points with `calibrateCamera` to derive the intrinsic matrix \( K \).
- **Height Estimation**: Using similar triangles property, the height \( H \) of the camera, relative to the car frame, was estimated with the formula:

<div style="font-size: 1.5em;">
H = (f × realHeight × imageHeight) / (objectHeightInImage × realDistance)
</div>

- **Distance Algorithm**: Employed the pinhole camera model, where the distance \( D \) to an object is given by:

<div style="font-size: 1.5em;">
D = (f × realHeight) / objectHeightInImage
</div>

## Lane Detection:
- **Pre-processing**: Converted images to HSV color space for effective color-based segmentation.
- **Detection**: Used OpenCV's `findContours` after color thresholding to detect lane markings. The contours were then filtered based on aspect ratio and area to isolate lane markings.

## Object Detection Network Training & Deployment:
- **Architecture**: Utilized a YOLO (You Only Look Once) model, which divides an image into an \( S × S \) grid. Each grid cell predicts \( B \) bounding boxes and confidence scores for these boxes.
- **Training**: The loss function for the YOLO model is a combination of coordinate loss, objectness loss, and classification loss:

<div style="font-size: 1.5em;">
Loss = λ<sub>coord</sub> ∑<sub>i=0</sub><sup>S²</sup> ∑<sub>j=0</sub><sup>B</sup> 1<sub>ij</sub><sup>obj</sup> [(x<sub>i</sub> - x̂<sub>i</sub>)² + (y<sub>i</sub> - ŷ<sub>i</sub>)²]
</div>

- **Conversion**: The trained PyTorch model was exported to the ONNX (Open Neural Network Exchange) format, which was then optimized and converted to a TensorRT engine for efficient deployment.

## Integration:
- **Pipeline**: A sequence of operations was established: image capture → lane detection → object recognition → distance calculation. This pipeline was executed for each frame, ensuring a seamless perception system for navigation.

---

This comprehensive system, housed on GitHub, exemplifies the seamless merger of traditional computer vision and modern deep learning techniques for real-time autonomous navigation tasks.
