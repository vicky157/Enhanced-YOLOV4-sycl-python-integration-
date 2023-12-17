# Enhancing Yolo-v4 Performance using Scalar Matrix Multiplication in oneAPI

## Introduction
This project explores the enhancement of Yolo-v4 performance through Scalar Matrix Multiplication (MM) in oneAPI. We focus on optimizing convolution layers in the Yolo-v4 model, integrating oneAPI with Python.

## Background
We utilize oneAPI to optimize deep learning computations in the Yolo-v4 model, aiming for improved efficiency and accuracy in object detection.

## Methodology
Our approach covers:
1. Scalar Matrix Multiplication in oneAPI
2. Python-C++ Integration
3. Convolution Layer Wrapper
4. Yolo-v4 Modification and Implementation
5. Performance Analysis

## Commands to Run
- Scalar MM: `icpx -fsycl smm.cpp -o smm`
- Python-C++ Integration: `icpx -fsycl -fPIC -shared -o libsmm.so shared.cpp`
- Convolution Wrapper: `python3 wrapper.py`
- Yolo-v4: `python3 yoto.py`, `python3 yolo.py`
- Performance Analysis: `python3 compare.py`

## Results
Demonstrates minimal speedup in convolution layers of the Yolo-v4 model on CPU devices.

## Challenges
Discusses challenges in FLOPs calculation for complex models with custom implementations.

## Conclusion
Highlights the potential of Scalar MM in oneAPI for deep learning optimization.

## Authors
- Vikash Singh (vxs465)
- Thomas Bornhorst (thb34)

## Institution
Case Western Reserve University


