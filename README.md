# Modified Yolov4Tiny RaspberryPi

This repository contains a PyTorch implementation of the architecture introduced in the ["Real-time object detection method based on improved YOLOv4-tiny"](https://arxiv.org/abs/2011.04244) for the final project of the Deep Learning course.


The network have been trained to detect and classify faces with and without masks. The directory [datset](https://github.com/benedettaliberatori/Modified-Yolov4Tiny-RaspberryPi/tree/main/dataset) contains images, labels and csv files used to associate them.

The directory [models](https://github.com/benedettaliberatori/Modified-Yolov4Tiny-RaspberryPi/tree/main/models) contains the following trained models: 
*  `model.pt `, baseline model. 
*  `model.tflite` , TFlite conversion of the previous one.
*  `pmodel.tflite` , pruned version in TFlite version. 

The directory [utilities](https://github.com/benedettaliberatori/Modified-Yolov4Tiny-RaspberryPi/tree/main/utilities) contains the implementation of the loss function, functions to deal with bounding boxes, the structured pruning pipeline used and a script to compute the anchor boxes with k-means clustering algorithm. 

The directory [metrics](https://github.com/benedettaliberatori/Modified-Yolov4Tiny-RaspberryPi/tree/main/metrics) contains the scripts used to compute the FPSs and MAPs.

## Usage

Run  

```bash
python detect/detectttf.py namemodel.tflite
```
to use the model converted in TFlite with the selected model
or
```bash
python detect/detect.py
```
to use the Pytorch one and it will automatically open your webcam. 




## Possible Problems

Loading the data you could experience problems with boxes' coordinates exceeding the ranges, such as: 

```bash
File "/home/yourname/anaconda3/lib/python3.8/site-packagesalbumentations/augmentations/bbox_utils.py", line 328, in check_bbox
    raise ValueError(
ValueError: Expected x_max for bbox (0.98828125, 0.20502645502645503, 1.0009765625, 0.2605820105820106, 1.0) to be in the range [0.0, 1.0], got 1.0009765625.)

```

This seems to be a regular problem with the albumentation data augmentation package. In the absence of a real solution and noticing that the error raised from very small values (~10e:-4) we have brutally get around the problem modifying the `check_bbox` function rounding values exceeding the margins to 0 or 1. 



