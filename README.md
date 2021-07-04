# Modified Yolov4Tiny RaspberryPi

This repository contains a PyTorch implementation of the architecture introduced in the ["Real-time object detection method based on improved YOLOv4-tiny"][https://arxiv.org/abs/2011.04244] paper. 

The network have been trained to detect and classify faces with and without masks. 

[models][https://github.com/benedettaliberatori/Modified-Yolov4Tiny-RaspberryPi/tree/main/models] contains the following trained models: 
*  `downblur.pt `, baseline model. 
*  `downblur.tflite` , TFlite conversion of the previous one.
*  `pruneddp.tflite` , pruned version in TFlite version.  

## Usage

Run  

```bash
python detect/detectttf.py
```
to use the model converted in TFlite
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



