# if this help to you,please give a star star star
# tensorrt-yolo
Now, just support Ubuntu, windows will be soon.
tensorrt7 support yolov3 yolov3-tiny yolov4 yolov4-tiny and so on if you train from darknet(AB), it can support

-Don't need onnx, directly transport .cfg and .weights to Tensorrt engine

this project borrow from [Deepstream](https://github.com/NVIDIA-AI-IOT/deepstream_reference_apps/tree/restructure) and [wang-xinyu/tensorrtx](https://github.com/wang-xinyu/tensorrtx)

## Excute:
```
1. clone.
2.set CMakeList.txt tensorrt path, opencv path.
3.in main.cpp, set diffenrt cfg and weights
4.set .cfg input_w and input_h,due to tensorrt upsample , input_w shuld equal input_h
5.copy .cfg and .weights file to folder 
6.build project
7.run ./yolo -s to build yolo engine
7.run ./yolo -d to start detect
