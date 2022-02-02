# tensorrt-infer-python

TensorRT is a SDK for hign-performance deep learning inference[https://developer.nvidia.com/tensorrt]. And we can write TensorRT code with python becanse TensorRT has python api. This repository is a sample TensorRT inference code with python which can infer image and output label. There is sample model which can inference Japanese Hiragana character.

## How to use
1. clone this repository to the environment which can use TensorRT
2. *cd RtSample*
3. *python3 rt_infer.py*

The behavor has been confirmed on Jetson Nano.<br>
First inference may take more time than average inference time.

## How to run your model
1. put your model(.trt) on *model* directory
2. put images which you want to infer on *data* directory
3. rewrite your model's input size (line 32,33), result correspondance table (line 33), model path (line 44), image path (line 133,141)
4. run the code
