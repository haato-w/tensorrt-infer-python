
#reference: https://forums.developer.nvidia.com/t/run-peoplenet-with-tensorrt/128000/21


import os
import time
import numpy as np
import cv2
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
from math import ceil



class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()



class RtInfer:
    def __init__(self, logger=None):
        self.image_h = 32 # please rewrite with your model's size
        self.image_w = 32 # please rewrite with your model's size
        # result_table is an correspondence table between model output and the result you want
        self.result_table = ['A', 'E', 'HA', 'HI', 'HO', 'HU', 'I', 'KA', 'KE', 'KI', 'KO', 'KU', 'MA', 'ME', 'MI', 'MO', 'MU', 
                                'NA', 'NE', 'NI', 'NO', 'NU', 'RA', 'RE', 'RI', 'RO', 'RU', 'SA', 'SE', 'SO', 'SU', 'TA', 'TE', 'TI', 
                                'TO', 'TU', 'U', 'WA', 'WO', 'YA', 'YU']# please rewrite with your model's table

        if logger == None:
            # TensorRT logger singleton
            TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        else:
            TRT_LOGGER = logger

        trt_engine_path_hiragana = os.path.join("model/model_for_hiragana.trt") # please write with your model path

        trt_runtime_hiragana = trt.Runtime(TRT_LOGGER)

        load_start = time.time()
        trt_engine_hiragana = self.load_engine(trt_runtime_hiragana, trt_engine_path_hiragana)
        print("model load time: ", time.time()-load_start)


        # This allocates memory for network inputs/outputs on both CPU and GPU
        allocate_start = time.time()
        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers(trt_engine_hiragana)

        self.context_hira = trt_engine_hiragana.create_execution_context()
        print("memory allocate time: ", time.time()-allocate_start)


    def load_engine(self, trt_runtime, engine_path):
        with open(engine_path, "rb") as f:
            engine_data = f.read()
        engine = trt_runtime.deserialize_cuda_engine(engine_data)
        return engine


    def allocate_buffers(self, engine, batch_size=1):
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()

        for binding in engine:
            size = trt.volume(engine.get_binding_shape(binding)) * batch_size
            dtype = np.float32
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            bindings.append(int(device_mem))
            # Append to the appropriate list.
            if engine.binding_is_input(binding):
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))
        return inputs, outputs, bindings, stream


    def image_preprocess(self, img):
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        dst2 = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 30)

        img_resized = cv2.resize(dst2, (self.image_w, self.image_h))

        return img_resized


    # inputs and outputs are expected to be lists of HostDeviceMem objects.
    def do_inference(self, context, bindings, inputs, outputs, stream, batch_size=1):
        # Transfer input data to the GPU.
        [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
        # Run inference.
        context.execute_async(
            batch_size=batch_size, bindings=bindings, stream_handle=stream.handle
        )
        # Transfer predictions back from the GPU.
        [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
        # Synchronize the stream
        stream.synchronize()
        # Return only the host outputs.
        return [out.host for out in outputs]


    def infer(self, image):
        image = self.image_preprocess(image)
        img = image.reshape(self.image_h, self.image_w, 1).astype('float32')/255
        np.copyto(self.inputs[0].host, img.ravel())

        # Fetch output from the model
        result = self.do_inference(
            self.context_hira, bindings=self.bindings, inputs=self.inputs, outputs=self.outputs, stream=self.stream
        )

        return result



if __name__ == '__main__':
    rt_infer = RtInfer()

    process_start = time.time()
    image = cv2.imread("data/sample.jpg") # please rewrite with your image path
    result = rt_infer.infer(image)
    print(result)
    string = rt_infer.result_table[result[0].argmax()]
    print("result: ", string)
    print("process time: ", time.time()-process_start)

    process_start = time.time()
    image = cv2.imread("data/sample2.jpg")# please rewrite with your image path
    result = rt_infer.infer(image)
    print(result)
    string = rt_infer.result_table[result[0].argmax()]
    print("result: ", string)
    print("process time: ", time.time()-process_start)

   