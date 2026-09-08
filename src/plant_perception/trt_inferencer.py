#!/usr/bin/env python3
"""
trt_inferencer.py — Native TensorRT inference engine wrapper for YOLO-Seg-Root.
Loads a pre-built serialized .engine file directly via the TensorRT runtime API
(precision is whatever the engine was exported with) and handles letterboxing
and forward inference.
"""

import os
import struct
import time
import numpy as np
import cv2
import tensorrt as trt
import torch

_TRT_TO_TORCH_DTYPE = {
    trt.DataType.FLOAT: torch.float32,
    trt.DataType.HALF: torch.float16,
    trt.DataType.INT32: torch.int32,
    trt.DataType.INT8: torch.int8,
    trt.DataType.BOOL: torch.bool,
}


class TRTInferencer:
    """Load a serialized TensorRT .engine and run inference with automatic letterboxing."""

    def __init__(self, model_path, device="tensorrt", imgsz=640, trt_fp16_enable=False):
        self.model_path = os.path.abspath(model_path)
        self.device = device.lower()
        self.imgsz = imgsz
        self.trt_fp16_enable = trt_fp16_enable  # unused — precision is fixed at engine export time

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Engine not found: {self.model_path}")

        with open(self.model_path, "rb") as f:
            raw = f.read()

        logger = trt.Logger(trt.Logger.WARNING)
        self.trt_runtime = trt.Runtime(logger)
        engine = self.trt_runtime.deserialize_cuda_engine(raw)
        if engine is None:
            # Ultralytics-exported engines prepend a length-prefixed JSON metadata
            # blob before the actual serialized plan; strip it and retry.
            meta_len = struct.unpack("<i", raw[:4])[0]
            engine = self.trt_runtime.deserialize_cuda_engine(raw[4 + meta_len:])
        if engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine: {self.model_path}")

        self.trt_engine = engine
        self.trt_context = engine.create_execution_context()

        self.input_name = None
        self.output_names = []
        self._trt_tensors = {}

        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            shape = tuple(engine.get_tensor_shape(name))
            dtype = _TRT_TO_TORCH_DTYPE[engine.get_tensor_dtype(name)]
            tensor = torch.empty(shape, dtype=dtype, device="cuda")
            self._trt_tensors[name] = tensor
            self.trt_context.set_tensor_address(name, tensor.data_ptr())
            if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_name = name
            else:
                self.output_names.append(name)

        print(f"[TRTInferencer] Loaded native TensorRT engine: {self.model_path}  Device: {self.device.upper()}")
        print(f"[TRTInferencer] Input:   {self.input_name} {tuple(self._trt_tensors[self.input_name].shape)} "
              f"{self._trt_tensors[self.input_name].dtype}")
        print(f"[TRTInferencer] Outputs: "
              f"{[(n, tuple(self._trt_tensors[n].shape), str(self._trt_tensors[n].dtype)) for n in self.output_names]}")

    def letterbox(self, img, color=(114, 114, 114)):
        """Resize with aspect-ratio preservation and pad to self.imgsz."""
        h, w = img.shape[:2]
        r = min(self.imgsz / h, self.imgsz / w)
        new_unpad = (int(round(w * r)), int(round(h * r)))

        dw = (self.imgsz - new_unpad[0]) / 2
        dh = (self.imgsz - new_unpad[1]) / 2

        img_resized = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

        img_padded = cv2.copyMakeBorder(
            img_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
        )
        return img_padded, r, (dw, dh)

    def preprocess(self, img_bgr):
        """Letterbox → RGB → CHW float32 tensor normalised to [0, 1]."""
        img_lb, ratio, pad = self.letterbox(img_bgr)
        blob = cv2.cvtColor(img_lb, cv2.COLOR_BGR2RGB)
        blob = blob.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32) / 255.0
        return blob, ratio, pad

    def run(self, img_bgr):
        """Run full inference pipeline. Returns (outputs, ratio, pad, ms)."""
        blob, ratio, pad = self.preprocess(img_bgr)

        t0 = time.perf_counter()
        input_tensor = self._trt_tensors[self.input_name]
        input_tensor.copy_(torch.from_numpy(blob))
        self.trt_context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        outputs = [self._trt_tensors[name].cpu().numpy() for name in self.output_names]
        inference_ms = (time.perf_counter() - t0) * 1000.0

        return outputs, ratio, pad, inference_ms


# Backwards compatibility alias
ONNXInferencer = TRTInferencer
