# -*- coding: utf-8 -*-
"""
-------------------------------------------------
Project Name: yolov5
File Name: window.py.py
Author: 肆十二
Description：图形化界面，可以检测摄像头、视频和图片文件
-------------------------------------------------
"""
import copy
import os
# 导入库
import shutil
import threading
import sys
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
import os.path as osp
from .models.common import DetectMultiBackend
from .utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from .utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from .utils.plots import Annotator, colors, save_one_box
from .utils.torch_utils import select_device, time_sync
import time
import numpy as np


# 在原先的基础上输出实际的信息堆积对
# 添加历史记录功能。
class MainWindow():
    # 基本配置不动，然后只动第三个界面
    def __init__(self):
        # 初始化界面
        super().__init__()
        self.output_size = 480
        self.img2predict = ""
        self.device = 'cpu'
        self.init_vid_id = '0'
        # self.vid_source = int(self.init_vid_id)
        self.vid_source = self.init_vid_id
        self.stopEvent = threading.Event()
        self.webcam = False
        self.stopEvent.clear()
        self.model = self.model_load(weights="F:/Huawei/Internship/Backend/WheatRecogBE/utils/yolomodule/best.pt", device=self.device)  # todo 指明模型加载的位置的设备
        self.conf_thres = 0.25  # confidence threshold
        self.iou_thres = 0.45  # NMS IOU thresholdv
        self.vid_gap = 1  # 摄像头视频帧保存间隔。


    # 模型初始化
    @torch.no_grad()
    def model_load(self, weights="",  # model.pt path(s)
                   device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
                   half=False,  # use FP16 half-precision inference
                   dnn=False,  # use OpenCV DNN for ONNX inference
                   ):
        device = select_device(device)
        half &= device.type != 'cpu'  # half precision only supported on CUDA
        model = DetectMultiBackend(weights, device=device, dnn=dnn)
        stride, names, pt, jit, onnx = model.stride, model.names, model.pt, model.jit, model.onnx
        # Half
        half &= pt and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
        if pt:
            model.model.half() if half else model.model.float()
        # print(model.onnx)
        print("模型加载完成，模型信息如上!")
        return model

    # 图片检测
    def detect_img(self,source,filerename):
        txt_results = []
        model = self.model
        output_size = self.output_size
        imgsz = [640, 640]  # inference size (pixels)
        conf_thres = self.conf_thres  # confidence threshold
        iou_thres = self.iou_thres  # NMS IOU threshold
        max_det = 1000  # maximum detections per image
        # device = self.device  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img = False  # show results
        save_txt = False  # save results to *.txt
        save_conf = False  # save confidences in --save-txt labels
        save_crop = False  # save cropped prediction boxes
        nosave = False  # do not save images/videos
        classes = None  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms = False  # class-agnostic NMS
        augment = False  # ugmented inference
        visualize = False  # visualize features
        line_thickness = 3  # bounding box thickness (pixels)
        hide_labels = False  # hide labels
        hide_conf = False  # hide confidences
        half = False  # use FP16 half-precision inference
        dnn = False  # use OpenCV DNN for ONNX inference
        print(source)
        device = select_device(self.device)
        webcam = False
        stride, names, pt, jit, onnx = model.stride, model.names, model.pt, model.jit, model.onnx
        imgsz = check_img_size(imgsz, s=stride)  # check image size
        save_img = not nosave and not source.endswith('.txt')  # save inference images
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt and not jit)
        bs = 1  # batch_size
        vid_path, vid_writer = [None] * bs, [None] * bs
        # Run inference
        if pt and device.type != 'cpu':
            model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.model.parameters())))  # warmup
        dt, seen = [0.0, 0.0, 0.0], 0
        for path, im, im0s, vid_cap, s in dataset:
            t1 = time_sync()
            im = torch.from_numpy(im).to(device)
            im = im.half() if half else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            t2 = time_sync()
            dt[0] += t2 - t1
            pred = model(im, augment=augment, visualize=visualize)
            t3 = time_sync()
            dt[1] += t3 - t2
            # NMS
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
            dt[2] += time_sync() - t3
            for i, det in enumerate(pred):  # per image
                seen += 1
                if webcam:  # batch_size >= 1
                    p, im0, frame = path[i], im0s[i].copy(), dataset.count
                    s += f'{i}: '
                else:
                    p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
                p = Path(p)  # to Path
                s += '%gx%g ' % im.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                imc = im0.copy() if save_crop else im0  # for save_crop
                annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        # todo 记录坐标信息
                        location = list(torch.tensor(xyxy).view(1, 4).cpu().numpy()[0])
                        re = [names[int(cls)], conf.cpu().numpy()] + location
                        txt_results.append(re)
                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(
                                -1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        if save_img or save_crop or view_img:  # Add bbox to image
                            c = int(cls)  # integer class
                            label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                            annotator.box_label(xyxy, label, color=colors(c, True))
                LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')
                im0 = annotator.result()
                im_record = copy.deepcopy(im0)
                resize_scale = output_size / im0.shape[0]
                im0 = cv2.resize(im0, (0, 0), fx=resize_scale, fy=resize_scale)
                # cv2.imwrite("images/tmp/single_result.jpg", im0)

                # 把这里的结果进行保存，按照时间进行保存
                time_re = str(filerename)
                cv2.imwrite("F:/Huawei/Internship/Backend/WheatRecogBE/app/static/outcome/{}".format(time_re), im_record)
                return len(txt_results)

    def detect_vid(self, source, filerename):
        all_counts = []
        model = self.model
        output_size = self.output_size
        imgsz = [640, 640]  # inference size (pixels)
        conf_thres = self.conf_thres  # confidence threshold
        iou_thres = self.iou_thres  # NMS IOU threshold
        max_det = 1000  # maximum detections per image
        view_img = False  # show results
        save_txt = False  # save results to *.txt
        save_conf = False  # save confidences in --save-txt labels
        save_crop = False  # save cropped prediction boxes
        nosave = False  # do not save images/videos
        classes = None  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms = False  # class-agnostic NMS
        augment = False  # augmented inference
        visualize = False  # visualize features
        line_thickness = 3  # bounding box thickness (pixels)
        hide_labels = False  # hide labels
        hide_conf = False  # hide confidences
        half = False  # use FP16 half-precision inference
        dnn = False  # use OpenCV DNN for ONNX inference
        webcam = self.webcam
        size = (imgsz[1], imgsz[0])  # 获取图片宽高度信息
        print(size)
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        videoWrite = cv2.VideoWriter(f"F:/Huawei/Internship/Backend/WheatRecogBE/app/static/outcome/{filerename}", fourcc, self.vid_gap, size)  # 确保 fps = 1.0
        device = select_device(self.device)
        stride, names, pt, jit, onnx = model.stride, model.names, model.pt, model.jit, model.onnx
        imgsz = check_img_size(imgsz, s=stride)  # check image size
        save_img = not nosave and not source.endswith('.txt')  # save inference images

        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt and not jit)
        fps = dataset.cap.get(cv2.CAP_PROP_FPS)  # 获取视频的帧率
        print(f"视频帧率: {fps}")

        frame_interval = int(fps)  # 每秒检测一次，跳过不必要的帧

        if pt and device.type != 'cpu':
            model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.model.parameters())))  # warmup

        dt, seen = [0.0, 0.0, 0.0], 0
        for frame_idx, (path, im, im0s, vid_cap, s) in enumerate(dataset):
            if frame_idx % frame_interval != 0:  # 跳过不必要的帧
                continue

            t1 = time_sync()
            im = torch.from_numpy(im).to(device)
            im = im.half() if half else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            t2 = time_sync()
            dt[0] += t2 - t1
            # Inference
            pred = model(im, augment=augment, visualize=visualize)
            t3 = time_sync()
            dt[1] += t3 - t2
            # NMS
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
            dt[2] += time_sync() - t3
            # Process predictions
            for i, det in enumerate(pred):  # per image
                txt_results = []
                seen += 1
                if webcam:  # batch_size >= 1
                    p, im0, frame = path[i], im0s[i].copy(), dataset.count
                    s += f'{i}: '
                else:
                    p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
                p = Path(p)  # to Path
                s += '%gx%g ' % im.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                imc = im0.copy() if save_crop else im0  # for save_crop
                annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        location = list(torch.tensor(xyxy).view(1, 4).cpu().numpy()[0])
                        re = [names[int(cls)], conf.cpu().numpy()] + location
                        txt_results.append(re)
                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(
                                -1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        if save_img or save_crop or view_img:  # Add bbox to image
                            c = int(cls)  # integer class
                            label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                            annotator.box_label(xyxy, label, color=colors(c, True))
                LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')
                im0 = annotator.result()
                im_record = copy.deepcopy(im0)
                frame = im0
                resize_scale = output_size / frame.shape[0]
                frame_resized = cv2.resize(frame, (0, 0), fx=resize_scale, fy=resize_scale)

                # 保存结果
                if frame_idx % (frame_interval * self.vid_gap) == 0:
                    frame_suitable = cv2.resize(frame_resized, (size[0], size[1]), interpolation=cv2.INTER_CUBIC)
                    videoWrite.write(frame_suitable)  # 将图片写入所创建的视频对象
                
                # 保存到
                all_counts.append(len(txt_results))

            if cv2.waitKey(25) & self.stopEvent.is_set() == True:
                self.stopEvent.clear()
                self.webcam_detection_btn.setEnabled(True)
                self.mp4_detection_btn.setEnabled(True)
                if dataset.cap != None:
                    dataset.cap.release()
                videoWrite.release()  # 确保视频文件被正确关闭
                self.reset_vid()
                break

        # 确保视频文件被正确关闭
        if videoWrite.isOpened():
            videoWrite.release()

        print("Video processing complete and saved to 'output.mp4'")
        return all_counts

    def check_record(self):
        os.startfile(osp.join(os.path.abspath(os.path.dirname(__file__)),"record"))

    # 视频线程关闭
    def close_vid(self):
        self.stopEvent.set()
        self.reset_vid()


