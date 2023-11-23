import os
import sys
from pathlib import Path
import torch
import rospy
import numpy as np

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from vision_msgs.msg import Detection2DArray, Detection2D, BoundingBox2D
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import torch.backends.cudnn as cudnn
from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.segment.general import process_mask, scale_masks
from utils.segment.plots import plot_masks_only
from utils.torch_utils import select_device, smart_inference_mode
from utils.augmentations import letterbox


class Segmentator:
    def __init__(self) -> None:
        self.weights = rospy.get_param("~weights", "./yolov7-seg.pt")
        print(self.weights)
        self.data = ROOT / 'data/coco128.yaml'
        self.img_topic = rospy.get_param("~img_topic", "image_raw")
        self.conf_thres = rospy.get_param("~conf_thres", 0.25)
        self.iou_thres = rospy.get_param("~iou_thres", 0.45)
        self.img_size = rospy.get_param("~img_size", 640)
        self.half = False
        self.device = ''
        self.dnn = False

        rospy.loginfo("Loading YOLO Model")
        self.device = select_device(self.device)
        self.model = DetectMultiBackend(self.weights, device=self.device, dnn=self.dnn, data=self.data, fp16=self.half)
        stride, self.names, pt = self.model.stride, self.model.names, self.model.pt
        self.imgsz = check_img_size(self.img_size, s=stride)  # check image size
        rospy.loginfo("YOLO Model loaded")
        self.bridge = CvBridge()
        self.img_publisher = rospy.Publisher(
            '/segmentation', Image, queue_size=1
        )
        self.img_subscriber = rospy.Subscriber(
            '/image_raw', Image, self.process_msg
        )

        pass
    
    def process_msg(self, img_msg: Image):
        print("imgreceived")
        np_img = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')

        #image pre-processing
        im0 = np_img
        im = letterbox(im0, self.imgsz, stride=self.model.stride, auto=self.model.pt)[0]  # padded resize
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)  # contiguous
        im = torch.from_numpy(im).to(self.device)
        im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        #inference
        pred, out = self.model(im, visualize=False)
        proto = out[1]

        #NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, 0, False, max_det=300, nm=32) #0 to only detect person

        #prediction processing
        for i, det in enumerate(pred):  # per image
            annotator = Annotator(im0, line_width=3, example=str(self.names))
            if len(det):
                masks = process_mask(proto[i], det[:, 6:], det[:, :4], im.shape[2:], upsample=True)  # HWC
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                # Mask plotting ----------------------------------------------------------------------------------------
                mcolors = [colors(int(cls), True) for cls in det[:, 5]]
                im_masks = plot_masks_only(im[i], masks)  # image with masks shape(imh,imw,3)
                annotator.im = scale_masks(im.shape[2:], im_masks, im0.shape)  # scale to original h, w
                # Mask plotting ----------------------------------------------------------------------------------------
            else : 
                print("noDetection")
            im0 = annotator.result()
            seg_msg = self.bridge.cv2_to_imgmsg(im0, encoding="bgr8")
            self.img_publisher.publish(seg_msg)
            cv2.imshow("result", im0)
            cv2.waitKey(1)  # 1 millisecond



if __name__ == "__main__":
    rospy.init_node("Segmentator_node")
    Publisher = Segmentator()
    rospy.spin()




