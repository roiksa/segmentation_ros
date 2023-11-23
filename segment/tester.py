#!/usr/bin/python3

import cv2
import numpy as np
import rospy
import time
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class ImageViewer:
    def __init__(self):
        self.bridge = CvBridge()
        self.img_subscriber = rospy.Subscriber(
            '/usb_cam/image_raw', Image, self.process_img_msg
        )
        rospy.loginfo("Subscribed to /image, ready to show image")

    def process_img_msg(self, img_msg: Image):
        self.id+=1
        """ callback function for publisher """
        np_img = self.bridge.imgmsg_to_cv2(
            img_msg, desired_encoding='bgr8'
        )
        print('fps: ',self.id/ (time.time()- self.start ))

if __name__ == "__main__":
    rospy.init_node("ImageViewer_node")
    Viewer = ImageViewer()
    rospy.spin()