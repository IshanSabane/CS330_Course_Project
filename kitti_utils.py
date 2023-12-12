import numpy as np
import pandas as pd
import cv2
import os
import torch
from torch import nn
import random
import itertools
import sys
import time
from typing import Tuple, List
import math



class Draw_Boxes():

    def __init__(self) -> None:
        
        self.calib_path = r"./datasets/kitti/calib/0001.txt"
        self.categories = ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc']
        self.class_id = np.arange(len(self.categories))

        self.class2id = dict(zip(self.categories,self.class_id))

        # labels and images file path
        self.labels_folder_path = r'./datasets/kitti/data_object_labels_2/training/label_2/'
        self.images_folder_path = r'./datasets/kitti/training/image_2/'

        # define some colors in BGR, add more if needed
        self.object_color_dict = {
            'Car': [0, 0, 255], # red
            'Van': [255, 0, 0], # blue
            'Truck': [0,255, 0], # green
            'Pedestrian': [240, 32, 160], # purple
            'Person_sitting': [0, 255, 255], # yellow
            'Cyclist': [255, 0, 255], # magenta
            'Tram': [0, 128, 128], # olive
            'Misc': [[0, 215, 255]] # gold
        }


    def get_sequence_calib(self) -> None:
        """
        returns the camera calibration parameters
        """
        with open(self.calib_path, 'r') as f:
            calib_lines = f.readlines()

        calib = dict()
        calib['P0'] = np.array(calib_lines[0].strip().split(' ')[1:], dtype=np.float32).reshape(3, 4)
        calib['P1'] = np.array(calib_lines[1].strip().split(' ')[1:], dtype=np.float32).reshape(3, 4)
        calib['P2'] = np.array(calib_lines[2].strip().split(' ')[1:], dtype=np.float32).reshape(3, 4)
        calib['P3'] = np.array(calib_lines[3].strip().split(' ')[1:], dtype=np.float32).reshape(3, 4)
        calib['Rect'] = np.array(calib_lines[4].strip().split(' ')[1:], dtype=np.float32).reshape(3, 3)
        calib['Tr_velo_cam'] = np.array(calib_lines[5].strip().split(' ')[1:], dtype=np.float32).reshape(3, 4)
        calib['Tr_imu_velo'] = np.array(calib_lines[6].strip().split(' ')[1:], dtype=np.float32).reshape(3, 4)

        return calib



    def parse_labels_txt(self, label_file, just_parsing = True):
        '''

        Receives a path for a single label file in string format

        Format --> (x1,y1,x2,y2, h_w_l, x_y_z, rotation_y)
        Returns a dict of the above said variables
        '''
        bbox_data_list = []

        with open(label_file, 'r') as f:
            
            file_name = label_file.split('.')[0].split('/')[-1]
            lines = f.read().splitlines()

            if not just_parsing:
                corresponding_image_path = os.path.join(self.images_folder_path, file_name)

            for line in lines:
                
                parts = line.split(' ')

                # 2D bounding Box 
                top_left_corner =[float(x) for x in  parts[4:6]]
                bottom_right_corner =[float(x) for x in  parts[6:8]]


                # bottom_left_corner = parts[4:6]
                # top_right_corner = parts[6:8]
                w = bottom_right_corner[0] - top_left_corner[0]
                h = bottom_right_corner[1] - top_left_corner[1]

                cx = top_left_corner[0] + w/2
                cy = top_left_corner[1] + h/2
                
                # import pdb
                # pdb.set_trace()
               

                # 3D Bounding box
                h_w_l = parts[8:11]
                x_y_z = parts[11:14]

                rotation_y = parts[14]
                object_type = parts[0]

           
                if object_type == "DontCare":
                    continue
                else:
                    temp_bbox_data = {

                        'object_type': object_type,
                        'class_id': self.class2id[object_type],
                        "bbox2D": tuple([int(float(i)) for i in parts[4:8]]),
                        "bbox2Dcwh": tuple([cx,cy,w,h]), 
                        "bbox3D": tuple([float(i) for i in x_y_z]+[float(i) for i in h_w_l]+[float(rotation_y)]),                        
                        'rotation_y': float(rotation_y)  # need to check the arithmetic value
                    }

                  
                    bbox_data_list.append(temp_bbox_data)

        if just_parsing:
            return bbox_data_list
        else:
            return bbox_data_list, corresponding_image_path, file_name



    def transform_3dbox_to_image(self, dimension, location, rotation, calib=None) -> np.array:
        """
        convert the 3d box to coordinates in pointcloud
        :param dimension: height, width, and length
        :param location: x, y, and z
        :param rotation: rotation parameter
        :return: transformed coordinates
        """
        # print("indide the bpx")
        calib = self.get_sequence_calib()
        height, width, length = dimension
        x, y, z = location
        x_corners = [length / 2, length / 2, -length / 2, -length / 2, length / 2, length / 2, -length / 2, -length / 2]
        y_corners = [0, 0, 0, 0, -height, -height, -height, -height]
        z_corners = [width / 2, -width / 2, -width / 2, width / 2, width / 2, -width / 2, -width / 2, width / 2]

        corners_3d = np.vstack([x_corners, y_corners, z_corners])

      

        # transform 3d box based on rotation along Y-axis
        R_matrix = np.array([[math.cos(rotation), 0, math.sin(rotation)],
                            [0, 1, 0],
                            [-math.sin(rotation), 0, math.cos(rotation)]])

        corners_3d = np.dot(R_matrix, corners_3d).T

        # shift the corners to from origin to location
        corners_3d = corners_3d + np.array([x, y, z])

        # only show 3D bounding box for objects in front of the camera
        # if np.any(corners_3d[:, 2] < 0.1):
        #     corners_3d_img = None
        # else:
        # from camera coordinate to image coordinate
        corners_3d_temp = np.concatenate((corners_3d, np.ones((8, 1))), axis=1)
        corners_3d_img = np.matmul(corners_3d_temp, calib['P2'].T)
        corners_3d_img = corners_3d_img[:, :2] / corners_3d_img[:, 2][:, None]
        
        # print("Corners 3d img: ", corners_3d_img)

        return corners_3d_img



    def show_sequence_rgb(self, label_file_path, img_path = None, vis_2dbox=False, vis_3dbox=False, save_img=False, save_path=None, wait_time=30):
        """
        image_label --> pass the parsed label file where each element in the list corresponds to the object that is to be detected in the corresponding "image_path"
        """

        bbox_data_list, image_path, sequence_name = self.parse_labels_txt(label_file_path, just_parsing=False)

        if img_path: 
            image_path = img_path

        calib = self.get_sequence_calib()

        # assert len(img_list) == len(labels), 'The number of image and number of labels do NOT match!'
        assert not(vis_2dbox == True and vis_3dbox == True), 'It is NOT good to visualize both 2D and 3D boxes simultaneously!'

        # create folder to save image if not existing
        if save_img:
            if save_path is None:
                if vis_2dbox:
                    save_path = os.path.join(f'./logs/kitti/vis/2D_box')
                elif vis_3dbox:
                    save_path = os.path.join(f'./logs/kitti/vis/3D_box')
                else:
                    print('Incorrect Parameters Passed')

        
        # to read the image
        img = cv2.imread(image_path + ".png")   # BGR image format
        thickness = 2

        cv2.imshow("Original image", img)

        
        # visualize 2d boxes in the image
        if vis_2dbox:
            save_path = os.path.join(save_path, sequence_name + ".png")
            
            # to iterate through multiple detections as an image can have multiple objects
            for object in bbox_data_list:
                
                object_type = object['object_type']
                bbox_color = self.object_color_dict[object_type]

                # to get the dimensions of the ground truth bounding box
                left_corner = object['bbox'][0:2]
                right_corner = object['bbox'][2:]
                print(left_corner,right_corner)

                cv2.rectangle(img, left_corner, right_corner, color=bbox_color, thickness=thickness)

                cv2.putText(img, text=object_type, org=(left_corner[0] , left_corner[1]),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4, color=bbox_color, thickness=thickness)
                
            cv2.imshow("2D box", img)

        
        # visualize 3d boxes in the image
        if vis_3dbox:
            save_path = os.path.join(save_path, sequence_name+".png")
            # load and show object bboxes
            for object in bbox_data_list:
                object_type = object['object_type']
                bbox_color = self.object_color_dict[object_type]

                # to get the dimensions of the ground truth bounding box
                h_w_l = object['h_w_l']
                x_y_z = object['x_y_z']
                rotation_y = object['rotation_y']

               
                corners_3d_img = self.transform_3dbox_to_image(h_w_l, x_y_z, rotation_y, calib)

                if corners_3d_img is None:
                    raise AssertionError("Something is wrong with the transform_3dbox_to_image function as it returned a None")
                else:
                    corners_3d_img = corners_3d_img.astype(int)

                    # draw lines in the image
                    # p10-p1, p1-p2, p2-p3, p3-p0
                    cv2.line(img, (corners_3d_img[0, 0], corners_3d_img[0, 1]),
                        (corners_3d_img[1, 0], corners_3d_img[1, 1]), color=bbox_color, thickness=thickness)
                    cv2.line(img, (corners_3d_img[1, 0], corners_3d_img[1, 1]),
                        (corners_3d_img[2, 0], corners_3d_img[2, 1]), color=bbox_color, thickness=thickness)
                    cv2.line(img, (corners_3d_img[2, 0], corners_3d_img[2, 1]),
                        (corners_3d_img[3, 0], corners_3d_img[3, 1]), color=bbox_color, thickness=thickness)
                    cv2.line(img, (corners_3d_img[3, 0], corners_3d_img[3, 1]),
                        (corners_3d_img[0, 0], corners_3d_img[0, 1]), color=bbox_color, thickness=thickness)

                    # p4-p5, p5-p6, p6-p7, p7-p0
                    cv2.line(img, (corners_3d_img[4, 0], corners_3d_img[4, 1]),
                        (corners_3d_img[5, 0], corners_3d_img[5, 1]), color=bbox_color, thickness=thickness)
                    cv2.line(img, (corners_3d_img[5, 0], corners_3d_img[5, 1]),
                        (corners_3d_img[6, 0], corners_3d_img[6, 1]), color=bbox_color, thickness=thickness)
                    cv2.line(img, (corners_3d_img[6, 0], corners_3d_img[6, 1]),
                        (corners_3d_img[7, 0], corners_3d_img[7, 1]), color=bbox_color, thickness=thickness)
                    cv2.line(img, (corners_3d_img[7, 0], corners_3d_img[7, 1]),
                        (corners_3d_img[4, 0], corners_3d_img[4, 1]), color=bbox_color, thickness=thickness)

                    # p0-p4, p1-p5, p2-p6, p3-p7
                    cv2.line(img, (corners_3d_img[0, 0], corners_3d_img[0, 1]),
                        (corners_3d_img[4, 0], corners_3d_img[4, 1]), color=bbox_color, thickness=thickness)
                    cv2.line(img, (corners_3d_img[1, 0], corners_3d_img[1, 1]),
                        (corners_3d_img[5, 0], corners_3d_img[5, 1]), color=bbox_color, thickness=thickness)
                    cv2.line(img, (corners_3d_img[2, 0], corners_3d_img[2, 1]),
                        (corners_3d_img[6, 0], corners_3d_img[6, 1]), color=bbox_color, thickness=thickness)
                    cv2.line(img, (corners_3d_img[3, 0], corners_3d_img[3, 1]),
                        (corners_3d_img[7, 0], corners_3d_img[7, 1]), color=bbox_color, thickness=thickness)

                    # draw front lines
                    cv2.line(img, (corners_3d_img[0, 0], corners_3d_img[0, 1]),
                            (corners_3d_img[5, 0], corners_3d_img[5, 1]), color=bbox_color, thickness=thickness)
                    cv2.line(img, (corners_3d_img[1, 0], corners_3d_img[1, 1]),
                            (corners_3d_img[4, 0], corners_3d_img[4, 1]), color=bbox_color, thickness=thickness)

                    cv2.putText(img, text=object_type, org=(corners_3d_img[4, 0]+5, corners_3d_img[4, 1]),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4, color=bbox_color, thickness=thickness)

            cv2.imshow('3D Bounding Box', img)

        if save_img:
            print('Save path: ', save_path)
            cv2.imwrite(save_path, img)

        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

        cv2.destroyAllWindows()










