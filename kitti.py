import numpy as np
import pandas as pd
import cv2
import os
import torch
from torch import nn
import random
import itertools
from torch.utils.data import IterableDataset
import sys
import time
from typing import Tuple, List
import math
import imageio
from tqdm import tqdm
from torchvision import transforms
from torchvision.io import read_image
from kitti_utils import Draw_Boxes
from transformers import DetrImageProcessor
from torch.utils.data import Sampler, Dataset,DataLoader
import PIL as Image
from transformers import DPTForDepthEstimation,DPTImageProcessor
from visualisation import visualize_image_with_boxes
import pdb


class  DataGenerator(IterableDataset):
    """
    Data Generator capable of generating batches of Kitti data.
    A "class" is considered as any one object category from the Kitti dataset.
    """

    def __init__(
        self,
        num_classes,
        num_samples_per_class,
        class_names,
        meta_train,
        cache=True,
        images_folder_path = "./datasets/kitti/all_data/",
        resize = (224,224),
        bounding_box_mode = None

    ):
        """
        Args:
            num_classes: Number of classes for classification (N-way)
            k: num samples to generate per class in one batch (K+1)
            images_folder_path: all_data folder path where the preproecssed files are already stored
            class_names: if you want to specifically extract certain classes (usually keep it as None)
            cache: whether to cache the images loaded
        """
        self.iteration_count = 0
        self.num_samples_per_class = num_samples_per_class
        self.n_queries = 1 # Add this later into the code.
        self.num_classes = num_classes
        self.class_names = class_names
        self.images_folder_path = images_folder_path
        self.resize = resize
        self.bounding_box_mode = bounding_box_mode

        self.preprocessor = DetrImageProcessor.from_pretrained('facebook/detr-resnet-50')
        self.preprocessor.size = resize
        self.depth_preprocessor = DPTImageProcessor.from_pretrained("Intel/dpt-large")
        self.depth_estimator = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")
       
        # Fixed Splits. TODO: Randomize this for multiple expirements.
        self.train_foldernames = ['Car', 'Truck', 'Pedestrian', 'Cyclist']
        self.test_foldernames = ['Van', 'Tram', 'Person_sitting']

        # folder names
        self.all_class_folders = os.listdir(self.images_folder_path)
      
        # to instantiate the draw boxes class
        self.draw_boxes = Draw_Boxes()

        # to enable caching
        self.image_caching = cache
        self.stored_images = {}
        
        # if meta-train or test
        if meta_train:
            self.meta_train = meta_train
        else:
            self.meta_train = False

        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'



    # to get the absolute paths of images and labels
    def extract_images_labels_paths(self, labels):

        image_paths = []
        label_paths = []

        for name in labels:

            path = os.path.join(self.images_folder_path, name)
            image_list = os.listdir(path)
            
            if 'labels' in image_list:
                image_list.remove('labels')


            for i in random.choices(range(len(image_list)),k = self.num_samples_per_class):
                filename = image_list[i]
                # print("Filename: ", filename)
                image_paths.append(os.path.join(path, image_list[i]))

                label_filename = os.path.join(path, 'labels', filename.split('.')[0]+".txt")
                # print("Label filename: ", label_filename)
                label_paths.append(label_filename)

        return image_paths, label_paths


    def image_to_tensor(self, filename):
        """
        Takes an image path and returns numpy array
        Args:
            filename: Image filename
            dim_input: Flattened shape of image
        Returns:
            1 channel image
        """
        if self.image_caching and (filename in self.stored_images):
            return self.stored_images[filename]
        image = cv2.imread(filename) # (C,H,W)

        if self.image_caching:
            self.stored_images[filename] = image

        return image
    

    def preprocessing(self, image_path, annotations_path, depth = False, just_resize=False):

        image = self.image_to_tensor(image_path)
        box_list = self.draw_boxes.parse_labels_txt(annotations_path)
        # Do we need calib for each image? 

        transform = transforms.Compose([
                                        transforms.ToPILImage(),
                                        transforms.Resize(self.resize),
                                        transforms.ToTensor()
                                        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the image
                                        ])

        resized_image = transform(image)

        #CV2 HW Image Format
        image_size = image.shape[:2]

        # Calculate the scaling factors
        scale_x = 1 / image_size[1]    # Width
        scale_y = 1 / image_size[0]    # Height
        calib = self.draw_boxes.get_sequence_calib()
        scaler = np.array([scale_x, scale_y])

        # Resize the bounding boxes
        annotations = {'labels':[], 'bbox2D':[], 'bbox2Dcwh':[], 'bbox3D':[]}
        annotations['image_size'] = torch.tensor([image_size])

        for box in box_list:
            
            x1, y1, x2, y2 = box['bbox2D']
            resized_x1 = float(x1 * scale_x)
            resized_y1 = float(y1 * scale_y)
            resized_x2 = float(x2 * scale_x)
            resized_y2 = float(y2 * scale_y)
            annotations['bbox2D'].append(np.array([resized_x1, resized_y1, resized_x2, resized_y2]))
            
            cx, cy, w, h = box['bbox2Dcwh']
            resized_cx = float(cx * scale_x)
            resized_cy = float(cy * scale_y)
            resized_w = float(w * scale_x)
            resized_h = float(h * scale_y)
            annotations['bbox2Dcwh'].append(np.array([resized_cx, resized_cy, resized_w, resized_h]))

            # x,y,z,h,w,l,alpha = box['bbox3D']
            annotations['bbox3D'].append(box['bbox3D']) 

            # self.draw_boxes.transform_3dbox_to_image([h,w,l],)

            # resized_x = float(x * scale_x)
            # resized_y = float(y * scale_y)
            # resized_w = float(w * scale_x)
            # resized_h = float(h * scale_y)
            # resized_z = float(z*scale_x*scale_y)
            # resized_l = float(l*scale_x*scale_y)

            # coordinates3D = np.array([resized_x, resized_y, resized_w, resized_h, resized_z, resized_l, alpha/(2*np.pi)])

            # annotations['bbox3D'].append(coordinates3D) 
            
            # coordinates3D = self.draw_boxes.transform_3dbox_to_image([h,w,l],[x,y,z],
            #                                                          box['rotation_y'],
            #                                                          calib)
            # coordinates3D *=scaler
            # coordinates3D = np.array([resized_x,resized_y,resized_z,resized_h,resized_w,resized_l,alpha])
            

            annotations['labels'].append(self.label2id[box['object_type']])            


        annotations['bbox2D'] = torch.tensor(np.stack(annotations['bbox2D']))
        annotations['bbox2Dcwh'] = torch.tensor(np.stack(annotations['bbox2Dcwh']))

        annotations['bbox3D'] = torch.tensor(np.stack(annotations['bbox3D']))
        annotations['labels'] = torch.tensor(np.array(annotations['labels']))


        

        return resized_image, annotations


    def _sample(self):
        """
        Samples a batch for training, validation, or testing
        Returns:
            A tuple of (1) Image batch and (2) Label batch:
                1. image batch has shape [K+1, N, 784] and is a numpy array
                2. label batch has shape [K+1, N, N] and is a numpy array
            where K is the number of "shots", N is number of classes
        """
        # print(self.class_names, self.meta_train)
        
        if self.class_names is None:
            if self.meta_train:
                class_folders_names = random.sample(self.train_foldernames, self.num_classes)
            else:
                class_folders_names = random.sample(self.test_foldernames, self.num_classes)
        else:
            pass

        image_paths_list, label_paths_list = self.extract_images_labels_paths(class_folders_names)
        
        print("Lengths of image and label paths: ", len(image_paths_list), len(label_paths_list))

        # Label Zero Corresponds to No Object Detected
        self.label2id = dict(zip(['No Object'] + class_folders_names, np.arange(0, len(class_folders_names)+1)  ))
        self.id2label = dict(zip(np.arange(0,len(class_folders_names)+1), ['No Object'] + class_folders_names))

        image_batch = []
        label_batch = []

        support_image_batch = []
        support_label_batch = []

        query_image_batch = []
        query_label_batch = []


        # to get the query ids
        query_idx_list = []
        for i in range(self.num_classes):
            query_idx = i * self.num_samples_per_class + self.num_samples_per_class - 1
            if query_idx > len(image_paths_list):
                break
            else:
                query_idx_list.append(query_idx)
        

        for i, (image_path, label_path) in enumerate(zip(image_paths_list, label_paths_list)):

            image, annotations = self.preprocessing(image_path,label_path)

            if False:
                visualize_image_with_boxes(image, annotations['bbox2D'],image_size = None)
        

            if self.bounding_box_mode == "2D_to_3D":
                
                if i in query_idx_list:

                    query_image_batch.append(image)
                    
                    annotations['bbox'] = annotations['bbox3D']
                    del annotations['bbox2D']
                    del annotations['bbox3D']
                    
                    query_label_batch.append(annotations)
                else:
                    support_image_batch.append(image)
                    
                    annotations['bbox'] = annotations['bbox2Dcwh']
                    del annotations['bbox2Dcwh']
                    del annotations['bbox3D']
                    
                    support_label_batch.append(annotations)
            
            if self.bounding_box_mode == "2D_to_2D":
                if i in query_idx_list:
                    query_image_batch.append(image)
                    
                    annotations['bbox'] = annotations['bbox2Dcwh']
                    del annotations['bbox2Dcwh']
                    del annotations['bbox3D']
                    
                    query_label_batch.append(annotations)
                else:
                    support_image_batch.append(image)
                    
                    annotations['bbox'] = annotations['bbox2Dcwh']
                    del annotations['bbox2Dcwh']
                    del annotations['bbox3D']
                    
                    support_label_batch.append(annotations)
            if self.bounding_box_mode == "3D_to_3D":
                if i in query_idx_list:
                    query_image_batch.append(image)
                    
                    annotations['bbox'] = annotations['bbox3D']
                    del annotations['bbox2Dcwh']
                    del annotations['bbox3D']
                    
                    query_label_batch.append(annotations)
                else:
                    support_image_batch.append(image)
                    
                    annotations['bbox'] = annotations['bbox3D']
                    del annotations['bbox2Dcwh']
                    del annotations['bbox3D']
                    
                    support_label_batch.append(annotations)


        return np.stack(support_image_batch), support_label_batch, np.stack(query_image_batch), query_label_batch, self.id2label
    


    def __iter__(self):
        while self.iteration_count < 100:
            self.iteration_count += 1
            yield self._sample()



