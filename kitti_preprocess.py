import os
import pandas as pd
import itertools
import random
import shutil
import sys
import torch
from torch import nn
import numpy as np
import pdb

labels_folder_path= r"/home/tejas/Documents/Stanford/CS_330/Final_Project/Data/Kitti/object_labels/training/label_2/"
images_folder_path = r"/home/tejas/Documents/Stanford/CS_330/Final_Project/Data/Kitti/object_images/training/image_2/"

# All classes in Kitti Dataset
categories = ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc']


label_files = os.listdir(labels_folder_path)
data = []

for file in label_files:
    if file.endswith('.txt'):
        with open(os.path.join(labels_folder_path, file), 'r') as f:
            file_name = file.split('.')
            lines = f.read().splitlines()
            for line in lines:
                parts = line.split(' ')
             
                # Parts[0] is the label 
                # Parts[1:] are coordinates and other numerical values
                data.append([str(file_name[0]) + ".png", parts[0], parts[1:]])


df = pd.DataFrame(data, columns=['filename', 'class', 'coordinates'])
grouped = df.groupby('class')

# Make a class specific list to extract images of the same class
class_map = {k: v[['filename', 'coordinates']].values.tolist() for k, v in grouped}
all_classes = list(class_map.keys())
# Store this class map. This should be it, do we need to create new folders for each class? 


# Extract any set of 10 examples from the above dictionary of classes
# Do this in the task loader. Given label, extract k images without replacement.
car_sample = random.sample(list(class_map['Car']), 100)
cyclist_sample = random.sample(list(class_map['Cyclist']), 100)
truck_sample = random.sample(list(class_map['Truck']), 100)
pedestrian_sample = random.sample(list(class_map['Pedestrian']), 100)
person_sitting_sample = random.sample(list(class_map['Person_sitting']), 100)
tram_sample = random.sample(list(class_map['Tram']), 100)
van_sample = random.sample(list(class_map['Van']), 100)


# creating paths
new_folder_path_car = './datasets/kitti/all_data/Car/'
new_folder_path_cyclist = './datasets/kitti/all_data/Cyclist/'
new_folder_path_truck = './datasets/kitti/all_data/Truck/'
new_folder_path_pedestrian = './datasets/kitti/all_data/Pedestrian/'
new_folder_path_person_sitting = './datasets/kitti/all_data/Person_sitting/'
new_folder_path_tram = './datasets/kitti/all_data/Tram/'
new_folder_path_van = './datasets/kitti/all_data/Van/'
# data_path = r'./datasets/kitti/training/image_2/'

data_path = images_folder_path

if not os.path.exists(new_folder_path_car):
    os.makedirs(new_folder_path_car)
    car_labels_path = os.path.join(new_folder_path_car, 'labels')
    if not os.path.exists(car_labels_path):
        os.makedirs(car_labels_path)
        

# cyclist
if not os.path.exists(new_folder_path_cyclist):
    os.makedirs(new_folder_path_cyclist)
    cyclist_labels_path = os.path.join(new_folder_path_cyclist, 'labels')
    if not os.path.exists(cyclist_labels_path):
        os.makedirs(cyclist_labels_path)

    
# truck
if not os.path.exists(new_folder_path_truck):
    os.makedirs(new_folder_path_truck)
    truck_labels_path = os.path.join(new_folder_path_truck, 'labels')
    if not os.path.exists(truck_labels_path):
        os.makedirs(truck_labels_path)


# van
if not os.path.exists(new_folder_path_van):
    os.makedirs(new_folder_path_van)
    van_labels_path = os.path.join(new_folder_path_van, 'labels')
    if not os.path.exists(van_labels_path):
        os.makedirs(van_labels_path)


# pedestrian
if not os.path.exists(new_folder_path_pedestrian):
    os.makedirs(new_folder_path_pedestrian)
    pedestrian_labels_path = os.path.join(new_folder_path_pedestrian, 'labels')
    if not os.path.exists(pedestrian_labels_path):
        os.makedirs(pedestrian_labels_path)


# person_sitting
if not os.path.exists(new_folder_path_person_sitting):
    os.makedirs(new_folder_path_person_sitting)
    person_sitting_labels_path = os.path.join(new_folder_path_person_sitting, 'labels')
    if not os.path.exists(person_sitting_labels_path):
        os.makedirs(person_sitting_labels_path)


# Tram
if not os.path.exists(new_folder_path_tram):
    os.makedirs(new_folder_path_tram)
    tram_labels_path = os.path.join(new_folder_path_tram, 'labels')
    if not os.path.exists(tram_labels_path):
        os.makedirs(tram_labels_path)



def store_given_class(object_type, file_path, saving_path):
    
    temp_data = []

    with open(file_path, 'r') as f:
        file_name = file_path.split('.')[0].split('/')[-1]

        lines = f.read().splitlines()

        for l in lines:
            parts = l.split(' ')

            if parts[0] == object_type:
                line = ' '.join(str(x) for x in parts)
                temp_data.append(line)

        saving_path = os.path.join(saving_path, file_name + ".txt")
        
        with open(f'{saving_path}', 'w') as file:
            for row in temp_data:
                file.write(row + '\n')


# copy car files
for car in car_sample:
    if os.path.exists(data_path + car[0]):
        image_path = data_path + car[0]
        shutil.copy(image_path, new_folder_path_car)

        # to get the .txt file
        name = car[0].split('.')[0]
        file_path = os.path.join(labels_folder_path, name + ".txt")
        saving_path = os.path.join(new_folder_path_car, 'labels')
        store_given_class('Car', file_path, saving_path)



# copy truck files
for truck in truck_sample:
    # print("here: ", data_path + truck[0])
    if os.path.exists(data_path + truck[0]):
        image_path = data_path + truck[0]
        shutil.copy(image_path, new_folder_path_truck)

        # to get the .txt file
        name = truck[0].split('.')[0]
        file_path = os.path.join(labels_folder_path, name + ".txt")
        saving_path = os.path.join(new_folder_path_truck, 'labels')
        store_given_class('Truck', file_path, saving_path)



# copy cyclist sample files
for cyclist in cyclist_sample:
    if os.path.exists(data_path + cyclist[0]):
        image_path = data_path + cyclist[0]
        shutil.copy(image_path, new_folder_path_cyclist)

        # to get the .txt file
        name = cyclist[0].split('.')[0]
        file_path = os.path.join(labels_folder_path, name + ".txt")
        saving_path = os.path.join(new_folder_path_cyclist, 'labels')
        store_given_class('Cyclist', file_path, saving_path)



# copy van sample files
for van in van_sample:
    if os.path.exists(data_path + van[0]):
        image_path = data_path + van[0]
        shutil.copy(image_path, new_folder_path_van)

        # to get the .txt file
        name = van[0].split('.')[0]
        file_path = os.path.join(labels_folder_path, name + ".txt")
        saving_path = os.path.join(new_folder_path_van, 'labels')
        store_given_class('Van', file_path, saving_path)



# copy pedestrian sample files 
for pedestrian in pedestrian_sample:
    if os.path.exists(data_path + pedestrian[0]):
        image_path = data_path + pedestrian[0]
        shutil.copy(image_path, new_folder_path_pedestrian)

        # to get the .txt file
        name = pedestrian[0].split('.')[0]
        file_path = os.path.join(labels_folder_path, name + ".txt")
        saving_path = os.path.join(new_folder_path_pedestrian, 'labels')
        store_given_class('Pedestrian', file_path, saving_path)



# copy person sitting files
for person_sitting in person_sitting_sample:
    if os.path.exists(data_path + person_sitting[0]):
        image_path = data_path + person_sitting[0]
        shutil.copy(image_path, new_folder_path_person_sitting)

        # to get the .txt file
        name = person_sitting[0].split('.')[0]
        file_path = os.path.join(labels_folder_path, name + ".txt")
        saving_path = os.path.join(new_folder_path_person_sitting, 'labels')
        store_given_class('Person_sitting', file_path, saving_path)


# copy tram files
for tram in tram_sample:
    if os.path.exists(data_path + tram[0]):
        image_path = data_path + tram[0]
        shutil.copy(image_path, new_folder_path_tram)

        # to get the .txt file
        name = tram[0].split('.')[0]
        file_path = os.path.join(labels_folder_path, name + ".txt")
        saving_path = os.path.join(new_folder_path_tram, 'labels')
        store_given_class('Tram', file_path, saving_path)
