"""Implementation of model-agnostic meta-learning for FSOD and Kitti."""
import sys
sys.path.append('..')
import os

import numpy as np
import torch
import pdb
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import copy
from torch import nn
import torch.nn.functional as F
from torch import autograd
from torch.utils import tensorboard
from torch.utils.tensorboard import SummaryWriter

from transformers import DetrImageProcessor, DetrForObjectDetection
from models.detr import DETRCustom
from torch.optim import AdamW
from utils import object_detection_loss, hungarian_matching, mean_iou, bbox_loss
import pandas as pd
from visualisation import visualize_image_with_boxes
SUMMARY_INTERVAL = 10
SAVE_INTERVAL = 10
LOG_INTERVAL = 10
VAL_INTERVAL = LOG_INTERVAL * 5
# NUM_TEST_TASKS = 5

def fetch_models(model_name):

    if model_name == 'DETR':
        processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
        model = DETRCustom


    return model, processor

class MAML:
    """Trains and assesses a MAML."""

    def __init__(
            self,
            model_name = 'DETR',
            num_classes=3,
            num_inner_steps = 100,
            inner_lr = 0.001,
            learn_inner_lrs = False,
            outer_steps = 200,
            outer_lr = 0.001,
            log_dir  = "./logs/kitti",
            device = 'cpu',
            writer = None,
            bounding_box_mode = None,
            depth_supervision = False
    ):
        """Inits MAML.

        Args:
            num_outputs (int): number of classes in a task
            num_inner_steps (int): number of inner-loop optimization steps
            inner_lr (float): learning rate for inner-loop optimization
                If learn_inner_lrs=True, inner_lr serves as the initialization
                of the learning rates.
            learn_inner_lrs (bool): whether to learn the above
            outer_lr (float): learning rate for outer-loop optimization
            log_dir (str): path to logging directory
            device (str): device to be used
        """
        
        # Store the Model Parameters here 
        self.device = device
        self.mode = 'none'
        self._num_inner_steps = num_inner_steps   
        self.outer_steps = outer_steps     
        self.inner_lr = inner_lr
        self._outer_lr = outer_lr
        self._log_dir = log_dir
        self._start_train_step = 0
        self.writer = writer
        self.bounding_box_mode = bounding_box_mode
        self.outer_loop_epoch = 0
        self.learn_inner_lrs = learn_inner_lrs
        os.makedirs(self._log_dir, exist_ok=True)


        model, self.processor = fetch_models(model_name)
        self.model = model(
            num_classes=num_classes
            ,bounding_box_mode=self.bounding_box_mode
            ,depth_supervision = depth_supervision, device = device)
         
        self.inner_model = model(
                                 num_classes=num_classes
                                ,bounding_box_mode=self.bounding_box_mode
                                ,depth_supervision = depth_supervision)
        
        state_dict = torch.hub.load_state_dict_from_url(
                              url='https://dl.fbaipublicfiles.com/detr/detr_demo-da2a99e9.pth'
                            , map_location='cpu'
                            , check_hash=True
                            )

        self.model.load_state_dict(state_dict, strict = False)

        self._meta_parameters = copy.deepcopy(self.model.state_dict())
        
        self._inner_lrs = {
            k: torch.tensor(inner_lr, requires_grad=learn_inner_lrs)
            for k in self._meta_parameters.keys()
        }
        self._optimizer = torch.optim.Adam(
            list(self._meta_parameters.values()) +
            list(self._inner_lrs.values()),
            lr=self._outer_lr
        )
    

    def _inner_loop(self, images, labels, train, writer):
        """Computes the adapted network parameters via the MAML inner loop.

        Args:
            images (Tensor): task support set inputs
                shape (num_images, channels, height, width)
            labels (Dict): List of Dictionary with labels and bbox
            train (bool): whether we are training or evaluating

        Returns:
            parameters (dict[str, Tensor]): adapted network parameters
            metrics (list[float]): support set metrics over the course of
                the inner loop, length num_inner_steps + 1
            gradients(list[float]): gradients computed from auto.grad, just needed
                for autograders, no need to use this value in your code and feel to replace
                with underscore       
        """

        parameters = copy.deepcopy(self._meta_parameters)
        
        self.inner_model.load_state_dict(parameters, strict = False)
        self.inner_model.to(self.device)
        self.inner_model.train()
        metrics = []

        optim = torch.optim.AdamW(self.inner_model.parameters(),lr = self.inner_lr)
        
        for step in range(self._num_inner_steps + 1 ):
            
            optim.zero_grad()
            results = self.inner_model(images, box_type = self.bounding_box_mode.split('_')[0])
            
            loss, mIOU,mAP,accuracy, match_box_list = object_detection_loss( 
                                          pred_boxes=results['pred_boxes']
                                         ,pred_logits=results['pred_logits']
                                         ,true_boxes=[x['bbox'].to(self.device) for x in labels]
                                         ,true_labels=[x['labels'].to(self.device)  for x in labels]
                                         ,box_type=self.bounding_box_mode.split('_')[0] # Box type for loss and metric calculations
                                        )
            outer_idx = self._start_train_step
            
            if step == 0:
                if train == True :
                    
                    self.writer.add_scalar("Training Loss/Support Set/Pre", loss, outer_idx )
                    self.writer.add_scalar("Training mIOU/Support Set/Pre", mIOU, outer_idx)
                    self.writer.add_scalar("Training MAP/Support Set/Pre", mAP,  outer_idx)
                    self.writer.add_scalar("Training Accuracy/Support Set/Pre", accuracy, outer_idx)
                else: 
                    self.writer.add_scalar("Testing Loss/Support Set/Pre", loss, outer_idx )
                    self.writer.add_scalar("Testing mIOU/Support Set/Pre", mIOU, outer_idx)
                    self.writer.add_scalar("Testing MAP/Support Set/Pre", mAP,  outer_idx)
                    self.writer.add_scalar("Testing Accuracy/Support Set/Pre", accuracy,  outer_idx)

            print(f"Support Set Loss: {loss.item():.3f} MAP: {mAP:.3f} Mean IOU: {mIOU:.3f}, Acc: {accuracy:.3f}")

            metrics.append(accuracy)
            
            
            if (step < self._num_inner_steps):           
                
                if self.learn_inner_lrs:
                    
                    gradients =  autograd.grad(loss, self.inner_model.parameters() 
                                               , create_graph=True
                                               , retain_graph=True
                                               , allow_unused=True)
                    
                    for param, grad, key in zip(self.inner_model.parameters(), gradients, self.inner_model.state_dict().keys()):

                        if grad is not None:
                                param.data -= self._inner_lrs[key] * grad

                else:
                    loss.backward(retain_graph = train)

                    optim.step()

        if True:
            for i in range(images.shape[0]):
                    visualize_image_with_boxes(  images[i]
                                               , labels[i]['bbox']
                                               , match_box_list[i]
                                               , image_size = labels[i]['image_size']
                                               , image_type=self.bounding_box_mode.split('_')[0]
                                               , file_name= 'Support_'+ self.bounding_box_mode +"_"+ self.mode)

        if train == True:
            
            self.writer.add_scalar("Training Loss/Support Set/Post ", loss, outer_idx )
            self.writer.add_scalar("Training mIOU/Support Set/Post", mIOU, outer_idx)
            self.writer.add_scalar("Training MAP/Support Set/Post", mAP,  outer_idx)
            self.writer.add_scalar("Training Accuracy/Support Set/Post", accuracy,  outer_idx)
        else: 
            self.writer.add_scalar("Testing Loss/Support Set/Post", loss, outer_idx )
            self.writer.add_scalar("Testing mIOU/Support Set/Post", mIOU, outer_idx)
            self.writer.add_scalar("Testing MAP/Support Set/Post", mAP,  outer_idx)
            self.writer.add_scalar("Testing Accuracy/Support Set/Post", accuracy,  outer_idx)

        parameters = self.inner_model.state_dict()

        return parameters, metrics , None

    def _outer_step(self, task_batch, train, writer = None):
        """Computes the MAML loss and metrics on a batch of tasks.

        Args:
            task_batch (tuple): batch of tasks from DataLoader
            train (bool): whether we are training or evaluating

        Returns:
            outer_loss (Tensor): mean MAML loss over the batch, scalar
        """

        
        images_support, labels_support, images_query, labels_query, mapper = task_batch
    
        images_support = torch.tensor(images_support).to(self.device)
        images_query = torch.tensor(images_query).to(self.device)


        parameters, inner_metrics ,_ = self._inner_loop(images_support,labels_support,train, writer)
        
        self.model.load_state_dict(parameters) # Main Model 
        self.model.to(self.device)

        # optim = AdamW(self.model.parameters(),self._outer_lr)

        if train!=True:
            self.model.eval()

        results = self.model(images_query, box_type = self.bounding_box_mode.split('_')[-1])

        loss, mIOU, mAP, acc, match_box_list = object_detection_loss( 
                                          pred_boxes=results['pred_boxes']
                                         ,pred_logits=results['pred_logits']
                                         ,true_boxes=[x['bbox'].to(self.device) for x in labels_query]
                                         ,true_labels=[x['labels'].to(self.device)  for x in labels_query]
                                         ,box_type=self.bounding_box_mode.split('_')[-1] # Box type for loss and metric calculations
                                        )
       
        if train == True:
            self.writer.add_scalar("Training Loss/Query Set", loss, self._start_train_step)
            self.writer.add_scalar("Training mIOU/Query Set", mIOU, self._start_train_step)
            self.writer.add_scalar("Training MAP/Query Set", mAP,self._start_train_step)
            self.writer.add_scalar("Training Accuracy/Query Set",acc,self._start_train_step)
        else: 
            self.writer.add_scalar("Testing Loss/Query Set", loss, self._start_train_step)
            self.writer.add_scalar("Testing mIOU/Query Set", mIOU, self._start_train_step)
            self.writer.add_scalar("Testing MAP/Query Set", mAP,self._start_train_step)
            self.writer.add_scalar("Testing Accuracy/Query Set",acc,self._start_train_step)


        if True:
            for i in range(images_query.shape[0]):
                visualize_image_with_boxes(images_query[i]
                                           , labels_query[i]['bbox']
                                           , match_box_list[i]
                                           , image_type=self.bounding_box_mode.split('_')[-1]
                                           , image_size=labels_query[i]['image_size']
                                           , file_name= 'Query_' + self.bounding_box_mode + "_" + self.mode )

        print(f"Query Set \n Loss: {loss.item():.3f} MAP: {mAP:.3f} Mean IOU: {mIOU:.3f} Acc: {acc:.3f}")
        return loss, inner_metrics

    def train(self, dataloader_meta_train, dataloader_meta_val=None, writer = None):
        """Train the MAML.

        Consumes dataloader_meta_train to optimize MAML meta-parameters
        while periodically validating on dataloader_meta_val, logging metrics, and
        saving checkpoints.

        Args:
            dataloader_meta_train (DataLoader): loader for train tasks
            dataloader_meta_val (DataLoader): loader for validation tasks
            writer (SummaryWriter): TensorBoard logger
        """
        print(f'Starting training at iteration {self._start_train_step}.')
        self.mode = 'train'

        for i in range(self._start_train_step, self.outer_steps):
            
            task = dataloader_meta_train._sample()  
        
            self._optimizer.zero_grad()

            outer_loss, inner_metrics = self._outer_step(task, train=True)

            if self.mode == 'train':
                outer_loss.backward()

            for name,value in self.model.named_parameters():
                if value.grad is not None:
                    self._meta_parameters[name] = self._meta_parameters[name].to(self.device) - self._outer_lr*value.grad

            for key, value in self._inner_lrs.items():
                if value.grad is not None: 

                    self._inner_lrs[key]-= self._outer_lr*value.grad


            self._start_train_step+=1
            print('Outer Loop Step', self._start_train_step)         
        
            if (i)%10 == 0:
                self._save(i)
            

    def test(self, dataloader_test):
        """Evaluate the MAML on test tasks.

        Args:
            dataloader_test (DataLoader): loader for test tasks
        """
        self.mode = 'test'
        accuracies = []
        self._start_train_step = 0
        for i in range(20): 
            task_batch = dataloader_test._sample()

            loss = self._outer_step(task_batch, train=False)
            print(f'Task{i}: Loss- {loss}')
            self._start_train_step += 1


    def load(self, checkpoint_step):
        """Loads a checkpoint.

        Args:
            checkpoint_step (int): iteration of checkpoint to load

        Raises:
            ValueError: if checkpoint for checkpoint_step is not found
        """
        target_path = (
            f'{os.path.join(self._log_dir, "state")}'
            f'{checkpoint_step}.pt'
        )
        if os.path.isfile(target_path):
            state = torch.load(target_path)
            self._meta_parameters = state['meta_parameters']
            self._inner_lrs = state['inner_lrs']
            self._optimizer.load_state_dict(state['optimizer_state_dict'])
            self._start_train_step = checkpoint_step + 1
            print(f'Loaded checkpoint iteration {checkpoint_step}.')
        else:
            raise ValueError(
                f'No checkpoint for iteration {checkpoint_step} found.'
            )

    def _save(self, checkpoint_step):
        """Saves parameters and optimizer state_dict as a checkpoint.

        Args:
            checkpoint_step (int): iteration to label checkpoint with
        """
        optimizer_state_dict = self._optimizer.state_dict()
        torch.save(
            dict(meta_parameters=self._meta_parameters,
                 inner_lrs=self._inner_lrs,
                 optimizer_state_dict=optimizer_state_dict),
            f'{os.path.join(self._log_dir, "state")}{checkpoint_step}.pt'
        )
        print('Saved checkpoint.')

