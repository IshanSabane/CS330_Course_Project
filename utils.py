import torch
import torch.nn.functional as F
import pdb
import numpy as np

from scipy.optimize import linear_sum_assignment
# from iou3d.oriented_iou_loss import cal_iou_3d,cal_giou_3d
# from pytorch3d.ops import box3d_overlap
# from torchmetrics.detection import mean_ap
# from pytorch3d.ops import iou_box3d
# from pytorch3d.transforms import transform3d
from kitti_utils import Draw_Boxes




# Converts center width and height to top left and bottom right corner coordinates.
def cxywh_xyxy(boxes):

    # Extract coordinates
    center_x, center_y, width, height = torch.unbind(boxes, dim=1)

    # Calculate (x1, y1, x2, y2)
    x1 = center_x - width / 2
    y1 = center_y - height / 2
    
    x2 = center_x + width / 2
    y2 = center_y + height / 2

    # Stack the results to form (n, 4) tensor
    xyxy_boxes = torch.stack((x1, y1, x2, y2), dim=1)

    return torch.clamp(xyxy_boxes,min = 0, max = 1)



# Finds the largest bounding box which encloses the input 3D bounding box. Takes batched input.
def get_bounding_box(boxes):
    # boxes (B,8,2)
    x_values = boxes[:, :, 0]
    y_values = boxes[:, :, 1]

    # Find top-left and bottom-right coordinates
    top_left_x = torch.min(x_values, dim=1)[0]
    top_left_y = torch.min(y_values, dim=1)[0]
    bottom_right_x = torch.max(x_values, dim=1)[0]
    bottom_right_y = torch.max(y_values, dim=1)[0]

    # Create a new tensor with shape (B, 4)
    result_tensor = torch.stack([top_left_x, top_left_y, bottom_right_x, bottom_right_y], dim=1)

    return result_tensor




# Return the projected coordinates using camera parameters
def camera3d_xyxy(boxes):

    output = []
    for i in range(boxes.shape[0]):
        output.append(transform_3dbox_to_image(boxes[i][3:6],boxes[i][0:3], boxes[i][6]))
    return torch.stack(output)


# Transforms the 3D coordinates into 8 2D points.
def transform_3dbox_to_image( dimension, location, rotation):
        """
        Convert the 3D box to coordinates in point cloud.
        :param dimension: height, width, and length
        :param location: x, y, and z
        :param rotation: rotation parameter
        :return: transformed coordinates
        """
        db = Draw_Boxes()
        calib = db.get_sequence_calib()

        height, width, length = dimension
        x, y, z = location

        x_corners = torch.tensor([length / 2, length / 2, -length / 2, -length / 2, length / 2, length / 2, -length / 2, -length / 2])
        y_corners = torch.tensor([0, 0, 0, 0, -height, -height, -height, -height])
        z_corners = torch.tensor([width / 2, -width / 2, -width / 2, width / 2, width / 2, -width / 2, -width / 2, width / 2])

        corners_3d = torch.stack([x_corners, y_corners, z_corners])

        # Transform 3D box based on rotation along Y-axis
        R_matrix = torch.tensor([[torch.cos(rotation), 0, torch.sin(rotation)],
                                 [0, 1, 0],
                                 [-torch.sin(rotation), 0, torch.cos(rotation)]], dtype=torch.float)

        corners_3d = torch.mm(R_matrix, corners_3d.float())

        # pdb.set_trace()
        # Shift the corners from origin to location
        corners_3d = corners_3d + torch.tensor([x, y, z]).view(-1,1)

        # From camera coordinate to image coordinate
        corners_3d_temp = torch.cat((corners_3d, torch.ones((1, 8))), dim=0).float()
        corners_3d_img = torch.mm(corners_3d_temp.t().float(), torch.from_numpy(calib['P2']).t().float())
        corners_3d_img = corners_3d_img[:, :2] / corners_3d_img[:, 2].view(-1, 1)


        return corners_3d_img


# Computes an IoU matrix. 
def calculate_iou_matrix(boxes1, boxes2):

    intersection = torch.clamp(torch.min(boxes1[:, None, 2:], boxes2[:, 2:]) - torch.max(boxes1[:, None, :2], boxes2[:, :2]), min=0)
    area_intersection = intersection[:, :, 0] * intersection[:, :, 1]
    
    area_boxes1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area_boxes2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    union = area_boxes1[:, None] + area_boxes2 - area_intersection
    iou = area_intersection / union.clamp(min=1e-16)

    return iou

# The Hungarian Matching which returns indices.
def hungarian_matching(iou_matrix):
    row_ind, col_ind = linear_sum_assignment(-iou_matrix.detach().cpu())
    return row_ind, col_ind



# Computes the average IoU between the matched bounding boxes.
def mean_iou(matched_pred_boxes, matched_target_boxes):
  
    iou_matrix = calculate_iou_matrix(matched_pred_boxes, matched_target_boxes)
    mean_iou = torch.diagonal(iou_matrix).sum() / matched_pred_boxes.shape[0]
    return mean_iou.item()


# Computes the Regression loss between the matched bounding boxes
def bbox_loss(matched_pred_boxes, matched_target_boxes):
    
    loss = F.smooth_l1_loss(matched_pred_boxes, matched_target_boxes, reduction='mean')
    return loss


# Computes bounding boxes between two random boxes
def box_iou(box1, box2):
    # Calculate IoU between two boxes
    intersection = torch.min(box1[:, 2:], box2[:, 2:]).clamp(0).prod(dim=1)
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    union = area1 + area2 - intersection
    iou = intersection / union.clamp(min=1e-6)
    return iou




# Computes the Generalised IoU Loss between the two input boxes 
def box_giou(box1, box2):
    # Calculate GIoU between two boxes
    iou = box_iou(box1, box2)

    xmin = torch.min(box1[:, 0], box2[:, 0])
    ymin = torch.min(box1[:, 1], box2[:, 1])
    xmax = torch.max(box1[:, 2], box2[:, 2])
    ymax = torch.max(box1[:, 3], box2[:, 3])

    C_width = torch.clamp(xmax - xmin, min=0)
    C_height = torch.clamp(ymax - ymin, min=0)

    C_area = C_width * C_height

    giou = iou - (C_area - iou) / C_area
    return giou


def giou_loss(pred_boxes, target_boxes):
    # Calculate GIoU loss
    giou = box_giou(pred_boxes, target_boxes)
    giou_loss = 1 - giou.mean()
    return giou_loss



# Final object detection loss function
def object_detection_loss(pred_boxes, pred_logits, true_boxes, true_labels, box_type = '2D', iou_threshold=0.75):
    """
    Calculate object detection loss for bounding boxes and labels.

    Parameters:
        N in the following description is the number of queries to decoder. Check detr model. (N = 100 default)
        pred_boxes (tensor): Predicted bounding boxes, shape (B, N, 4).
        pred_logits (tensor): Predicted labels, shape (B, N, n_classes).
        true_boxes (tensor): Ground truth bounding boxes, shape list of [(M, 4)] with length B.
        true_labels (tensor): Ground truth labels, list of [(M,)] with length B.
        num_classes (int): Number of classes.
        iou_threshold (float): IoU threshold for matching predicted and true boxes.

    Returns:
        total_loss (tensor): Total loss.
    """

    total_loss = 0 
    classification_loss = 0
    regression_loss = 0
    gloss = 0
    # regression_loss2 = 0
    # giou = 0
    # lossh = 0
    accuracy = 0 
    miou=0
    map = 0
    matched_box_list = []
    for i in range(len(true_boxes)):

       
        # 2D bounding boxes
        if box_type == '2D':
            pred_xyxy = cxywh_xyxy(pred_boxes[i])
            true_xyxy = cxywh_xyxy(true_boxes[i])
            iou_matrix = calculate_iou_matrix(pred_xyxy,true_xyxy)
        # true_boxes[i] = true_boxes[i][:,0:4]

        if box_type =='3D':
            
            # pdb.set_trace()
            # true_coords = transform_3dbox_to_image(  true_boxes[i][:,3:6]
            #                                        , true_boxes[i][:,0:3]
            #                                        , true_boxes[i][:,6]
            #                                        )
            true_coords = camera3d_xyxy(true_boxes[i])
            pred_coords = camera3d_xyxy(pred_boxes[i])

            true_xyxy = get_bounding_box(true_coords)
            pred_xyxy = get_bounding_box(pred_coords)
            
            # true_xyxy = cxywh_xyxy(true_boxes[i][:, 0:4])
            # pred_xyxy = cxywh_xyxy(pred_boxes[i][:, 0:4])
            
            # true_xyxy = true_coords[:,[4,5,14,15]]
            # pred_xyxy = pred_coords[:,[4,5,14,15]]

            # pdb.set_trace()
            

            iou_matrix = calculate_iou_matrix(pred_xyxy,true_xyxy)
            # pdb.set_trace()

        #     iou_matrix = calculate_iou_matrix3d(pred_boxes[i][:, [0,1,4,3,2,5,6]], true_boxes[i][:, [0,1,4,3,2,5,6]])

        

        row_ind, col_ind = hungarian_matching(iou_matrix)
        
        if box_type == "3D":
            matched_box_list.append(pred_boxes[i])
        else:
            matched_box_list.append(pred_xyxy[row_ind].detach().clone())


        if box_type=='2D':
            regression_loss += bbox_loss(pred_xyxy[row_ind], true_xyxy[col_ind])
        else:
            # pdb.set_trace()
            regression_loss += bbox_loss(pred_boxes[i][row_ind], true_boxes[i][col_ind])


        matched_iou_matrix = calculate_iou_matrix(pred_xyxy[row_ind],true_xyxy[col_ind])

        gloss += 1- torch.mean(torch.diagonal(matched_iou_matrix))
        
        miou += mean_iou(pred_xyxy[row_ind],true_xyxy[col_ind])
       
        classification_loss += F.cross_entropy(pred_logits[i][row_ind], true_labels[i][col_ind])

        ylabels = torch.argmax(pred_logits[i][row_ind],dim = 1)
        
        accuracy += (ylabels == true_labels[i][col_ind]).sum()/len(row_ind)
        map += MeanAveragePrecision(iou_matrix)  
        # pdb.set_trace() 
    # print(torch.mean(matched_iou_matrix))
    if miou/len(true_boxes) > 0.5:
        total_loss = 0.3*classification_loss + 0.5*regression_loss + 0.2*gloss
    else:
        total_loss = 0.4*regression_loss + 0.4*gloss + 0.2*classification_loss
        # total_loss = regression_loss 
        
    return total_loss/len(true_boxes), miou/len(true_boxes) ,map/len(true_boxes), accuracy/len(true_boxes), matched_box_list




# Metric for Object Detection
def MeanAveragePrecision(iou_matrix): 
    mapk = 0 
    for threshold in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
        temp =  torch.max(iou_matrix, dim = 0)[0]

        mapk += 0.1*torch.sum(temp>threshold)/iou_matrix.shape[1]
        

    return mapk


