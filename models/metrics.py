import torch
from torchvision.ops import nms
from sklearn.metrics import precision_recall_curve, auc

def apply_nms(boxes, scores, iou_threshold=0.5, score_threshold=0.5):
    
   
    mask = scores > score_threshold
    boxes = boxes[mask]
    scores = scores[mask]

    # Apply NMS
    keep = nms(boxes, scores, iou_threshold)

    return keep

def compute_ap_after_nms(predicted_boxes, predicted_scores, predicted_labels, true_boxes, true_labels, iou_threshold=0.5, score_threshold=0.5):
    # Apply NMS
    keep_indices = nms(predicted_boxes, predicted_scores, iou_threshold)

    # Keep only the selected bounding boxes after NMS
    selected_boxes = predicted_boxes[keep_indices]
    selected_scores = predicted_scores[keep_indices]
    selected_labels = predicted_labels[keep_indices]

    # Calculate precision and recall
    tp, fp, fn = calculate_tp_fp_fn_with_labels(selected_boxes, selected_labels, true_boxes, true_labels, iou_threshold)

    # Compute precision and recall
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # Compute AP using precision-recall curve
    precision, recall, _ = precision_recall_curve([1] * len(true_boxes), selected_scores)
    ap = auc(recall, precision)

    return ap

def calculate_tp_fp_fn_with_labels(predicted_boxes, predicted_labels, true_boxes, true_labels, iou_threshold):
    # Initialize counts
    tp, fp, fn = 0, 0, 0

    # Mark all ground truth boxes as not matched
    matched_gt_boxes = torch.zeros(len(true_boxes), dtype=torch.bool)

    # Iterate through predicted boxes
    for pred_box, pred_label in zip(predicted_boxes, predicted_labels):
        ious = calculate_iou(pred_box, true_boxes)

        # Find the index of the ground truth box with the highest IoU
        max_iou, max_iou_idx = ious.max(dim=0)

        # Check if the IoU is above the threshold, the ground truth box is not already matched, and the labels match
        if max_iou >= iou_threshold and not matched_gt_boxes[max_iou_idx] and pred_label == true_labels[max_iou_idx]:
            tp += 1
            matched_gt_boxes[max_iou_idx] = True
        else:
            fp += 1

    # Calculate false negatives
    fn = len(true_boxes) - tp

    return tp, fp, fn


def calculate_iou(box1, box2):
    # Calculate intersection coordinates
    x1 = torch.maximum(box1[0], box2[:, 0])
    y1 = torch.maximum(box1[1], box2[:, 1])
    x2 = torch.minimum(box1[2], box2[:, 2])
    y2 = torch.minimum(box1[3], box2[:, 3])

    # Calculate intersection area
    intersection_area = torch.maximum(0, x2 - x1) * torch.maximum(0, y2 - y1)

    # Calculate union area
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    union_area = box1_area + box2_area - intersection_area

    # Calculate IoU
    iou = intersection_area / union_area

    return iou

# # Replace these tensors with your actual predictions and ground truth annotations
# predicted_boxes = None  # (N, 4)
# predicted_scores = None  # (N,)
# predicted_labels = None  # (N,)

# true_boxes = None  # (M, 4)
# true_labels = None # (M,)
# # Compute AP after NMS
# ap = compute_ap_after_nms(predicted_boxes, 
#                           predicted_scores, 
#                           predicted_labels,
#                           true_boxes,
#                           true_labels)

# # Print the computed AP
# print(f"Average Precision (AP) after NMS: {ap:.4f}")


