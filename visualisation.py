import matplotlib.pyplot as plt
from matplotlib import patches
import cv2 
import numpy as np
from torchvision import transforms
import random
from kitti_utils import Draw_Boxes
import pdb


# define some colors in BGR, add more if needed
object_color_dict = {
    'Car': [0, 0, 255], # red
    'Van': [255, 0, 0], # blue
    'Truck': [0,255, 0], # green
    'Pedestrian': [240, 32, 160], # purple
    'Person_sitting': [0, 255, 255], # yellow
    'Cyclist': [255, 0, 255], # magenta
    'Tram': [0, 128, 128], # olive
    'Misc': [[0, 215, 255]] # gold
}




def visualize_image_with_boxes(image, boxes, pred_boxes=None, image_size = None, image_type=None, file_name = 'NA'):
    # Convert PyTorch tensor to NumPy array
    
    if image_size is not None:
        # pdb.set_trace()
        transform = transforms.Resize(image_size.cpu().numpy().tolist()[0])
        image_np = transform(image).permute(1, 2, 0).cpu().numpy()
    else:

        image_np = image.permute(1, 2, 0).cpu().numpy()

    # Convert to uint8 for OpenCV
    image_np = (image_np * 255).astype(np.uint8)
    
    image_cv2 = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    scale = np.array([image_cv2.shape[1], image_cv2.shape[0], image_cv2.shape[1], image_cv2.shape[0]])
    

    if image_type == "2D":
        if boxes is not None: 
            for box in boxes:
                # x1, y1, x2, y2 = (box.cpu().numpy() * scale).astype(int)
                # image_cv2 = cv2.rectangle(image_cv2, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                cx, cy, w, h = (box.cpu().numpy() * scale).astype(int)
                image_cv2 = cv2.rectangle(image_cv2
                                          , (int(cx - w/2), int(cy - h/2))
                                          , (int(cx + w/2), int(cy + h/2))
                                          , (0, 255, 0), 2)

        if pred_boxes is not None:
        
            for box in pred_boxes:
                x1, y1, x2, y2 = (box.cpu().numpy() * scale).astype(int)
                image_cv2 = cv2.rectangle(image_cv2, (x1, y1), (x2, y2), (0, 0, 255), 2)


            # Display the image using OpenCV
            num = random.randint(0, 10000)
            cv2.imwrite(f"./logs/images_2D/{file_name}_{num}.png", image_cv2)
            # cv2.imshow('Image with Boxes', image_cv2)
        #     cv2.waitKey(0)
        # cv2.destroyAllWindows()

    
    if image_type == "3D":

        boxes = boxes.detach().cpu().numpy().flatten()
        scalexyz = np.array([  image_cv2.shape[1]
                             , image_cv2.shape[0]
                             , image_cv2.shape[0]*image_cv2.shape[1]])
        
        scalehwl = np.array([  image_cv2.shape[0]
                             , image_cv2.shape[1]
                             , image_cv2.shape[0]*image_cv2.shape[1]])
        
        
        draw_boxes = Draw_Boxes()
        # corners_3d_img = draw_boxes.transform_3dbox_to_image(  boxes[[3,2,5]]*scalehwl
                                                            #  , boxes[[0,1,4]]*scalexyz
                                                            #  , boxes[6]*2*np.pi)
        corners_3d_img = draw_boxes.transform_3dbox_to_image(  boxes[3:6]
                                                             , boxes[0:3]
                                                             , boxes[6])
      



        if corners_3d_img is None:
            raise AssertionError("Something is wrong with the transform_3dbox_to_image function as it returned a None")
        else:
            corners_3d_img = corners_3d_img.astype(int)

            img = image_cv2
            thickness = 2
            bbox_color = object_color_dict['Truck']
            # pdb.set_trace()

            # Ground Truth
            # draw lines in the image
            # p10-p1, p1-p2, p2-p3, p3-p0
            # try:
            cv2.line(img, (corners_3d_img[0, 0], corners_3d_img[0, 1]),
                (corners_3d_img[1, 0], corners_3d_img[1, 1]), color=bbox_color, thickness=thickness)
            # pdb.set_trace()
            cv2.line(img, (corners_3d_img[1, 0], corners_3d_img[1, 1]),
                (corners_3d_img[2, 0], corners_3d_img[2, 1]), color=bbox_color, thickness=thickness)
            cv2.line(img, (corners_3d_img[2, 0], corners_3d_img[2, 1]),
                (corners_3d_img[3, 0], corners_3d_img[3, 1]), color=bbox_color, thickness=thickness)
            cv2.line(img, (corners_3d_img[3, 0], corners_3d_img[3, 1]),
                (corners_3d_img[0, 0], corners_3d_img[0, 1]), color=bbox_color, thickness=thickness)

            # p4-p5, p5-p6, p6-p7, p7-p0
            # pdb.set_trace()
            cv2.line(img, (corners_3d_img[4, 0], corners_3d_img[4, 1]),
                (corners_3d_img[5, 0], corners_3d_img[5, 1]), color=bbox_color, thickness=thickness)
            # pdb.set_trace()
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

            # cv2.putText(img, text='Car', org=(corners_3d_img[4, 0]+5, corners_3d_img[4, 1]),
            #             fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4, color=bbox_color, thickness=thickness)
                
            # except:
            #     print("in execept for gt")



       # FOR PREDICTION OF BOUNDING BOXES
        # draw_boxes = Draw_Boxes()
        pred_boxes = pred_boxes.detach().cpu().numpy().flatten()
        # pdb.set_trace()
        # corners_3d_img = draw_boxes.transform_3dbox_to_image(  pred_boxes[[3,2,5]]*scalehwl
                                                            #  , pred_boxes[[0,1,4]]*scalexyz
                                                            #  , pred_boxes[6]*2*np.pi)



        corners_3d_img = draw_boxes.transform_3dbox_to_image(  pred_boxes[3:6]
                                                             , pred_boxes[0:3]
                                                             , pred_boxes[6])
      


        if corners_3d_img is None:
            raise AssertionError("Something is wrong with the transform_3dbox_to_image function as it returned a None")
        else:
            corners_3d_img = corners_3d_img.astype(int)

            img = image_cv2
            thickness = 2
            bbox_color = object_color_dict['Car']

            
            try:
                # Ground Truth
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

                # cv2.putText(img, text='Car', org=(corners_3d_img[4, 0]+5, corners_3d_img[4, 1]),
                #             fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4, color=bbox_color, thickness=thickness)
                
            except:
                print("In execept for pred")

            # Display the image using OpenCV
            num = random.randint(0, 10000)
            cv2.imwrite(f"./logs/images_3D/{file_name}_{num}.png", image_cv2)
            # cv2.imshow('Image with Boxes', image_cv2)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
        

    # cv2.imshow('3D Bounding Box', img)

# if save_img:
#     print('Save path: ', save_path)
#     cv2.imwrite(save_path, img)

# while True:
#     key = cv2.waitKey(1) & 0xFF
#     if key == ord('q'):
#         break

# cv2.destroyAllWindows()



















# def visualize_image_with_boxes(image, boxes, pred_boxes= None):
#     # Convert image to numpy array
#     image_np = image.permute(1, 2, 0).cpu().numpy()

#     # Create figure and axes
#     fig, ax = plt.subplots(1)

#     # Display the image
#     ax.imshow(image_np)

#     # Add bounding boxes to the image
#     for box in boxes:
#         # Extract normalized coordinates
#         x1, y1, x2, y2 = box.cpu().numpy()

#         # Convert normalized coordinates to pixel coordinates
#         h, w = image.shape[1], image.shape[2]
#         x1_pixel, y1_pixel = x1 * w, y1 * h
#         x2_pixel, y2_pixel = x2 * w, y2 * h

#         # Calculate width and height
#         w_pixel = x2_pixel - x1_pixel
#         h_pixel = y2_pixel - y1_pixel

#         # Create a Rectangle patch
#         rect = patches.Rectangle((x1_pixel, y1_pixel), w_pixel, h_pixel, linewidth=1, edgecolor='r', facecolor='none')

#         # Add the patch to the Axes
#         ax.add_patch(rect)

#     plt.show()