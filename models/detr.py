import torch 
import torchvision
from torchvision.models import resnet50, ResNet50_Weights
from transformers import DetrForObjectDetection, DetrImageProcessor, DPTImageProcessor, DPTForDepthEstimation
from torch import nn
import pdb
from torchvision import transforms
# class DetrMLPPredictionHead(nn.Module):
#     """
#     Very simple multi-layer perceptron (MLP, also called FFN), used to predict the normalized center coordinates,
#     height and width of a bounding box w.r.t. an image.

#     Copied from https://github.com/facebookresearch/detr/blob/master/models/detr.py

#     """

#     def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
#         super().__init__()
#         self.num_layers = num_layers
#         h = [hidden_dim] * (num_layers - 1)
#         self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

#     def forward(self, x):
#         for i, layer in enumerate(self.layers):
#             x = nn.functional.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
#         return x



# class DETR(torch.nn.Module):
#     def __init__(self, height = 800, width = 1333, num_classes = 5, mlp_layers = 3, bbox_size = 4) -> None:
#         super().__init__()


#         self.processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
#         self.processor.size = (height,width)

#         self.detr = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

#         self.class_labels_classifier = torch.nn.Linear(
#             self.detr.config.d_model, num_classes + 1
#         )  # We add one for the "no object" class


#         # Change this layer for 3D bounding box for kitti dataset
#         self.bbox_predictor = DetrMLPPredictionHead(input_dim = self.detr.config.d_model # 256 default hidden size
#                                                     , hidden_dim = self.detr.config.d_model
#                                                     , output_dim = bbox_size  # bounding box
#                                                     , num_layers = mlp_layers  # MLP layers
#                                                     )

#     def forward(self, image_batch: torch.Tensor, labels = None ):

        
#         # output = self.detr.model(image_batch)
#         # sequence_output = output[0]
#         # logits = self.class_labels_classifier(sequence_output)
#         # bboxes = self.bbox_predictor(sequence_output).sigmoid()
#         results = self.detr(image_batch,labels =labels)

#         return results
    
#     # def train(self): 
#     #     self.detr.train()

#     def eval(self): 
#         self.model.eval()






class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)  # Softmax activation for multi-class classification

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.relu(x)
        x = self.layer3(x)
        # x = self.softmax(x)
        return x

class DETRCustom(nn.Module):
    """
    Demo DETR implementation.

    Demo implementation of DETR in minimal number of lines, with the
    following differences wrt DETR in the paper:
    * learned positional encoding (instead of sine)
    * positional encoding is passed at input (instead of attention)
    * fc bbox predictor (instead of MLP)
    The model achieves ~40 AP on COCO val5k and runs at ~28 FPS on Tesla V100.
    Only batch size 1 supported.

    """
    def __init__(self, num_classes, bounding_box_mode, hidden_dim=256, nheads=8,
                 num_encoder_layers=6, num_decoder_layers=6, n_queries = 100, device = None, depth_supervision = False):
        super().__init__()

        # create ResNet-50 backbone

        # self.backbone = resnet50(weights = ResNet50_Weights.DEFAULT ).train()
        self.backbone = resnet50()
        del self.backbone.fc
        self.depth_supervision = depth_supervision
        self.bounding_box_mode = bounding_box_mode
        if device: 
            self.device = device
        else:
            self.device = 'cpu'


        if depth_supervision == True:

            self.depth_preprocessor = DPTImageProcessor.from_pretrained("Intel/dpt-large")
            self.depth_estimator = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")
            self.depth_estimator.eval()
            self.depth_postprocess = transforms.Compose([transforms.Resize((16,16))])
            self.conv1 = nn.Conv2d(2048 + 4096, hidden_dim, 1)


        self.conv = nn.Conv2d(2048, hidden_dim,1)

        # create a default PyTorch transformer
        self.transformer = nn.Transformer(
            hidden_dim, nheads, num_encoder_layers, num_decoder_layers, batch_first=True)

        # prediction heads, one extra class for predicting non-empty slots
        # note that in baseline DETR linear_bbox layer is 3-layer MLP
        
        # learn this new 
        self.linear_class= nn.Linear(hidden_dim, 92)

        # Pretrained weights for this 
        self.linear_bbox = nn.Linear(hidden_dim, 4)
        # Pretrained (cx,cy,w,h)
        
        if self.depth_supervision == True:
        
            self.common = MLP(2*hidden_dim, hidden_dim//2, hidden_dim//4)
        else:
            self.common = MLP(hidden_dim, hidden_dim//2, hidden_dim//4)
        
        
        self.linear_class_custom = nn.Linear(hidden_dim//4, num_classes + 1)
        # self.linear_bbox_custom = nn.Linear(hidden_dim//4, 7)

        self.linear_bbox_2D = nn.Linear(hidden_dim//4, 4)
        self.linear_bbox_3D = nn.Linear(hidden_dim//4, 7)
        
        # [x, y, w, h, z, l, alpha]

        # output positional encodings (object queries)
        self.query_pos = nn.Parameter(torch.rand(n_queries, hidden_dim))
        # self.query_pos1 = nn.Parameter(torch.rand(20, hidden_dim))

        # spatial positional encodings
        # note that in baseline DETR we use sine positional encodings
        self.row_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
        self.col_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))

    def forward(self, inputs, box_type = '2D'):
        # propagate inputs through ResNet-50 up to avg-pool layer




        B,C,W,H = inputs.shape

        with torch.no_grad():
            x = self.backbone.conv1(inputs)
            x = self.backbone.bn1(x)
            x = self.backbone.relu(x)
            x = self.backbone.maxpool(x)

            x = self.backbone.layer1(x)
            x = self.backbone.layer2(x)
            x = self.backbone.layer3(x)
            x = self.backbone.layer4(x)


       
            # pdb.set_trace()               
        # else:
        
        hidden = self.conv(x)



        # construct positional encodings
        H, W = hidden.shape[-2:]
        
        # pos = torch.cat([
        #     self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
        #     self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
        # ], dim=-1).flatten(0, 1).unsqueeze(1)
        # hidden = self.transformer(pos + 0.1 * hidden.flatten(2).permute(2, 0, 1),                   
        # self.query_pos.unsqueeze(1)).transpose(0, 1)

        pos = torch.cat([
                        self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
                        self.row_embed[:H].unsqueeze(1).repeat(1, W, 1)
                        ]
                        ,dim=-1).flatten(0, 1)

        
       
        
        # hout = self.transformer(pos + 0.1 * hidden.flatten(2).permute(0,2,1),
        #                      self.query_pos.unsqueeze(0).repeat(B,1,1))
        
        hout = self.transformer(pos + 0.1 * hidden.flatten(2).permute(0,2,1),
                             self.query_pos.unsqueeze(0).repeat(B,1,1))
        # pdb.set_trace()





        if self.depth_supervision == True:
        
            temp  = self.depth_preprocessor(inputs, do_rescale = False, return_tensors = 'pt')
            with torch.no_grad():
                # self.to('cpu')
                self.depth_estimator.to(self.device)
                temp = temp.pixel_values.float().to(self.device)
                depth_map = self.depth_estimator(temp).predicted_depth
                depth_map = self.depth_postprocess(depth_map).view(B,-1)
                
                # Shape of 256. Do not change this
            # pdb.set_trace()
        
            hnew = torch.concat([hout,depth_map.unsqueeze(1).repeat(1,100,1).to('cuda')], dim = -1)
            hf = self.common(hnew)
        
        else:    
            hf = self.common(hout)

        # finally project transformer outputs to class labels and bounding boxes
        if box_type == '2D':

            return {'pred_logits': self.linear_class_custom(hf), 
                    'pred_boxes': self.linear_bbox_2D(hf).sigmoid()}

        else: 

            return {'pred_logits': self.linear_class_custom(hf), 
                    'pred_boxes': self.linear_bbox_3D(hf)}
