import torch 
from transformers import DPTForDepthEstimation, DPTImageProcessor

class DepthEstimator(torch.nn.Module):
    '''
    
    This module allows us to extract the depth map of the input monocular image. 
    Note that the default output depth map would be (384*384) after processing.
    To change the size or keep original size please pass the size as (h,w) while initialising this class object.
    '''


    def __init__(self, height = 384, width = 384) -> None:
        super().__init__()

        size = {"height": height, "width": width}

        self.processor = DPTImageProcessor(size = size ).from_pretrained("Intel/dpt-large")
        self.model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")

    def forward(self, image_batch):

        inputs = self.processor(images=image_batch, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model(**inputs)
            predicted_depth = outputs.predicted_depth

        predictions = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size= image_batch.size[::-1],
            mode="bicubic",
            align_corners=False,
        )

        return predictions
    