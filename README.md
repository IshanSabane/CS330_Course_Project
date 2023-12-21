# CS330 Deep Multi-Task and Meta Learning Course Project
# Enhancing Autonomous Navigation Few-Shot 3D Object Detection Using Depth Supervision

### Getting Started

#### Installation/Dependencies

Use the requirements.txt file to create a new python or conda enivronment. Use the following script to create a conda environment with all the required dependencies.

`
conda create -n meta3D --python=3.9
conda activate meta3D 
pip install -r requirements.txt
`

#### Datasets

To download the Stereo Kitti Dataset for evaluating the 3D meta learning model, please register on the official website and then download the left and right object detection images and uzip them in the datasets folder.


#### Model 

We currently support DETR model from hugging face as the backbone for the object detection task. The pretrained weights are avaialable on hugging face which are trained on COCO object detection dataset.

Use the run.sh file run the respective tasks.


<div>
	<img src = "./poster.png", alt =  "ProjectPoster"></img>
</div>


