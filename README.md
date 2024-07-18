# Tea-leaf-diseases-detection-based-on-YOLOv9
tea leaf diseases detection based on YOLOv9 (computer vision project)
import os
HOME = os.getcwd()
print(HOME)
//installation
!git clone https://github.com/SkalskiP/yolov9.git
%cd yolov9
!pip install -r requirements.txt -q
!wget -P {HOME}/weights -q https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-e.pt
!wget -P {HOME}/weights -q https://github.com/WongKinYiu/yolov9/releases/download/v0.1/gelan-c.pt
!wget -P {HOME}/weights -q https://github.com/WongKinYiu/yolov9/releases/download/v0.1/gelan-e.pt
!ls -la {HOME}/weights
!wget -P {HOME}/data -q https://media.roboflow.com/notebooks/examples/dog.jpeg
%cd {HOME}/yolov9
!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="AGfuNwvA5KninjvE1LRP")
project = rf.workspace("AUS").project("tea-76um3")
version = project.version(2)
dataset = version.download("yolov9")
%cd {HOME}/yolov9

!python train.py \
--batch 16 --epochs 25 --img 640 --device 0 --min-items 0 --close-mosaic 15 \
--data {dataset.location}/data.yaml \
--weights {HOME}/weights/gelan-c.pt \
--cfg models/detect/gelan-c.yaml \
--hyp hyp.scratch-high.yaml
import glob

from IPython.display import Image, display

for image_path in glob.glob(f'{HOME}/yolov9/runs/detect/exp3/*.jpg')[:5]:
      display(Image(filename=image_path, width=600))
