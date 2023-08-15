import yaml
import cv2
import matplotlib.pyplot as plt
import numpy as np

import ultralytics
from roboflow import Roboflow

import config



class_to_name = {
    0: "ball",
    1: "keeper",
    2: "player",
    3: "referee",
}

model_to_folder = {
    "yolov8n.pt": "nano",
    "yolov8s.pt": "small",
    "yolov8m.pt": "medium",
    "yolov8l.pt": "large",
    "yolov8x.pt": "xlarge"
}

def visualize(im_path, label_path):
    name = im_path
    im = cv2.imread(name)
    im_height, im_width, _ = im.shape

    labels = label_path

    with open(labels, "r") as f:
        lines = f.readlines()
        for line in lines:
            box = line.strip().split(" ")
            cls, x, y, w, h = int(box[0]), float(box[1]), float(box[2]), float(box[3]), float(box[4])
            x1, y1, x2, y2 = ultralytics.utils.ops.xywh2xyxy(np.array([x, y, w, h]))
            x1, y1, x2, y2 = int(x1*im_width), int(y1*im_height), int(x2*im_width),int(y2*im_height)

            cv2.rectangle(im, (x1, y1), (x2, y2), (204, 0, 0), 1)
            cv2.putText(im, text=class_to_name[cls], org=(x1, y1-5), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6, color=(204, 0, 0), thickness=1)

    img = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(12, 8), dpi=150)
    plt.imshow(img)
    plt.show()


def load_rf_data(version):
    rf = Roboflow(api_key=config.api_key)
    project = rf.workspace("mikkel-ds").project("football-player-detection-xvszo")
    dataset = project.version(version).download("yolov8")

    with open(f"{dataset.location}/data.yaml") as f:
        doc = yaml.safe_load(f)

        doc['train'] = f"{dataset.location}/train"
        doc['val'] = f"{dataset.location}/valid"

    with open(f'{dataset.location}/data.yaml', 'w') as f:
        yaml.safe_dump(doc, f)
    
    return dataset.location

