import yaml
import cv2
import numpy as np
import os
import torch

class Dataset(torch.utils.data.Dataset):

    def __init__(self, dataset_location, dataset_type, imgsz=640, transforms=None):

        self.imgsz = imgsz
        self.transforms = transforms
        self.dataset_type = dataset_type
        with open(f"{dataset_location}/data.yaml") as f:
            doc = yaml.safe_load(f)
            self.img_dir = doc[dataset_type]+"/images"           
            self.label_dir = doc[dataset_type]+"/labels"
            self.classes = doc['names']
            self.classes.insert(0, "background")

        self.imgs = [image for image in sorted(os.listdir(self.img_dir)) if image.endswith(".jpg")]

    def __getitem__(self, index):
        img_name = self.imgs[index]
        image_path = os.path.join(self.img_dir, img_name)
        
        # Converting images to correct size and color
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)
        aspect_ratio = img.shape[0] / img.shape[1]
        height = int(aspect_ratio * self.imgsz)
        img = cv2.resize(img, (self.imgsz, height), interpolation=cv2.INTER_AREA)

        img /= 255.0

        annot_filename = img_name[:-4] + ".txt"
        annot_path = os.path.join(self.label_dir, annot_filename)

        boxes = []
        labels = []
        with open(annot_path, "r") as f:
            for line in f:
                class_id, x, y, w, h = line.strip().split()
                class_id = int(class_id)
                x_min = float(x) - float(w) / 2
                y_min = float(y) - float(h) / 2
                x_max = float(x) + float(w) / 2
                y_max = float(y) + float(h) / 2

                x_min = int(x_min * self.imgsz)
                y_min = int(y_min * height)
                x_max = int(x_max * self.imgsz)
                y_max = int(y_max * height)

                boxes.append([x_min, y_min, x_max, y_max])
                labels.append(class_id)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([index])
        target["area"] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        target["iscrowd"] = torch.zeros((boxes.shape[0],), dtype=torch.int64)

        if self.transforms:
            sample = self.transforms(image=img, bboxes=target['boxes'], labels=labels)
            img = sample['image']
            target['boxes'] = torch.tensor(sample['bboxes'], dtype=torch.float32)   

        return img, target
    
    
    def __len__(self):
        return len(self.imgs)