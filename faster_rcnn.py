import cv2

import torch
print(torch.__version__)
import torchvision
print(torchvision.__version__)
from torchvision import transforms as torchtrans
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from frcnn_tools.engine import train_one_epoch, evaluate
import utils
from utils import *

from rcnn_dataset import Dataset

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


def create_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


# main
if __name__ == "__main__":

    # DEVICE
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # PARAMETERS
    BATCH_SIZE = 4
    NUM_EPOCHS = 10
    IMGSZ = 640
    LEARNING_RATE = 0.0005
    WEIGHT_DECAY = 0.0005

    # DATALOADERS
    dataset_location = utils.load_rf_data(8)

    transforms = A.Compose([ToTensorV2(p=1.0)], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})

    train_dataset = Dataset(dataset_location, dataset_type="train", imgsz=IMGSZ, transforms=transforms)
    val_dataset = Dataset(dataset_location, dataset_type="val", imgsz=IMGSZ, transforms=transforms)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, collate_fn=utils.collate_fn
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=utils.collate_fn
    )

    # MODEL
    model = create_model(num_classes=5)
    model.to(device)

    params  = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # TRAINING
    for epoch in range(NUM_EPOCHS):
        train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=50)
        lr_scheduler.step()
        evaluate(model, val_loader, device=device)
    
    # FINAL EVALUATION
    evaluate(model, val_loader, device=device)

    # SAVE MODEL
    torch.save(model.state_dict(), "faster_rcnn.pt")