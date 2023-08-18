import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torchvision


def inference(img: np.ndarray, model: torchvision.models.detection.FasterRCNN, device: torch.device) -> None:
    """
    img: un-normalized numpy array with size (H, W, C)
    model: FasterRCNN model
    device: torch.device

    """
    img = img / 255.0
    img = torch.tensor(img, dtype=torch.float).permute(2, 0, 1).unsqueeze(0)
    img = img.to(device)
    print(f"Image device: {img.device}")

    if next(model.parameters()).device:
        print(f"Model device: CUDA")
    else:
        print(f"Model device: CPU")
        
    model.eval()
    res = model(img)

    draw_results(img, res)


def draw_results(img: torch.tensor, res: list) -> None:
    """
    img: torch.tensor
    res: list of dict with keys: ['boxes', 'labels', 'scores']
    
    """

    img = img.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    boxes = res[0]['boxes'].cpu().detach().numpy()
    labels = res[0]['labels'].cpu().detach().numpy()
    for i in range(len(boxes)):
        box = boxes[i]
        label = labels[i]

        x_min = int(box[0])
        y_min = int(box[1])
        x_max = int(box[2])
        y_max = int(box[3])

        # Draw bounding box
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)
        # Draw label
        cv2.putText(img, str(label), (x_min, y_min), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    plt.figure(figsize=(10, 6), dpi=150)
    plt.imshow(img)