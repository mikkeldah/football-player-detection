import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torchvision


def inference(img: np.ndarray, model: torchvision.models.detection.FasterRCNN, device: torch.device, conf: float = 0.25) -> None:
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

    # Filter out low confidence predictions
    res[0]['boxes'] = res[0]['boxes'][res[0]['scores'] > conf]
    res[0]['labels'] = res[0]['labels'][res[0]['scores'] > conf]
    res[0]['scores'] = res[0]['scores'][res[0]['scores'] > conf]

    draw_results(img, res)


label_to_color = {
    0: (0, 0, 0),
    1: (0, 255, 0),
    2: (0, 0, 255),
    3: (255, 0, 0),
    4: (255, 255, 0),
}

def draw_results(img: torch.tensor, res: list) -> None:
    """
    img: torch.tensor
    res: list of dict with keys: ['boxes', 'labels', 'scores']
    
    """

    img = img.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    boxes = res[0]['boxes'].cpu().detach().numpy()
    labels = res[0]['labels'].cpu().detach().numpy()
    scores = res[0]['scores'].cpu().detach().numpy()
    for i in range(len(boxes)):
        box = boxes[i]
        label = labels[i]
        score = scores[i]

        color = label_to_color[label]

        x_min = int(box[0])
        y_min = int(box[1])
        x_max = int(box[2])
        y_max = int(box[3])

        # Draw bounding box
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 1)
        # Draw label
        cv2.putText(img, f"{str(label)}: {str(round(score, 2))}", (x_min, y_min-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    plt.figure(figsize=(10, 6), dpi=150)
    plt.imshow(img)


def show_annotated_image(img, target):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    for box in target['boxes']:
        x1, y1, x2, y2 = int(box[0].item()), int(box[1].item()), int(box[2].item()), int(box[3].item())
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)

    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()