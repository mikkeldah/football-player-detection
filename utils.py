import cv2
import matplotlib.pyplot as plt
import ultralytics
import numpy as np


class_to_name = {
    0: "ball",
    1: "keeper",
    2: "player",
    3: "referee",
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

