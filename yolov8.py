from ultralytics import YOLO
from utils import load_rf_data, model_to_folder


def train_yolov8(model_name, dataset_location, epochs=100, batch_size=8, imgsz=640):
    res_folder_name = model_to_folder[model_name]
    model = YOLO(model_name)
    results = model.train(data=f'{dataset_location}/data.yaml', epochs=epochs, batch=batch_size, imgsz=imgsz, verbose=True, name=res_folder_name, exist_ok=True)
    return results

def validate_yolov8(model_name):
    model_type = model_to_folder[model_name]
    model_l = YOLO(f"runs/detect/{model_type}/weights/best.pt")
    metrics = model_l.val(name=f"val_{model_type}", exist_ok=True)
    return metrics



if __name__ == "__main__":

    # PARAMETERS
    epochs = 100
    batch_size = 8
    imgsz = 640
    MODELS = ['yolov8l.pt', 'yolov8m.pt', 'yolov8s.pt', 'yolov8n.pt']

    dataset_location = load_rf_data(version=9)

    for MODEL in MODELS:
        res = train_yolov8(model_name=MODEL, dataset_location=dataset_location, epochs=epochs, batch_size=batch_size, imgsz=imgsz)
        metrics = validate_yolov8(model_name=MODEL)
    
