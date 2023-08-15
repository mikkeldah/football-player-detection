from ultralytics import YOLO
from utils import load_rf_data, model_to_folder


def train_yolov8(model_name, dataset_location, epochs=100, batch_size=8, imgsz=640):
    res_folder_name = model_to_folder[model_name]
    model = YOLO(model_name)
    results = model.train(data=f'{dataset_location}/data.yaml', epochs=epochs, batch=batch_size, imgsz=imgsz, verbose=True, name=res_folder_name, exist_ok=True)
    return results

def validate(model_name):
    model_type = model_to_folder[model_name]
    model_l = YOLO(f"runs/detect/{model_type}/weights/best.pt")
    metrics = model_l.val(name="val_large", exist_ok=True)
    return metrics



if __name__ == "__main__":

    MODEL = 'yolov8n.pt'

    dataset_location = load_rf_data(version=7)

    res = train_yolov8(model_name=MODEL, dataset_location=dataset_location, epochs=100, batch_size=8, imgsz=640)
    print(res)
    metrics = validate(model_name=MODEL)
    print(metrics)