import torch
from ultralytics import YOLO


class ObjectDetection:
    def __init__(self, model_path, device="cpu", confidence_threshold=0.8):
        self.device = torch.device(device)
        self.model = YOLO(model_path)
        self.model.to(self.device)
        self.confidence_threshold = confidence_threshold

    def detect_objects(self, frame):
        frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
        with torch.no_grad():
            detections = self.model(frame_tensor)[0]
        return detections

    def filter_detections(self, detections, allowed_classes):
        results = []
        if detections is None or len(detections.boxes) == 0:
            return results

        for det in detections.boxes.data.cpu().numpy():
            confidence = det[4]
            if confidence < self.confidence_threshold:
                continue
            class_id = int(det[5])
            if class_id not in allowed_classes:
                continue
            x_min, y_min, x_max, y_max = map(int, det[:4])
            width, height = x_max - x_min, y_max - y_min
            results.append([[x_min, y_min, width, height], confidence, class_id])
        return results
