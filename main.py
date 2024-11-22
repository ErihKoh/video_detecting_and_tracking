from object_detection_gui import ObjectDetectionGUI
from object_detection import ObjectDetectionAndTracking
from threading import Thread


if __name__ == "__main__":
    # Инициализация объекта для детекции и трекинга
    app = ObjectDetectionAndTracking(
        model_path="yolov5su.pt",
        video_source=0,
        confidence_threshold=0.8
    )

    # Инициализация GUI
    gui = ObjectDetectionGUI(app)

    # Запуск GUI
    gui.run()