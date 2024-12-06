import sys
from PyQt5.QtWidgets import QApplication
from object_detection import ObjectDetection
from data.object_tracking import ObjectTracking
from data.video_processing import VideoProcessing
from logging_module import Logging
from object_detection_gui import ObjectDetectionGUI  # Імпортуйте ваш GUI клас


def main():
    # Налаштування
    model_path = "../yolov8n.pt"
    video_source = 0
    output_dir = "../data"
    allowed_classes = {0: "person"}  # Замініть на відповідні класи

    # Ініціалізація компонентів
    object_detector = ObjectDetection(model_path, confidence_threshold=0.8)
    object_tracker = ObjectTracking()
    video_processor = VideoProcessing(video_source, output_dir)
    logger = Logging(output_dir)

    # Створення GUI
    app = QApplication(sys.argv)
    gui = ObjectDetectionGUI(video_processor, logger, object_detector)

    # Запуск GUI
    gui.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()