import sys
from PyQt5.QtWidgets import QApplication
from object_detection import ObjectDetectionAndTracking
from object_detection_gui import ObjectDetectionGUI  # Імпортуємо GUI із окремого файлу


def main():
    # Параметри для ObjectDetectionAndTracking
    model_path = "yolov8n.pt"  # Замініть на шлях до вашої моделі
    video_source = 1  # Використовується камера за замовчуванням
    confidence_threshold = 0.7  # Порог впевненості для детекції

    try:
        # Ініціалізація об'єкта детекції
        detection_app = ObjectDetectionAndTracking(
            model_path=model_path,
            video_source=video_source,
            confidence_threshold=confidence_threshold,
        )

        # Ініціалізація PyQt застосунку
        qt_app = QApplication(sys.argv)
        gui = ObjectDetectionGUI(detection_app)

        # Запуск GUI
        gui.show()
        sys.exit(qt_app.exec_())

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
