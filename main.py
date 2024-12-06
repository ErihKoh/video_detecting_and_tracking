import sys
from PyQt5.QtWidgets import QApplication
from object_detection import ObjectDetectionAndTracking
from object_detection_gui import ObjectDetectionGUI  # Імпортуємо GUI із окремого файлу
from utils import detection_params


def main(params):
    try:
        # Ініціалізація об'єкта детекції
        detection_app = ObjectDetectionAndTracking(
            *params
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
    main(detection_params)
