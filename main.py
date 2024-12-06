import os
import sys
from PyQt5.QtWidgets import QApplication

from modules.logger import Logger
from modules.object_tracking import ObjectTracking
from modules.video_recorder import VideoRecorder
from modules.object_detection import ObjectDetectionAndTracking
from modules.object_detection_gui import ObjectDetectionGUI  # Імпортуємо GUI із окремого файлу
from utils.config import detection_params, video_source, classes


def main():
    try:
        output_dir = os.path.expanduser("data")
        logger = Logger(output_dir)
        video_recorder = VideoRecorder(video_source, output_dir)
        tracker = ObjectTracking()
        # Ініціалізація об'єкта детекції
        detection_app = ObjectDetectionAndTracking(
            *detection_params, classes, logger, tracker, video_recorder, output_dir
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
