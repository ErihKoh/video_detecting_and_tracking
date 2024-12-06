import os


class Logger:
    def __init__(self, output_dir):
        self.logs_dir = os.path.join(output_dir, "logs")
        os.makedirs(self.logs_dir, exist_ok=True)
        self.log_file = os.path.join(self.logs_dir, "activity_log.txt")
        self._initialize_log()

    def _initialize_log(self):
        """Ініціалізація лог-файлу."""
        with open(self.log_file, "w") as f:
            f.write("Object Detection and Tracking Log\n")

    def log_detections(self, detected_objects):
        """Записує інформацію про виявлені об'єкти."""
        if not detected_objects:
            return
        with open(self.log_file, "a") as f:
            for obj in detected_objects:
                f.write(f"{obj}\n")