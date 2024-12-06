import os

class Logging:
    def __init__(self, logs_dir):
        self.log_file = os.path.join(logs_dir, "activity_log.txt")
        os.makedirs(logs_dir, exist_ok=True)
        with open(self.log_file, "w") as f:
            f.write("Object Detection Log\n")

    def log_detections(self, detected_objects):
        with open(self.log_file, "a") as f:
            for obj in detected_objects:
                f.write(f"{obj}\n")