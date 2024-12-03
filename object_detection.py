import os
import datetime
import cv2
import torch
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from helper import create_video_writer, save_screenshot, calculate_fps, draw_text
from utils import deep_sort_params, classes


class ObjectDetectionAndTracking:
    def __init__(self, model_path, video_source=0, confidence_threshold=0.8, allowed_classes=classes):
        self.allowed_classes = allowed_classes
        self.output_file = None
        self.confidence_threshold = confidence_threshold
        self.GREEN = (0, 255, 0)  # Колір для "person"
        self.BLUE = (255, 0, 0)  # Колір для інших об'єктів
        self.WHITE = (255, 255, 255)
        self.RED = (0, 0, 255)

        # Ініціалізація пристрою для обчислень
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Ініціалізація відеозахоплення
        self.video_cap = cv2.VideoCapture(video_source)
        if not self.video_cap.isOpened():
            raise RuntimeError(f"Error: Cannot open video source {video_source}")

        # Створення директорій для збереження
        self.output_dir = os.path.expanduser("./data")
        self.screenshot_dir = os.path.join(self.output_dir, "screenshots")
        self.movies_dir = os.path.join(self.output_dir, "movies")
        os.makedirs(self.screenshot_dir, exist_ok=True)
        os.makedirs(self.movies_dir, exist_ok=True)

        # Завантаження моделі YOLO
        self.model = YOLO(model_path)
        self.model.to(self.device)

        # Ініціалізація трекера DeepSort
        self.tracker = DeepSort(**deep_sort_params)

        # Статус запису
        self.is_recording = False
        self.writer = None

        # Час для сповіщень про скріншот
        self.screenshot_notification_time = None

        # Ініціалізація FPS
        self.prev_time = datetime.datetime.now()
        self.fps = 0.0

    def process_frame(self, frame):
        """Обробка одного кадру для детекції та трекінгу."""
        original_size = frame.shape[1::-1]  # Оригінальні (ширина, висота)
        frame_resized = cv2.resize(frame, (640, 640))

        # Підготовка кадру для моделі
        frame_tensor = torch.from_numpy(frame_resized).permute(2, 0, 1).unsqueeze(0).float().to(self.device)

        with torch.no_grad():
            detections = self.model(frame_tensor)[0]

        if detections is None or len(detections.boxes) == 0:
            return frame  # Повернення без змін, якщо немає детекцій

        results = self._extract_results(detections)

        # Оновлення трекера
        tracks = self.tracker.update_tracks(results, frame=frame_resized)

        # Малювання треків
        self._draw_tracks(frame_resized, tracks)

        # Масштабування до оригінальних розмірів
        frame_output = cv2.resize(frame_resized, original_size)

        # Додавання FPS, статусу та сповіщень
        self.fps = calculate_fps(self.prev_time)
        self.prev_time = datetime.datetime.now()
        draw_text(frame_output, f"FPS: {self.fps:.2f}", (10, 30), (0, 255, 0))
        self._draw_status(frame_output)

        return frame_output

    def _extract_results(self, detections):
        """Обробка результатів детекції YOLO."""
        results = []
        for det in detections.boxes.data.cpu().numpy():
            confidence = det[4]
            if confidence < self.confidence_threshold:
                continue
            class_id = int(det[5])
            class_name = self.model.names.get(class_id, "Unknown")
            if class_name not in self.allowed_classes:  # Фільтр класів
                continue
            x_min, y_min, x_max, y_max = map(int, det[:4])
            width, height = x_max - x_min, y_max - y_min
            results.append([[x_min, y_min, width, height], confidence, class_id])
        return results

    def _draw_tracks(self, frame, tracks):
        """Малювання треків."""
        for track in tracks:
            if not track.is_confirmed():
                continue
            class_id = track.det_class
            class_name = self.model.names.get(class_id, "Unknown")
            if class_name not in self.allowed_classes:  # Фільтр класів
                continue
            x_min, y_min, x_max, y_max = map(int, track.to_ltrb())

            color = self.GREEN if class_name.lower() == "person" else self.BLUE
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
            text = f"{class_name} ID: {track.track_id}"
            draw_text(frame, text, (x_min, y_min - 10), self.WHITE)

    def _draw_status(self, frame):
        """Малювання статусу запису."""
        status_text = f"Recording: {'ON' if self.is_recording else 'OFF'}"
        status_color = self.GREEN if self.is_recording else self.RED
        draw_text(frame, status_text, (10, 60), status_color)

    def toggle_recording(self):
        """Перемикання запису."""
        if self.is_recording:
            self.is_recording = False
            if self.writer:
                self.writer.release()
                self.writer = None
        else:
            self.is_recording = True
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.output_file = os.path.join(self.movies_dir, f"output_{timestamp}.mp4")
            self.writer = create_video_writer(self.video_cap, self.output_file)

    def save_screenshot(self, frame):
        """Збереження скріншоту."""
        save_screenshot(frame, self.screenshot_dir)

    def run(self):
        """Запуск програми."""
        while True:
            ret, frame = self.video_cap.read()
            if not ret:
                print("Error: Unable to read frame.")
                break

            processed_frame = self.process_frame(frame)
            cv2.imshow("Object Detection and Tracking", processed_frame)

            if self.is_recording and self.writer:
                self.writer.write(processed_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.video_cap.release()
        if self.writer:
            self.writer.release()
        cv2.destroyAllWindows()