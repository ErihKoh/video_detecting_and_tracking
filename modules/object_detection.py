import os
import datetime
import cv2
import torch
from ultralytics import YOLO

from utils.helper import calculate_fps, draw_text, filter_image, draw_datetime
from utils.config import classes


class ObjectDetectionAndTracking:
    def __init__(self, model_path, confidence_threshold=0.7, allowed_classes=classes, logger=None, tracker=None,
                 video_recorder=None, output_dir=None):
        self.allowed_classes = allowed_classes
        self.output_dir = output_dir
        self.logger = logger
        self.confidence_threshold = confidence_threshold
        self.RED = (0, 0, 255)
        self.GREEN = (0, 255, 0)

        # Ініціалізація пристрою для обчислень
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Ініціалізація відеозахоплення
        self.video_recorder = video_recorder
        self.video_cap = self.video_recorder.video_cap

        # Завантаження моделі YOLO
        self.model = YOLO(model_path)
        self.model.to(self.device)

        # Ініціалізація трекера DeepSort
        self.tracker = tracker

        # Час про скріншот
        self.screenshot_notification_time = None

        # Ініціалізація FPS
        self.prev_time = datetime.datetime.now()
        self.fps = 0.0

    def process_frame(self):
        """Обробка одного кадру для детекції та трекінгу."""
        # Фільтрація зображення
        frame = filter_image(self.video_recorder.read_frame())

        # Додавання часу та дати
        draw_datetime(frame)

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
        tracks = self.tracker.update_tracks(results, frame=frame_resized, )

        # Логування інформації
        self._log_detections(tracks)

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

    def _log_detections(self, tracks):
        """Логування виявлених об'єктів через Logger."""
        detected_objects = [f"Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, "
                            f"Class: {self.model.names[track.det_class]}, ID: {track.track_id}"
                            for track in tracks if track.is_confirmed()]
        self.logger.log_detections(detected_objects)

    def _draw_status(self, frame):
        """Малювання статусу запису."""
        status_text = f"Recording: {'ON' if self.video_recorder.is_recording else 'OFF'}"
        status_color = self.GREEN if self.video_recorder.is_recording else self.RED
        draw_text(frame, status_text, (10, 60), status_color)

    def toggle_recording(self):
        self.video_recorder.toggle_recording()

    #
    def save_screenshot(self, frame):
        self.video_recorder.save_screenshot(frame)

    def run(self):
        """Основний цикл програми для обробки відеопотоку."""
        try:
            while True:
                ret, frame = self.video_recorder.read_frame()
                if not ret:
                    print("Не вдалося отримати кадр. Завершення роботи.")
                    break

                # Обробка кадру
                processed_frame = self.process_frame(frame)

                # Виведення на екран
                cv2.imshow("Video Feed", processed_frame)

                # Обробка натиснень клавіш
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):  # Вихід за запитом
                    break

        except Exception as e:
            print(f"Сталася помилка: {e}")
        finally:
            self.video_recorder.release()
            cv2.destroyAllWindows()
            print("Роботу завершено.")
