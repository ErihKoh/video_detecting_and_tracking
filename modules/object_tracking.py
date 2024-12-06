import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort
from utils.helper import draw_text
from utils.config import deep_sort_params, classes


class ObjectTracking:
    def __init__(self):
        self.tracker = DeepSort(**deep_sort_params)
        self.GREEN = (0, 255, 0)  # Колір для "person"
        self.BLUE = (255, 0, 0)  # Колір для інших об'єктів
        self.WHITE = (255, 255, 255)
        self.classes = classes

    def update_tracks(self, results, frame):
        """Оновлення треків на основі результатів детекції."""
        tracks = self.tracker.update_tracks(results, frame=frame)
        self._draw_tracks(frame, tracks)  # Виклик малювання треків
        return tracks

    def _draw_tracks(self, frame, tracks):
        """Малювання треків на кадрі."""
        for track in tracks:
            if not track.is_confirmed():
                continue
            class_id = track.det_class
            x_min, y_min, x_max, y_max = map(int, track.to_ltrb())
            color = self.GREEN if class_id == 0 else self.BLUE
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
            text = f"Class: {self.classes[class_id]} ID: {track.track_id}"
            draw_text(frame, text, (x_min, y_min - 10), self.WHITE)