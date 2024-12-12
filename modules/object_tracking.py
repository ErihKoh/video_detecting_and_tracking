import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort
from utils.helper import draw_text
from utils.config import deep_sort_params, classes, colors


class ObjectTracking:
    def __init__(self):
        self.tracker = DeepSort(**deep_sort_params)
        self.trajectories = {}
        self.GREEN = colors.get('GREEN')  # Колір для "person"
        self.BLUE = colors.get('BLUE')  # Колір для інших об'єктів
        self.WHITE = colors.get('WHITE')
        self.RED = colors.get('RED')
        self.classes = classes

    def update_tracks(self, results, frame):
        """Оновлення треків на основі результатів детекції."""
        tracks = self.tracker.update_tracks(results, frame=frame)
        self._update_trajectories(tracks)  # Оновлення траєкторій
        self._draw_tracks(frame, tracks)  # Виклик малювання треків
        return tracks

    def _update_trajectories(self, tracks):
        """Оновлення траєкторій для активних треків."""
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            center_x = int((track.to_tlbr()[0] + track.to_tlbr()[2]) / 2)
            center_y = int((track.to_tlbr()[1] + track.to_tlbr()[3]) / 2)
            if track_id not in self.trajectories:
                self.trajectories[track_id] = []
            self.trajectories[track_id].append((center_x, center_y))

    def _draw_tracks(self, frame, tracks):
        """Малювання треків на кадрі."""
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            class_id = track.det_class
            x_min, y_min, x_max, y_max = map(int, track.to_ltrb())
            color = self.GREEN if class_id == 0 else self.BLUE
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
            text = f"Class: {self.classes[class_id]} ID: {track_id}"
            draw_text(frame, text, (x_min, y_min - 10), self.WHITE)

            if class_id == 0:
                # Малювання траєкторій
                if track_id in self.trajectories:
                    points = self.trajectories[track_id]
                    for i in range(1, len(points)):
                        cv2.line(frame, points[i - 1], points[i], self.RED, 2)

