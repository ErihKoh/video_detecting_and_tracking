from deep_sort_realtime.deepsort_tracker import DeepSort
from utils import deep_sort_params


class ObjectTracking:
    def __init__(self):
        self.tracker = DeepSort(**deep_sort_params)

    def update_tracks(self, detections, frame):
        return self.tracker.update_tracks(detections, frame=frame)
