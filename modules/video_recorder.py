import cv2
import os
import datetime

from utils.helper import create_video_writer, save_screenshot, draw_recording_timer


class VideoRecorder:
    def __init__(self, video_source, output_dir):

        self.video_cap = cv2.VideoCapture(video_source)
        if not self.video_cap.isOpened():
            raise RuntimeError(f"Error: Cannot open video source {video_source}")

        self.is_recording = False
        self.writer = None
        self.record_start_time = None
        self.record_end_time = None
        self.output_dir = output_dir
        self.output_file = None
        self.movies_dir = os.path.join(output_dir, "movies")
        self.screenshot_dir = os.path.join(output_dir, "screenshots")
        os.makedirs(self.movies_dir, exist_ok=True)
        os.makedirs(self.screenshot_dir, exist_ok=True)

    def toggle_recording(self):
        """Перемикання запису."""
        if self.is_recording:
            self.is_recording = False
            self.record_end_time = datetime.datetime.now()
            if self.writer:
                self.writer.release()
                self.writer = None
        else:
            self.is_recording = True
            self.record_start_time = datetime.datetime.now()
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.output_file = os.path.join(self.movies_dir, f"output_{timestamp}.mp4")
            self.writer = create_video_writer(self.video_cap, self.output_file)

    def save_screenshot(self, frame):
        """Збереження скріншоту."""
        save_screenshot(frame, self.screenshot_dir)

    def read_frame(self):
        ret, frame = self.video_cap.read()

        # Додавання інформації про запис із секундоміром
        if self.is_recording:
            draw_recording_timer(frame, self.record_start_time)

        if not ret:
            raise RuntimeError("Failed to read frame from video source")
        return frame

    def release(self):
        self.video_cap.release()
        if self.writer:
            self.writer.release()