import os
import datetime
import cv2
from helper import (create_video_writer, save_screenshot, filter_image,
                    draw_datetime, draw_recording_timer)


class VideoProcessing:
    def __init__(self, video_source, output_dir):
        self.video_cap = cv2.VideoCapture(video_source)
        if not self.video_cap.isOpened():
            raise RuntimeError(f"Error: Cannot open video source {video_source}")

        self.output_dir = output_dir
        self.screenshot_dir = os.path.join(self.output_dir, "screenshots")
        self.movies_dir = os.path.join(self.output_dir, "movies")
        os.makedirs(self.screenshot_dir, exist_ok=True)
        os.makedirs(self.movies_dir, exist_ok=True)

        self.is_recording = False
        self.record_start_time = None
        self.writer = None

    def toggle_recording(self):
        if self.is_recording:
            self.is_recording = False
            if self.writer:
                self.writer.release()
                self.writer = None
        else:
            self.is_recording = True
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            output_file = os.path.join(self.movies_dir, f"output_{timestamp}.mp4")
            self.writer = create_video_writer(self.video_cap, output_file)

    def save_screenshot(self, frame):
        save_screenshot(frame, self.screenshot_dir)

    def process_frame(self, frame):
        frame = filter_image(frame)
        draw_datetime(frame)
        if self.is_recording:
            draw_recording_timer(frame, self.record_start_time)
        return frame
