import os
import datetime
import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from helper import create_video_writer


class ObjectDetectionAndTracking:
    def __init__(self, model_path, video_source=0, confidence_threshold=0.8):
        self.output_file = None
        self.confidence_threshold = confidence_threshold
        self.GREEN = (0, 255, 0)
        self.WHITE = (255, 255, 255)

        # Initialize video capture
        self.video_cap = cv2.VideoCapture(video_source)

        # Create output directories
        self.output_dir = os.path.expanduser("/Users/erihkoh/GitHub/video_detecting_and_tracking/data")
        self.screenshot_dir = os.path.join(self.output_dir, "screenshots")
        self.movies_dir = os.path.join(self.output_dir, "movies")
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.screenshot_dir, exist_ok=True)

        # Load YOLO model
        self.model = YOLO(model_path)

        # Initialize DeepSort tracker
        self.tracker = DeepSort(max_age=50)

        # Recording state
        self.is_recording = False
        self.writer = None

        # Screenshot notification state
        self.screenshot_notification_time = None

    def process_frame(self, frame):
        """Process a single frame for detection and tracking."""
        start = datetime.datetime.now()

        # Run YOLO model on the frame
        detections = self.model(frame)[0]

        # Extract results
        results = self._extract_results(detections)

        # Update the tracker
        tracks = self.tracker.update_tracks(results, frame=frame)

        # Draw detections and tracks
        self._draw_tracks(frame, tracks)

        # Compute FPS and annotate the frame
        self._draw_fps(frame, start)

        # Add recording status, controls text, and screenshot notification
        self._draw_status_and_controls(frame)
        self._draw_screenshot_notification(frame)

        return frame

    def _extract_results(self, detections):
        """Extract bounding boxes and confidences from detections."""
        results = []
        for data in detections.boxes.data.tolist():
            confidence = data[4]
            if float(confidence) < self.confidence_threshold:
                continue
            xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
            class_id = int(data[5])
            results.append([[xmin, ymin, xmax - xmin, ymax - ymin], confidence, class_id])
        return results

    def _draw_tracks(self, frame, tracks):
        """Draw bounding boxes and track IDs on the frame."""
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            ltrb = track.to_ltrb()
            xmin, ymin, xmax, ymax = map(int, ltrb)

            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), self.GREEN, 2)
            cv2.rectangle(frame, (xmin, ymin - 20), (xmin + 20, ymin), self.GREEN, -1)
            cv2.putText(frame, str(track_id), (xmin + 5, ymin - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.WHITE, 2)

    def _draw_fps(self, frame, start):
        """Compute and draw FPS on the frame."""
        end = datetime.datetime.now()
        fps = f"FPS: {1 / (end - start).total_seconds():.2f}"
        cv2.putText(frame, fps, (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 8)

    def _draw_status_and_controls(self, frame):
        """Draw recording status and functional buttons text."""
        controls_text = "Press 'R' to toggle recording | Press 'S' for screenshot | Press 'Q' to quit"
        recording_text = "Recording: ON" if self.is_recording else "Recording: OFF"
        recording_color = (0, 255, 0) if self.is_recording else (0, 0, 255)

        # Draw recording status
        cv2.putText(frame, recording_text, (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, recording_color, 2)

        # Draw controls text
        cv2.putText(frame, controls_text, (50, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.WHITE, 2)

    def _draw_screenshot_notification(self, frame):
        """Draw a notification on the frame if a screenshot was recently taken."""
        if self.screenshot_notification_time:
            elapsed = (datetime.datetime.now() - self.screenshot_notification_time).total_seconds()
            if elapsed < 2:  # Show notification for 2 seconds
                notification_text = "Screenshot saved!"
                cv2.putText(frame, notification_text, (50, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                self.screenshot_notification_time = None  # Clear notification after 2 seconds

    def toggle_recording(self):
        """Toggle the recording state."""
        if self.is_recording:
            print("Stopping recording...")
            self.is_recording = False
            if self.writer is not None:
                self.writer.release()
        else:
            print("Starting recording...")
            self.is_recording = True
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.output_file = os.path.join(self.movies_dir, f"output_{timestamp}.mp4")
            self.writer = create_video_writer(self.video_cap, self.output_file)

    def save_screenshot(self, frame):
        """Save a screenshot to the screenshots' directory."""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        screenshot_path = os.path.join(self.screenshot_dir, f"screenshot_{timestamp}.png")
        cv2.imwrite(screenshot_path, frame)
        self.screenshot_notification_time = datetime.datetime.now()  # Set notification time
        print(f"Screenshot saved to {screenshot_path}")

    def run(self):
        """Run the object detection and tracking pipeline."""
        while True:
            ret, frame = self.video_cap.read()
            if not ret:
                break

            frame = self.process_frame(frame)

            # Show the frame
            cv2.imshow("Frame", frame)

            # Write to the output file if recording is active
            if self.is_recording and self.writer is not None:
                self.writer.write(frame)

            # Check for keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                self.toggle_recording()
            elif key == ord("s"):
                self.save_screenshot(frame)

        # Release resources
        self.video_cap.release()
        if self.writer is not None:
            self.writer.release()
        cv2.destroyAllWindows()


# if __name__ == "__main__":
#     app = ObjectDetectionAndTracking(
#         model_path="yolov5su.pt",
#         video_source=0,
#         confidence_threshold=0.8
#     )
#     app.run()
