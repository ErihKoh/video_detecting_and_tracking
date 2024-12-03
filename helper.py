import datetime
import cv2
import os


def create_video_writer(video_cap, output_filename):
    frame_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_cap.get(cv2.CAP_PROP_FPS) or 30)  # Використання стандартного FPS, якщо значення невідоме

    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264
    writer = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))
    if not writer.isOpened():
        raise IOError(f"Failed to open VideoWriter for {output_filename}")
    return writer


def save_screenshot(frame, screenshot_dir):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    screenshot_path = os.path.join(screenshot_dir, f"screenshot_{timestamp}.png")
    cv2.imwrite(screenshot_path, frame)
    print(f"Screenshot saved to {screenshot_path}")


def calculate_fps(prev_time):
    elapsed_time = (datetime.datetime.now() - prev_time).total_seconds()
    return 1 / elapsed_time if elapsed_time > 0 else 0


def draw_text(frame, text, position, color):
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)