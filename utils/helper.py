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


def log_detections(path, detected_objects):
    """Логування інформації про детекцію об’єктів лише при їх наявності."""
    if not detected_objects:
        return

    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(path, "a") as f:
        f.write(f"--- Лог від {current_time} ---\n")
        for obj in detected_objects:
            f.write(f"{current_time} - {obj}\n")
        f.write("\n")  # Розділення записів


def filter_image(frame):
    """Обробка зображення: фільтрація шуму та корекція освітлення."""
    frame = cv2.GaussianBlur(frame, (5, 5), 0)  # Зменшення шуму
    frame = cv2.normalize(frame, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)  # Корекція освітлення
    return frame


def draw_datetime(frame):
    """Малює поточний час і дату у правому верхньому куті."""
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Позиція тексту
    position = (frame.shape[1] - 300, 30)  # Відступ від правого верхнього кута
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8  # Збільшений шрифт
    color = (255, 255, 255)  # Білий текст
    thickness = 2
    cv2.putText(frame, current_time, position, font, font_scale, color, thickness)


def draw_recording_timer(frame, start_time):
    """Малює секундомір часу запису під поточним часом."""
    elapsed_time = datetime.datetime.now() - start_time
    seconds = elapsed_time.total_seconds()
    formatted_time = f"{int(seconds // 3600):02}:{int((seconds % 3600) // 60):02}:{int(seconds % 60):02}"
    # Позиція тексту
    position = (frame.shape[1] - 300, 70)  # Нижче поточного часу
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7  # Менший шрифт
    color = (0, 0, 255)  # Червоний текст
    thickness = 2
    cv2.putText(frame, formatted_time, position, font, font_scale, color, thickness)
