import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import threading
import cv2


class ObjectDetectionGUI:
    def __init__(self, app):
        """
        Создание графического интерфейса для объекта `ObjectDetectionAndTracking`.
        :param app: Экземпляр класса `ObjectDetectionAndTracking`.
        """
        self.app = app
        self.root = tk.Tk()
        self.root.title("Object Detection and Tracking")
        self.root.geometry("900x700")
        self.root.configure(bg="black")

        # Видео виджет
        self.video_label = tk.Label(self.root, bg="black")
        self.video_label.pack(padx=10, pady=10, expand=True, fill=tk.BOTH)

        # Панель управления
        self.controls_frame = tk.Frame(self.root, bg="gray")
        self.controls_frame.pack(fill=tk.X, padx=10, pady=10)

        self.start_button = tk.Button(
            self.controls_frame, text="Start Recording", bg="green", fg="white", command=self.start_recording
        )
        self.start_button.grid(row=0, column=0, padx=5, pady=5)

        self.stop_button = tk.Button(
            self.controls_frame, text="Stop Recording", bg="red", fg="white", command=self.stop_recording
        )
        self.stop_button.grid(row=0, column=1, padx=5, pady=5)

        self.screenshot_button = tk.Button(
            self.controls_frame, text="Take Screenshot", bg="blue", fg="white", command=self.take_screenshot
        )
        self.screenshot_button.grid(row=0, column=2, padx=5, pady=5)

        self.quit_button = tk.Button(
            self.controls_frame, text="Quit", bg="darkred", fg="white", command=self.quit_application
        )
        self.quit_button.grid(row=0, column=3, padx=5, pady=5)

        # Поток для обновления видео
        self.running = True
        self.video_thread = threading.Thread(target=self.update_video)
        self.video_thread.start()

    def update_video(self):
        """Обновление видеопотока в интерфейсе."""
        while self.running:
            ret, frame = self.app.video_cap.read()
            if not ret:
                break
            frame = self.app.process_frame(frame)

            # Конвертация OpenCV BGR в RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

    def start_recording(self):
        """Начать запись видео."""
        self.app.toggle_recording()

    def stop_recording(self):
        """Остановить запись видео."""
        self.app.toggle_recording()

    def take_screenshot(self):
        """Сохранить текущий кадр как снимок экрана."""
        ret, frame = self.app.video_cap.read()
        if ret:
            self.app.save_screenshot(frame)
            messagebox.showinfo("Screenshot", "Screenshot saved successfully!")

    def quit_application(self):
        """Закрыть приложение."""
        self.running = False
        self.app.video_cap.release()
        if self.app.writer is not None:
            self.app.writer.release()
        self.root.destroy()

    def run(self):
        """Запуск основного цикла GUI."""
        self.root.mainloop()