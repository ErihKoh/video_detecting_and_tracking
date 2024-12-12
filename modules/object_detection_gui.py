from PyQt5.QtWidgets import QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget, QHBoxLayout, QMessageBox, \
    QFormLayout, QGroupBox, QSpinBox, QSlider
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt
import cv2


class ObjectDetectionGUI(QMainWindow):
    def __init__(self, app):
        super().__init__()
        self.app = app
        self.running = True

        # Налаштування вікна
        self.setWindowTitle("Object Detection and Tracking")
        self.setGeometry(100, 100, 1280, 720)

        # Центральний віджет
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        # Відео віджет
        self.video_label = QLabel(self)
        self.video_label.setStyleSheet("background-color: black;")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(1024, 576)

        # Кнопки
        self.start_button = QPushButton("Start Recording")
        self.start_button.clicked.connect(self.start_recording)

        self.stop_button = QPushButton("Stop Recording")
        self.stop_button.clicked.connect(self.stop_recording)

        self.screenshot_button = QPushButton("Take Screenshot")
        self.screenshot_button.clicked.connect(self.take_screenshot)

        self.quit_button = QPushButton("Quit")
        self.quit_button.clicked.connect(self.quit_application)

        # Розташування кнопок
        buttons_layout = QHBoxLayout()
        buttons_layout.addWidget(self.start_button)
        buttons_layout.addWidget(self.stop_button)
        buttons_layout.addWidget(self.screenshot_button)
        buttons_layout.addWidget(self.quit_button)

        # Параметри управління
        self.confidence_slider = QSlider(Qt.Horizontal)
        self.confidence_slider.setMinimum(1)
        self.confidence_slider.setMaximum(100)
        self.confidence_slider.setValue(int(self.app.confidence_threshold * 100))
        self.confidence_slider.valueChanged.connect(self.update_confidence_threshold)
        self.confidence_slider.setFixedSize(200, 20)  # Ширина: 200 пікселів, висота: 20 пікселів

        self.confidence_label = QLabel(f"Conf. Threshold: {self.app.confidence_threshold:.2f}")
        self.confidence_label.setAlignment(Qt.AlignCenter)

        # Макет управління параметрами
        parameter_layout = QVBoxLayout()
        parameter_layout.addWidget(QLabel("Conf. Threshold:"))
        parameter_layout.addWidget(self.confidence_slider)
        parameter_layout.addWidget(self.confidence_label)

        parameter_group = QGroupBox("Detection Parameters")
        parameter_group.setLayout(parameter_layout)

        # Основний макет
        layout = QVBoxLayout(self.central_widget)
        layout.addWidget(self.video_label)
        layout.addLayout(buttons_layout)
        layout.addWidget(parameter_group)

        # Таймер для оновлення відео
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_video)
        self.timer.start(30)

    def update_video(self):
        """Оновлення відеопотоку."""
        if not self.running:
            return

        ret, frame = self.app.video_cap.read()
        if ret:
            frame = self.app.process_frame()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Перетворення у QImage
            height, width, channel = frame.shape
            bytes_per_line = channel * width
            qimg = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)

            # Відображення на QLabel
            pixmap = QPixmap.fromImage(qimg)
            self.video_label.setPixmap(pixmap.scaled(
                self.video_label.width(),
                self.video_label.height(),
                Qt.KeepAspectRatio
            ))

    def start_recording(self):
        if self.app.video_recorder.is_recording:
            QMessageBox.warning(self, "Recording", "Recording is already in progress!")
        else:
            self.app.toggle_recording()
            QMessageBox.information(self, "Recording", "Recording started.")

    def stop_recording(self):
        if not self.app.video_recorder.is_recording:
            QMessageBox.warning(self, "Recording", "Recording is not active!")
        else:
            self.app.toggle_recording()
            QMessageBox.information(self, "Recording", "Recording stopped.")

    def take_screenshot(self):
        ret, frame = self.app.video_cap.read()
        if ret:
            self.app.save_screenshot(frame)
            QMessageBox.information(self, "Screenshot", "Screenshot saved successfully!")
        else:
            QMessageBox.warning(self, "Screenshot", "Failed to capture screenshot.")

    def update_confidence_threshold(self):
        """Оновлення порогу довіри."""
        new_threshold = self.confidence_slider.value() / 100.0
        self.app.confidence_threshold = new_threshold
        self.confidence_label.setText(f"Confidence Threshold: {new_threshold:.2f}")

    def quit_application(self):
        self.running = False
        self.app.video_cap.release()
        if self.app.video_recorder.writer is not None:
            self.app.writer.release()
        self.close()
