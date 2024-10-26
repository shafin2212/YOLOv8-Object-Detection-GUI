import sys
import cv2
import numpy as np
from PyQt5.QtCore import QTimer, QThread, pyqtSignal, Qt
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QFileDialog, QVBoxLayout, QWidget, QHBoxLayout
from ultralytics import YOLO
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# Worker thread for running YOLO inference
class YOLOWorker(QThread):
    frame_updated = pyqtSignal(np.ndarray, dict)

    def __init__(self, model, cap):
        super().__init__()
        self.model = model
        self.cap = cap
        self.running = True

    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                # Run inference
                results = self.model(frame)
                annotated_frame = results[0].plot()
                detected_objects = {self.model.names[int(cls)]: results[0].boxes.conf[i].item() for i, cls in enumerate(results[0].boxes.cls)}
                self.frame_updated.emit(annotated_frame, detected_objects)

    def stop(self):
        self.running = False
        self.cap.release()

# Canvas for 3D visualization
class MplCanvas(FigureCanvas):
    def __init__(self):
        fig = plt.figure()
        self.ax = fig.add_subplot(111, projection='3d')
        super().__init__(fig)

    def plot(self, detected_objects):
        self.ax.clear()
        if not detected_objects:  # Check if dictionary is empty
            self.draw()  # Clear the canvas if no objects detected
            return

        x = np.arange(len(detected_objects))
        y = list(detected_objects.values())
        z = np.zeros(len(detected_objects))

        self.ax.bar3d(x, y, z, dx=0.2, dy=0.2, dz=y, color='b', alpha=0.7)
        self.ax.set_xticks(x)
        self.ax.set_xticklabels(list(detected_objects.keys()))
        self.ax.set_xlabel('Objects')
        self.ax.set_ylabel('Confidence')
        self.ax.set_zlabel('Count')

        self.draw()

# Main application window
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("YOLOv8 Detection GUI")
        self.setGeometry(200, 100, 1200, 600)

        # Set dark blue background
        self.setStyleSheet("background-color: #001F3F;")

        # Load model button
        self.load_button = QPushButton("Load Model")
        self.load_button.clicked.connect(self.select_model)
        self.load_button.setStyleSheet("border-radius: 15px; background-color: #007BFF; color: white;")

        # Start detection button
        self.start_button = QPushButton("Start Detection")
        self.start_button.setEnabled(False)
        self.start_button.clicked.connect(self.start_detection)
        self.start_button.setStyleSheet("border-radius: 15px; background-color: #28A745; color: white;")

        # Stop detection button
        self.stop_button = QPushButton("Stop Detection")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_detection)
        self.stop_button.setStyleSheet("border-radius: 15px; background-color: #DC3545; color: white;")

        # Camera feed display
        self.camera_feed = QLabel()
        self.camera_feed.setAlignment(Qt.AlignCenter)
        self.camera_feed.setStyleSheet("background-color: #333;")

        # Detected objects display
        self.detected_objects_label = QLabel()
        self.detected_objects_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.detected_objects_label.setStyleSheet("color: white; font-size: 16px; background-color: rgba(0, 0, 0, 0.5); padding: 10px;")
        self.detected_objects_label.setFixedWidth(200)

        # Initialize the 3D graph canvas
        self.canvas = MplCanvas()

        # Arrange the layout
        layout = QHBoxLayout()
        left_layout = QVBoxLayout()
        left_layout.addWidget(self.load_button)
        left_layout.addWidget(self.start_button)
        left_layout.addWidget(self.stop_button)
        left_layout.addWidget(self.camera_feed)

        right_layout = QVBoxLayout()
        right_layout.addWidget(self.detected_objects_label)
        right_layout.addWidget(self.canvas)

        layout.addLayout(left_layout)
        layout.addLayout(right_layout)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # Set up YOLO and video capture variables
        self.model = None
        self.cap = None
        self.worker = None

    def select_model(self):
        pt_file, _ = QFileDialog.getOpenFileName(self, "Select YOLO Model", "", "PyTorch Model (*.pt)")
        if pt_file:
            self.model = YOLO(pt_file)
            print(f"Model loaded successfully from {pt_file}")
            self.start_button.setEnabled(True)

    def start_detection(self):
        if self.model:
            # Select a video file
            video_file, _ = QFileDialog.getOpenFileName(self, "Select Video File", "", "Video Files (*.mp4 *.avi)")
            if video_file:
                self.cap = cv2.VideoCapture(video_file)
                self.worker = YOLOWorker(self.model, self.cap)
                self.worker.frame_updated.connect(self.update_camera_feed)
                self.worker.start()
                self.start_button.setEnabled(False)
                self.stop_button.setEnabled(True)
                print("Starting YOLOv8 detection...")

    def stop_detection(self):
        if self.worker:
            self.worker.stop()
            self.worker.wait()
            self.worker = None
            self.cap.release()
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.camera_feed.clear()
            self.detected_objects_label.clear()
            self.canvas.ax.clear()
            self.canvas.draw()  # Clear the graph
            print("Detection stopped.")

    def update_camera_feed(self, frame, detected_objects):
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)

        pixmap = QPixmap.fromImage(qt_image)
        self.camera_feed.setPixmap(pixmap)

        # Update detected objects display
        detected_objects_text = "\n".join([f"{obj}: {conf:.2f}" for obj, conf in detected_objects.items()])
        self.detected_objects_label.setText(f"Detected Objects:\n{detected_objects_text}")

        # Update the 3D graph representation
        self.canvas.plot(detected_objects)

    def paintEvent(self, event):
        # Draw dark blue background
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(0, 31, 63))
        painter.end()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
