import cv2
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.uix.label import Label
from ultralytics import YOLO

class DetectionApp(App):
    def build(self):
        self.model = YOLO("D:/yolov8/yolov8s.pt")  # Update with your model path
        self.camera = cv2.VideoCapture(0)  # Change to the correct camera index if needed
        self.img_widget = Image()
        self.is_detecting = False

        layout = BoxLayout(orientation='vertical')
        layout.add_widget(self.img_widget)

        self.start_button = Button(text='Start Detection', size_hint=(1, 0.1))
        self.start_button.bind(on_press=self.start_detection)
        layout.add_widget(self.start_button)

        self.stop_button = Button(text='Stop Detection', size_hint=(1, 0.1))
        self.stop_button.bind(on_press=self.stop_detection)
        layout.add_widget(self.stop_button)

        self.status_label = Label(text='Status: Waiting', size_hint=(1, 0.1))
        layout.add_widget(self.status_label)

        return layout

    def start_detection(self, instance):
        try:
            self.is_detecting = True
            self.status_label.text = "Status: Detecting"
            Clock.schedule_interval(self.update_frame, 1.0 / 30.0)  # Update at 30 FPS
        except Exception as e:
            self.status_label.text = f"Error: {str(e)}"
            self.is_detecting = False

    def stop_detection(self, instance):
        self.is_detecting = False
        self.status_label.text = "Status: Stopped"
        Clock.unschedule(self.update_frame)

    def update_frame(self, dt):
        if self.is_detecting:
            ret, frame = self.camera.read()
            if ret:
                results = self.model(frame)
                annotated_frame = results[0].plot()  # Assuming results[0] gives the image with annotations
                self.img_widget.texture = self.convert_to_texture(annotated_frame)
            else:
                self.status_label.text = "Error: Unable to read from camera."

    def convert_to_texture(self, frame):
        buf = cv2.flip(frame, 0)  # Flip the frame vertically
        buf = buf.tobytes()  # Convert to bytes for texture
        texture = self.img_widget.texture
        texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        texture.flip_vertical()
        return texture

    def on_stop(self):
        self.camera.release()  # Release the camera when the app is closed

if __name__ == '__main__':
    try:
        DetectionApp().run()
    except Exception as e:
        print(f"Application error: {str(e)}")
