OLOv8 Object Detection GUI
Overview
This project is a real-time object detection application built using Python, OpenCV, and YOLOv8. The interface, created with PyQt5, enables users to detect objects from both live camera feeds and video files, with additional 3D data visualization for tracking detection confidence levels.

Features
Real-time Object Detection: Uses YOLOv8 for fast and accurate object recognition across various input sources.
Video and Camera Detection: Detect objects from a live camera or choose a pre-recorded video file.
User-Friendly GUI: Interactive interface built with PyQt5 for a seamless user experience.
3D Data Visualization: Integrated 3D bar chart displays detected object types and their
To set up a virtual environment for your YOLOv8 Object Detection project, you can follow these steps:
1. Navigate to Your Project Directory
First, open a terminal or command prompt and navigate to the folder where your project is located.

```bash
cd /path/to/your/project
```
2. Create a Virtual Environment
Use `venv` to create a virtual environment named `env`. 

```bash
python -m venv env
```

This command will create a folder named `env` in your project directory, containing all necessary dependencies.
3. Activate the Virtual Environment
Now, activate the virtual environment.

- On Windows:

  ```bash
  .\env\Scripts\activate
  ```
  On macOS and Linux:**

  ```bash
  source env/bin/activate
  ```

Once activated, your terminal will show `(env)` before the directory path, indicating that the virtual environment is active.
4. Install Required Packages
After activating the environment, install the dependencies listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```
5. Deactivate the Virtual Environment (When Done)
To deactivate the virtual environment when you’re finished, simply use:

```bash
deactivate
```

Now your project is isolated from your system Python packages, making it easier to manage dependencies.
To ensure smooth execution of the YOLOv8 Object Detection application, please follow these guidelines for organizing project files:

Project Structure
Place the following files in the same directory as your main application script to ensure everything runs correctly:

1.YOLOv8 Model File (.pt):
   - The `.pt` file, which contains the pre-trained YOLOv8 model, should be stored in the same directory. This will allow the application to load the model without specifying complex paths.

2. Other Required Files (if any):
   - Additional data files, configuration files, or any other resources your application relies on should also be placed in this directory for seamless access.
Sample Project Directory Structure:
Your project folder should look similar to this:

```plaintext
/your_project_directory
│
├── yolov8_object_detection.py         # Main application script
├── yolov8_model.pt                    # YOLOv8 model file
├── requirements.txt                   # List of dependencies
└── env/                               # (Optional) Virtual environment folder
```

Running the Application
After setting up your files as shown above and installing the dependencies, you can run the application directly by executing the main script:

```bash
python yolov8_object_detection.py
```

This organization will simplify the setup and make it easier for others to understand and replicate your project structure if shared publicly.
