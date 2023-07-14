# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Main script to run the object detection routine."""
import argparse
import sys
import time
import sqlite3
from datetime import datetime

from flask import Flask, render_template, Response, request

app = Flask(__name__)
app.debug = True

conn = sqlite3.connect('database.db')
print('Opened database successfully!')

conn.execute('CREATE TABLE IF NOT EXISTS objects (id INTEGER PRIMARY KEY AUTOINCREMENT, object_name TEXT, object_percentage TEXT, object_datetime TEXT)')
print('Create table successfully')
conn.close()

import cv2
from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision
import utils




def run(model: str, camera_id: int, width: int, height: int, num_threads: int,
        enable_edgetpu: bool) -> None:
  """Continuously run inference on images acquired from the camera.

  Args:
    model: Name of the TFLite object detection model.
    camera_id: The camera id to be passed to OpenCV.
    width: The width of the frame captured from the camera.
    height: The height of the frame captured from the camera.
    num_threads: The number of CPU threads to run the model.
    enable_edgetpu: True/False whether the model is a EdgeTPU model.
  """

  # Variables to calculate FPS
  counter, fps = 0, 0
  start_time = time.time()

  # Start capturing video input from the camera
  cap = cv2.VideoCapture(camera_id)
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

  # Visualization parameters
  row_size = 20  # pixels
  left_margin = 24  # pixels
  text_color = (0, 0, 255)  # red
  font_size = 1
  font_thickness = 1
  fps_avg_frame_count = 10

  # Initialize the object detection model
  base_options = core.BaseOptions(
      file_name=model, use_coral=enable_edgetpu, num_threads=num_threads)
  detection_options = processor.DetectionOptions(
      max_results=5, score_threshold=0.4)
  options = vision.ObjectDetectorOptions(
      base_options=base_options, detection_options=detection_options)
  detector = vision.ObjectDetector.create_from_options(options)

  # Continuously capture images from the camera and run inference
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      sys.exit(
          'ERROR: Unable to read from webcam. Please verify your webcam settings.'
      )

    counter += 1
    image = cv2.flip(image, 1)

    # Convert the image from BGR to RGB as required by the TFLite model.
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Create a TensorImage object from the RGB image.
    input_tensor = vision.TensorImage.create_from_array(rgb_image)

    # Run object detection estimation using the model.
    detection_result = detector.detect(input_tensor)
    
    # Connecting to database
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    
    for detection in detection_result.detections:
        category = detection.categories[0]
        category_name = category.category_name
        category_percentage = category.score
        print(category_name)
        print(category_percentage)
        print("")
        
        c.execute("INSERT INTO objects (object_name, object_percentage, object_datetime) VALUES (?, ?, ?)", (category_name, category_percentage, datetime.now().strftime('%d/%m/%Y %H:%M:%S')))
        
    conn.commit()
    conn.close()
        
        

    # Draw keypoints and edges on input image
    image = utils.visualize(image, detection_result)

    # Calculate the FPS
    if counter % fps_avg_frame_count == 0:
      end_time = time.time()
      fps = fps_avg_frame_count / (end_time - start_time)
      start_time = time.time()

    # Show the FPS
    fps_text = 'FPS = {:.1f}'.format(fps)
    text_location = (left_margin, row_size)
    cv2.putText(image, fps_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                font_size, text_color, font_thickness)

    # Stop the program if the ESC key is pressed.
    if cv2.waitKey(1) == 27:
      break
    yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + cv2.imencode('.jpg', image)[1].tobytes() + b'\r\n\r\n')

  cap.release()
  cv2.destroyAllWindows()


def main():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      '--model',
      help='Path of the object detection model.',
      required=False,
      default='efficientdet_lite0.tflite')
  parser.add_argument(
      '--cameraId', help='Id of camera.', required=False, type=int, default=0)
  parser.add_argument(
      '--frameWidth',
      help='Width of frame to capture from camera.',
      required=False,
      type=int,
      default=640)
  parser.add_argument(
      '--frameHeight',
      help='Height of frame to capture from camera.',
      required=False,
      type=int,
      default=480)
  parser.add_argument(
      '--numThreads',
      help='Number of CPU threads to run the model.',
      required=False,
      type=int,
      default=4)
  parser.add_argument(
      '--enableEdgeTPU',
      help='Whether to run the model on EdgeTPU.',
      action='store_true',
      required=False,
      default=False)
  args = parser.parse_args()

  run(args.model, int(args.cameraId), args.frameWidth, args.frameHeight,
      int(args.numThreads), bool(args.enableEdgeTPU))

@app.route('/video_feed')
def video_feed():
    model = 'efficientdet_lite0.tflite'  # Replace with the path to your TensorFlow Lite model
    camera_id = 0  # Set the camera ID appropriately
    width = 640  # Set the desired width of the captured frames
    height = 480  # Set the desired height of the captured frames
    num_threads = 4  # Set the number of CPU threads to run the model
    enable_edgetpu = False  # Set to True if the model is an EdgeTPU model
 
    # Call the modified run() function to get the processed frames
    return Response(run(model, camera_id, width, height, num_threads, enable_edgetpu),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/live')
def live():
    return render_template('live.html')

# Method for fetching data from database
def fetch_data():
    conn = sqlite3.connect("database.db")
 
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM objects')
    rows = cursor.fetchall()
    return rows

@app.route('/archive', methods = ['GET'])
def archive():
    
    rows = fetch_data()
    rows_per_page = 10
    total_pages = int(len(rows) / rows_per_page) + 1
    
 
    return render_template('archive.html', rows = rows, total_pages = total_pages, rows_per_page = rows_per_page)
 
@app.route('/delete', methods = ['GET', 'POST'])
def delete_data():
    if request.method == 'POST':
        conn = sqlite3.connect("database.db")
        cursor = conn.cursor()
        cursor.execute('DELETE FROM objects')
        conn.commit()
        cursor.close()
        conn.close()
        
    return render_template('index.html')

if __name__ == '__main__':
  app.run(host='0.0.0.0', port = 5000)
