# Liveness-Detector
A liveness detector capable of spotting fake faces and performing anti-face spooﬁng in face recognition systems.

## Program Structure
* dataset
  * fake_images
  * real_images
* face_detector
  * deploy.prototxt
  * res10_300x300_ssd_iter_140000.caffemodel
* videos
  * fake_generated_videos
  * real_generated_videos
* generate_frames.py
* le.pickle
* liveness.model
* livenessnet.py
* test.py
* trainmodel.py

There are three main directories in the project:
1. dataset: This directory consists of two classes of images:
        Fake images captured from a camera aimed at the screen while playing a video of some face.
        Real images captured from a selfie video with the phone.
2. face_detector: Consists of our pretrained Caffe face detector to locate face ROIs.
3. videos: The videos are collected from people where they first capture themselves on their phones and then that video is captured on a laptop gain to generate fake frames

## Process:
1. Collected videos from friends of themselves of around 10-15 seconds.These are the <strong>"real"</strong> videos.
2. Created <strong>"fake"</strong> videos by capturing the real videos on my laptop by playing them on my phone.
3. Manually arrange them into subdirectories of real and fake in the videos folder.
4. Ran the <strong>generate_frames.py</strong> script in the terminal/cmd to generate image frames of the videos, whilst manually arranging the generated images into subdirectories in the dataset directories.
   <br>To run: <br>&ensp;&ensp;&ensp; i. for real videos <strong><code>python generate_frames.py --input videos/real1.mp4 --output dataset/real --detector face_detector --skip 1</code></strong>
   <br>To run: <br>&ensp;&ensp;&ensp; ii. for fake videos <strong><code>python generate_frames.py --input videos/fake1.mp4 --output dataset/fake --detector face_detector --skip 4</code></strong>
   <br><br> Here skip is used to skip N frames between detections because adjacent frames will be similar.
<br>
![Generate_frames1](images/Generate_frames1.jpg?raw=true "Generate_frames1")
<br>
![Generate_frames2](images/Generate_frames2.jpg?raw=true "Generate_frames2")
<br>

5. Ran the <strong>trainmodel.py</strong> script by typing in the terminal/cmd: <strong><code>python trainmodel.py --dataset dataset --model liveness.model --le le.pickle</code></strong>
