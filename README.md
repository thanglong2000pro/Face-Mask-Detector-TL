# Face Mask Detector-TL

## INTRODUCTION ##
**COVID-19 face mask detector using OpenCV, Keras/TensorFlow, and Deep Learning**
- This is my project in DL Class, which is organized by AI4E. The COVID-19 mask detector I'm building here today could potentially be used to help ensure your safety and the safety of others. COVID-19 face mask detector used to:
  - Detect COVID-19 face masks in images
  - Detect face masks in video
  - Detect face masks in real-time video streams
- Solution:
  - Detecting faces in images/video
  - Extracting each individual face
  - Applying our face mask classifier

## PROJECT DEMO ##
- **Detect in images**
<img src="Readme_images/demo_image.png">
<img src="Readme_images/demo_image2.png">

- **Detect in real-time video streams**
<img src="Readme_images/demo_webcam.gif">

## FRAMEWORK USED ##
- OpenCV
- Tensorflow
- Keras
- Caffe-based face detector
- MobileNetV2

## INSTALLATION AND RUNNING ##
1. Clone the repo
2. Use the package manager pip to install package
'pip install -r requirement.txt'
3. Open source code and read how to run

## RESULT ##
<img src="Readme_images/Evaluating Network.png">

**Plot**
<img src="Readme_images/plot.png">

## NEED IMPROVEMENT ##
Our current method of detecting whether a person is wearing a mask or not is a two-step process:

Step 1: Perform face detection
Step 2: Apply our face mask detector to each face
If enough of the face is obscured, the face cannot be detected, and therefore, the face mask detector will not be applied.

## REFERENCE ##
- [Pyimagesearch](https://www.pyimagesearch.com/2020/05/04/covid-19-face-mask-detector-with-opencv-keras-tensorflow-and-deep-learning/)
- [Towardsdatascience](https://towardsdatascience.com/covid-19-face-mask-detection-using-tensorflow-and-opencv-702dd833515b)
- [data-flair blog](https://data-flair.training/blogs/face-mask-detection-with-python/)
- [Github](https://github.com/chandrikadeb7/Face-Mask-Detection)
- [Youtube](https://www.youtube.com/watch?v=Ax6P93r32KU)
