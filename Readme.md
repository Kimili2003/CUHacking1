For this project, there are 5 different
Running and install instructions
1. for the normal_picture_detect, please install all the nessary packages in the requirements.txt file.(some of the libaries may
   not be able use in MacOS)
2. for the shi_pin_bike_detect, this is detecting for most objects on the road in Beijing(video taken by myself so no copyright issues)
   if the code didn't work, try change the path of the video in the code to the absolute path of the video on your computer(because it works on mine).
3. The shi_pin_plane_count is mainly build for the satellite(maybe military use). The program can detect planes from top view and draw out
   all(70%-87% accuracy) of the plane show in picture or every frame of the video. It should be using CUDA to run it but if not, it will run on CPU.
4. The dynamic is a use of the camera detection for 80 different types of objects.
5. The false_smoke_detect is not successful yet due to small number of epochs and image_size. It will mainly detect things correct but with wrong name
   attached to it. It's build for abnormal driving behaviour detect.

Problem faced:
1. Apple Silicon mps error from YOLOv7(benched model for our project)
2. MPS efficiency not working great
3. CUDA not working on my computer(Torch and CUDA toolkit problem)
4. Training error