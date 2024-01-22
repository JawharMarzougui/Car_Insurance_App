from roboflow import Roboflow


#we will be using roboflow pre-trained model
rf = Roboflow(api_key="Tc2h5YtqzXnEyY01651e")
project = rf.workspace().project("car-damage-detection-t0g92")
model = project.version(3).model
