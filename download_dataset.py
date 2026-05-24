from roboflow import Roboflow

rf = Roboflow(api_key="wS4ds9EfOb5XlJ1lISsJ")

project = rf.workspace("lanedetection-lihli").project("lane_train_v6-2egiy")

version = project.version(1)
dataset = version.download("yolov8")

print("✅ Dataset downloaded!")
print("📁 Saved to:", dataset.location)