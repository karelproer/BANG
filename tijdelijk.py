from robot import *
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

camera_config = {
    "top": OpenCVCameraConfig(index_or_path=0, width=1920, height=1080, fps=30),
    "wrist":OpenCVCameraConfig(index_or_path=0, width=1920, height=1080, fps=30),
}

leader, follower = connect_leader, connect_follower()
while True:
    follower.send_action(leader.get_observation)