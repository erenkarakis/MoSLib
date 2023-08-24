import MoSLib
from math import atan2
import cv2

depth_factor = MoSLib.math_utils.calculate_depth_factor(1280, 70.0)
print(depth_factor)

point_angle = MoSLib.math_utils.calculate_point_angle(depth_factor, (1280, 720), (320, 0), 0)
print(point_angle)

img = cv2.imread("Images/mars_surface1.jpg")
img = cv2.resize(img, (1280, 720))
MoSLib.visual_utils.show_camera_center(img)
cv2.imshow("Img", img)
cv2.waitKey(0)
