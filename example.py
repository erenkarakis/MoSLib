import MoSLib
import math
import cv2

vertical_angle = 64.0
horizontal_angle = 78.0
c2c_distance = 5 + 15/16

depth_factors = MoSLib.math_utils.depth_factors((911, 686), horizontal_angle, vertical_angle)
print(depth_factors)

point_angles_l = MoSLib.math_utils.point_angles((911, 686), depth_factors[0], depth_factors[1], (797, 509)) # 509 
point_angles_r = MoSLib.math_utils.point_angles((911, 686), depth_factors[0], depth_factors[1], (536, 509)) # 482

output = MoSLib.triangulate_streo(point_angles_l, point_angles_r, c2c_distance)

print(output)

img_l = cv2.imread("Images/l2.png")
img_r = cv2.imread("Images/r2.png")
MoSLib.visual_utils.show_camera_center(img_l)

cv2.imshow("Img Left", img_l)
cv2.imshow("Img Right", img_r)
cv2.waitKey(0)
