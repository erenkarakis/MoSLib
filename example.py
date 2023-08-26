import MoSLib
import math
import cv2

vertical_angle = 64.0
horizontal_angle = 78.0
c2c_distance = 5 + 15/16

depth_factor_h = MoSLib.math_utils.calculate_depth_factor(1382, horizontal_angle)
depth_factor_v = MoSLib.math_utils.calculate_depth_factor(512, vertical_angle)

point_angle_r_h = MoSLib.math_utils.calculate_point_angle(depth_factor_h, (1382, 512), (1101, 130), 0)
point_angle_r_v = MoSLib.math_utils.calculate_point_angle(depth_factor_v, (1382, 512), (1101, 130), 1)

point_angle_l_h = MoSLib.math_utils.calculate_point_angle(depth_factor_h, (1382, 512), (1137, 161), 0)
point_angle_l_v = MoSLib.math_utils.calculate_point_angle(depth_factor_v, (1382, 512), (1137, 161), 1)


img_l = cv2.imread("Images/kitti_left.png")
img_r = cv2.imread("Images/kitti_right.png")

depth_h = MoSLib.math_utils.triangulate_streo(point_angle_l_h, point_angle_r_h, c2c_distance)
depth_v = MoSLib.math_utils.triangulate_streo(point_angle_l_v, point_angle_r_v, c2c_distance)


cv2.imshow("Img Left", img_l)
cv2.imshow("Img Right", img_r)
cv2.waitKey(0)

def distance_from_origin(*coordinates):
    return math.sqrt(sum([x**2 for x in coordinates]))
