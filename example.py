import MoSLib
from math import atan2

depth_factor = MoSLib.math_utils.calculate_depth_factor(1280, 70.0)
print(depth_factor)

point_angle = MoSLib.math_utils.calculate_point_angle(depth_factor, (1280, 720), (320, 0), 0)
print(point_angle)