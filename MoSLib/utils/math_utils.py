from math import sqrt
import cv2
from numpy import ndarray


def find_distance_px(pt1, pt2) -> float:
    return sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)

def calculate_focal_legth(widthInCm, distanceFromCam, widthInPixels) -> float:
    
    focalLength = ((widthInPixels * distanceFromCam) / widthInCm)
    return focalLength

def calculate_distance_monocular(focalLenth, widthInCm, widthInPixels) -> float:

    distance = ((widthInCm * focalLenth) / widthInPixels)
    return distance

def triangulate_streo(img_left: cv2.Mat | ndarray, img_right: cv2.Mat | ndarray, c2c_distance: float, cam_view_angle: float) -> float:
    """
    Returns basic information about the exchange.
    
    ### Parameters
        `img_left: cv2.Mat | ndarray`:
            Left frame/image of the streo cam.
        `img_right: cv2.Mat | ndarray`:
            Right frame/image of the streo cam.
        `c2c_distance: float`:
            Distance betwwen two cameras.
        `cam_view_angle: float`:
            Camera viewing angle in degrees.
    
    ### Returns
        Depth information about given point.
    """