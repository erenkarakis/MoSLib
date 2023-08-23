from math import sqrt, tan, atan2, radians, degrees
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

def triangulate_streo(img_left: cv2.Mat | ndarray, img_right: cv2.Mat | ndarray, c2c_distance: float) -> float:
    """
    Calculates the depth information to given point and returns x, y, z data.
    
    ### Parameters
        `img_left: cv2.Mat | ndarray`:
            Left frame/image of the streo cam.
        `img_right: cv2.Mat | ndarray`:
            Right frame/image of the streo cam.
        `c2c_distance: float`:
            Distance betwwen two cameras.
    
    ### Returns
        Depth information about given point.
    """

def calculate_depth_factor(px_length: int, cam_view_angle: float) -> float:
    """
    For calculating depth factor for camera. Other calculations depends on depth factor.\n
    So if there is a miscalculation on depth measurement check depth factor or manueally tune it.

    ### !! IMPORTANT !!
        You must calculate separately for vertical and horizontal. So you need vertical\n
        and horizontal viewing angles of your camera. Otherwise your calculations will be wrong.
    
    ### Parameters
        `px_length: int`:
            Pixel length of the camera
        `cam_view_angle: float`:
            Your camera's field of view. Check or measure it.
    
    ### Returns
        Depth factor
    """

    half_view_angle = cam_view_angle / 2
    depth_factor = (px_length / 2) / tan(radians(half_view_angle))
    return depth_factor

def calculate_point_angle(depth_factor: float, cam_wh: tuple[int, int], pt_coordinate: tuple, orientation: bool=0) -> float:
    """
    For calculating depth factor for camera. Other calculations depends on depth factor.\n
    So if there is a miscalculation on depth measurement check depth factor or manueally tune it.

    ### !! IMPORTANT !!
        You must calculate separately for vertical and horizontal. So you need vertical\n
        and horizontal viewing angles of your camera. Otherwise your calculations will be wrong.
    
    ### Parameters
        `depth_factor: float`:
            The depth factor you calculated earlier.
        `cam_wh: tuple`:
            Camera width and height in a tuple.
        `pt_coordinate: tuple`:
            Coordinate of the selected point.
        `orientation: bool`:
            Vertical or horizontal calculation (0 means horizontal, 1 means vertical)
    
    ### Returns
        Angle of the given point. COULD BE NEGATIVE!
    """

    px = 0

    if orientation not in (0, 1):
        raise ValueError("Only 0 and 1 accepted!")

    px = pt_coordinate[orientation] - (cam_wh[orientation] / 2)

    angle_of_pt = degrees(atan2(px, depth_factor))

    return angle_of_pt # Returned value is from half angle of camera to right or left
