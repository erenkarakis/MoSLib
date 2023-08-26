from math import sqrt, tan, atan2, radians, degrees
import cv2
from numpy import ndarray


def find_distance_px(pt1: tuple[int, int], pt2: tuple[int, int]) -> float:
    """
    Calculates the pixel distance between given points in the frame.
    
    ### Parameters
        `pt1: tuple[int, int]`:
            First point.
        `pt2: tuple[int, int]`:
            Second point.
    
    ### Returns
        Distance between given points.
    """

    return sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)

def calculate_focal_length(widthInCm: float, distanceFromCam: float, widthInPixels: int) -> float:
    """
    This function is for calibration. You need to run this for learn focal length\n
    of your camera.
    
    ### Parameters
        `widthInCm: float`:
            The size of the object to be measured.
        `distanceFromCam: float`:
            Distance to the object to be measured.
        `widthInPixels: float`:
            The pixel size of the object to be measured in the frame/image.
    
    ### Returns
        Focal length of your camera.
    """
    
    focalLength = ((widthInPixels * distanceFromCam) / widthInCm)
    return focalLength

def calculate_distance_monocular(focalLength: float, widthInCm: float, widthInPixels: int) -> float:
    """
    Calculates the depth information to given point and returns x, y, z data.
    
    ### Parameters
        `focalLength: float`:
            Focal length you calculated earlier.
        `widthInCm: float`:
            The size of the object to be measured.
        `widthInPixels: float`:
            The pixel size of the object to be measured in the frame/image.
    
    ### Returns
        Distance information about given point.
    """

    distance = ((widthInCm * focalLength) / widthInPixels)
    return distance

def triangulate_streo(left_cam_pt_angle: float, right_cam_pt_angle: float, c2c_distance: float) -> float:
    """
    Calculates the depth information to given point and returns x, y, z data.
    
    ### Parameters
        `left_cam_pt_angle: float`:
            Left camera to point angle.
        `right_cam_pt_angle: float`:
            Right camera to point angle.
        `c2c_distance: float`:
            Distance between two camera.
    
    ### Returns
        Depth information about given point.
    """

    depth = (c2c_distance * tan(left_cam_pt_angle) * tan(right_cam_pt_angle)) / (tan(left_cam_pt_angle) + tan(right_cam_pt_angle))
    return abs(depth)


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
    For calculating angle to given point. This is a very important function for triangulation.\n
    Apply for each camera.

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
