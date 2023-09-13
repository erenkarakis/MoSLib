from math import sqrt, tan, atan, atan2, radians, degrees
import cv2
from numpy import ndarray


def distance_from_origin(*coordinates: tuple[int, int]) -> float:
    """
    Calculates the pixel distance between given points in the frame.
    
    ### Parameters
        Just give the coordinates in a tuple.
    
    ### Returns
        Distance between given points.
    """

    return sqrt(sum([x**2 for x in coordinates]))

def focal_length(widthInCm: float, distanceFromCam: float, widthInPixels: int) -> float:
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

def distance_monocular(focalLength: float, widthInCm: float, widthInPixels: int) -> float:
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

def coordinate_2d(left_cam_pt_angle: float, right_cam_pt_angle: float, c2c_distance: float, is_degrees: bool=True) -> tuple[int, int]:
    """
    Calculates the x and y coordinates to given point and returns x, y data.
    
    ### Parameters
        `left_cam_pt_angle: float`:
            Left camera to point angle in degrees.
        `right_cam_pt_angle: float`:
            Right camera to point angle in degrees.
        `c2c_distance: float`:
            Distance between two camera.
    
    ### Returns
        2D Coordinate of the point.
    """
    print("A", left_cam_pt_angle, right_cam_pt_angle)

    if is_degrees:
        left_cam_pt_angle = radians(left_cam_pt_angle)
        right_cam_pt_angle = radians(right_cam_pt_angle)

    left_tan = tan(left_cam_pt_angle)
    right_tan = tan(right_cam_pt_angle)

    Y = c2c_distance / ( 1/left_tan + 1/right_tan )
    X = Y/left_tan

    return X, Y


def depth_factors(cam_wh: tuple[int, int], angle_width: float, angle_height: float) -> tuple[float, float]:
    """
    For calculating depth factor for camera. Other calculations depends on depth factor.\n
    So if there is a miscalculation on depth measurement check depth factor or manueally tune it.
    
    ### Parameters
        `px_length: int`:
            Pixel length of the camera
        `cam_view_angle: float`:
            Your camera's field of view. Check or measure it.
    
    ### Returns
        Depth factor
    """

    xDepthFactor = (cam_wh[0] / 2) / tan(radians(angle_width / 2))
    yDepthFactor = (cam_wh[1] / 2) / tan(radians(angle_height / 2))

    return xDepthFactor, yDepthFactor

def point_angles(cam_wh: tuple[int, int], xDepthFactor: float, yDepthFactor: float, coordinate) -> tuple[float, float]:
    """
    For calculating angle to given point. This is a very important function for triangulation.\n
    Apply for each camera.
    
    ### Parameters
        `depth_factor: float`:
            The depth factor you calculated earlier.
        `cam_wh: tuple`:
            Camera width and height in a tuple.
        `pt_coordinate: tuple`:
            Coordinate of the selected point.
    
    ### Returns
        Angle of the given point.
    """
    x = coordinate[0] - cam_wh[0] / 2
    y = cam_wh[1] - coordinate[1]

    x_tan = x / xDepthFactor
    y_tan = y / yDepthFactor

    x_degree = degrees(atan(x_tan))
    y_degree = degrees(atan(y_tan))

    return x_degree, y_degree # Returned value is from half angle of camera to right or left


def perspective_projection(rcam_pt: tuple[int, int], lcam_pt: tuple[int, int], fx: float, fy: float, c2c_distance: float, pixel_density: float, focal_length: float, home_point: tuple[int, int]):
    pass
    #x = ((b * (lcam_pt[0] - home_point[0]) / ))
    
