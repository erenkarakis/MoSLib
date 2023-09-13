import cv2
from numpy import ndarray
from math import tan, radians
from .utils import math_utils


def ORB_detector(img1: cv2.Mat | ndarray, img2: cv2.Mat | ndarray, nfeatures: int=1000, debug: bool=False):

    orb = cv2.ORB_create(nfeatures=nfeatures)

    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    bruteForceMatcher = cv2.BFMatcher()
    matches = bruteForceMatcher.knnMatch(des1, des2, k=2)

    goodMatches = []

    for m1, m2 in matches:
        if m1.distance < 0.75 * m2.distance:
            goodMatches.append([m1])
    
    match_pts = []

    for match in goodMatches: # TODO: Make another function for drawing debug things
        p1 = kp1[match[0].queryIdx].pt
        p2 = kp2[match[0].queryIdx].pt
        match_pts.append((p1, p2))
        cv2.circle(img1, (int(p1[0]), int(p1[1])), 8, (255, 0, 255), cv2.FILLED)
        cv2.circle(img2, (int(p2[0]), int(p2[1])), 8, (255, 0, 255), cv2.FILLED)

        #matches = cv2.drawMatchesKnn(img1, kp1, img2, kp2, goodMatches, None, flags=2)
    #goodMatches, matches
    return match_pts

def focal_length(self, widthInCm: float, distanceFromCam: float, widthInPixels: int) -> float:
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

def distance_monocular(self, focal_legth: float, widthInCm: float, widthInPixels: int) -> float:
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

    distance = ((widthInCm * focal_length) / widthInPixels)
    return distance

def triangulate_streo(self, left_cam: tuple[float, float], right_cam: tuple[float, float], c2c_distance: float) -> tuple[float, float, float, float]:
    """
    Calculates the depth information to given point and returns x, y, z data.
    
    ### Parameters
        `left_cam: tuple[float, float]`:
            Left camera to point angles for X axis and Y axis.
        `right_cam: tuple[float, float]`:
            Right camera to point angles for X axis and Y axis.
        `c2c_distance: float`:
            Distance between two camera.
    
    ### Returns
        Depth information about given point.
    """
    pass

def find_disparity(rcam_pt: tuple[int, int], lcam_pt: tuple[int, int]):
    return lcam_pt[0] - rcam_pt[0]

def perspective_projection(rcam_pt: tuple[int, int], lcam_pt: tuple[int, int], home_point: tuple[int, int], fl: float, c2c_distance: float, disparity: int):

    if disparity == 0 or disparity == None:
        return

    x = (((c2c_distance * (lcam_pt[0] - home_point[0])) / disparity))
    y = ((c2c_distance * fl * (lcam_pt[1] - home_point[1])) / (fl * disparity))

    z = (c2c_distance * fl) / disparity

    return x, y, z

