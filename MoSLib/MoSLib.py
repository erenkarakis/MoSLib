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

    if debug:
        for match in goodMatches: # TODO: Make another function for drawing debug things
            p1 = kp1[match[0].queryIdx].pt
            p2 = kp2[match[0].queryIdx].pt
            cv2.circle(img1, (int(p1[0]), int(p1[1])), 8, (255, 0, 255), cv2.FILLED)
            cv2.circle(img2, (int(p2[0]), int(p2[1])), 8, (255, 0, 255), cv2.FILLED)

        matches = cv2.drawMatchesKnn(img1, kp1, img2, kp2, goodMatches, None, flags=2)

    return goodMatches, matches

def triangulate_streo(left_cam: tuple[float, float], right_cam: tuple[float, float], c2c_distance: float) -> tuple[float, float, float, float]:
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

    left_x_angle, left_y_angle = left_cam
    right_x_angle, right_y_angle = right_cam

    y_angle = (left_y_angle + right_y_angle) / 2 

    X, Y = math_utils.coordinate_2d(left_x_angle, right_x_angle, c2c_distance)

    Z = tan(radians(y_angle)) * math_utils.distance_from_origin(X, Y)

    D = math_utils.distance_from_origin(X, Y, Z)

    return X, Y, Z, D

    
