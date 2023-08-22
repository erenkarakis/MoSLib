import cv2
from numpy import ndarray

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

    
