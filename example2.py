import numpy as np
import cv2
import argparse
import sys
import MoSLib


def save_coefficients(mtx, dist, path):
    """ Save the camera matrix and the distortion coefficients to given path/file. """
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
    cv_file.write("K", mtx)
    cv_file.write("D", dist)
    # note you *release* you don't close() a FileStorage object
    cv_file.release()


def save_stereo_coefficients(path, K1, D1, K2, D2, R, T, E, F, R1, R2, P1, P2, Q):
    """ Save the stereo coefficients to given path/file. """
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
    cv_file.write("K1", K1)
    cv_file.write("D1", D1)
    cv_file.write("K2", K2)
    cv_file.write("D2", D2)
    cv_file.write("R", R)
    cv_file.write("T", T)
    cv_file.write("E", E)
    cv_file.write("F", F)
    cv_file.write("R1", R1)
    cv_file.write("R2", R2)
    cv_file.write("P1", P1)
    cv_file.write("P2", P2)
    cv_file.write("Q", Q)
    cv_file.release()


def load_coefficients(path):
    """ Loads camera matrix and distortion coefficients. """
    # FILE_STORAGE_READ
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)

    # note we also have to specify the type to retrieve other wise we only get a
    # FileNode object back instead of a matrix
    camera_matrix = cv_file.getNode("K").mat()
    dist_matrix = cv_file.getNode("D").mat()

    cv_file.release()
    return [camera_matrix, dist_matrix]


def load_stereo_coefficients(path):
    """ Loads stereo matrix coefficients. """
    # FILE_STORAGE_READ
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)

    # note we also have to specify the type to retrieve other wise we only get a
    # FileNode object back instead of a matrix
    K1 = cv_file.getNode("K1").mat()
    D1 = cv_file.getNode("D1").mat()
    K2 = cv_file.getNode("K2").mat()
    D2 = cv_file.getNode("D2").mat()
    R = cv_file.getNode("R").mat()
    T = cv_file.getNode("T").mat()
    E = cv_file.getNode("E").mat()
    F = cv_file.getNode("F").mat()
    R1 = cv_file.getNode("R1").mat()
    R2 = cv_file.getNode("R2").mat()
    P1 = cv_file.getNode("P1").mat()
    P2 = cv_file.getNode("P2").mat()
    Q = cv_file.getNode("Q").mat()

    cv_file.release()
    return [K1, D1, K2, D2, R, T, E, F, R1, R2, P1, P2, Q]

parser = argparse.ArgumentParser(description='Camera calibration')
parser.add_argument('--calibration_file', type=str, required=True, help='Path to the stereo calibration file')
parser.add_argument('--left_source', type=str, required=True, help='Left video or v4l2 device name')
parser.add_argument('--right_source', type=str, required=True, help='Right video or v4l2 device name')
parser.add_argument('--is_real_time', type=int, required=True, help='Is it camera stream or video')

args = parser.parse_args()

# is camera stream or video
if args.is_real_time:
    cap_left = cv2.VideoCapture(args.left_source, cv2.CAP_V4L2)
    cap_right = cv2.VideoCapture(args.right_source, cv2.CAP_V4L2)
else:
    cap_left = cv2.VideoCapture(args.left_source)
    cap_right = cv2.VideoCapture(args.right_source)

K1, D1, K2, D2, R, T, E, F, R1, R2, P1, P2, Q = load_stereo_coefficients(args.calibration_file)  # Get cams params

if not cap_left.isOpened() and not cap_right.isOpened():  # If we can't get images from both sources, error
    print("Can't opened the streams!")
    sys.exit(-9)

# Change the resolution in need
cap_right.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # float
cap_right.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # float

cap_left.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # float
cap_left.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # float

while True:  # Loop until 'q' pressed or stream ends
    # Grab&retreive for sync images
    if not (cap_left.grab() and cap_right.grab()):
        print("No more frames")
        break

    _, leftFrame = cap_left.retrieve()
    _, rightFrame = cap_right.retrieve()
    height, width, channel = leftFrame.shape  # We will use the shape for remap

    # Undistortion and Rectification part!
    leftMapX, leftMapY = cv2.initUndistortRectifyMap(K1, D1, R1, P1, (width, height), cv2.CV_32FC1)
    left_rectified = cv2.remap(leftFrame, leftMapX, leftMapY, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
    rightMapX, rightMapY = cv2.initUndistortRectifyMap(K2, D2, R2, P2, (width, height), cv2.CV_32FC1)
    right_rectified = cv2.remap(rightFrame, rightMapX, rightMapY, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)

    # We need grayscale for disparity map.
    gray_left = cv2.cvtColor(left_rectified, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(right_rectified, cv2.COLOR_BGR2GRAY)

    disparity_image = MoSLib.depth_map(gray_left, gray_right)  # Get the disparity map

    # Show the images
    cv2.imshow('left(R)', leftFrame)
    cv2.imshow('right(R)', rightFrame)
    cv2.imshow('Disparity', disparity_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Get key to stop stream. Press q for exit
        break

# Release the sources.
cap_left.release()
cap_right.release()
cv2.destroyAllWindows()