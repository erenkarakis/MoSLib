import MoSLib
import math
import cv2
import time

vertical_angle = 64.0
horizontal_angle = 78.0
c2c_distance = 15

lcam = cv2.VideoCapture()
lcam.open("/dev/v4l/by-id/usb-046d_081b_852B89E0-video-index0")
lcam.set(3, 1280)
lcam.set(4, 720)

rcam = cv2.VideoCapture()
rcam.open("/dev/v4l/by-id/usb-046d_081b_A625B8D0-video-index0")
rcam.set(3, 1280)
rcam.set(4, 720)

previousTime = 0

while True:
    cTime = time.time()

    success1, frame_rcam = lcam.read()
    success2, frame_lcam = rcam.read()

    # frame_rcam = cv2.flip(frame_rcam, 0)
    # frame_rcam = cv2.flip(frame_rcam, 1)

    frame_lcam = cv2.cvtColor(frame_lcam, cv2.COLOR_BGR2GRAY)
    frame_rcam = cv2.cvtColor(frame_rcam, cv2.COLOR_BGR2GRAY)

    match_pts = MoSLib.ORB_detector(frame_rcam, frame_lcam)

    for match in match_pts:
        disparity = MoSLib.find_disparity(match[1], match[0])
        try:
            x, y, z = MoSLib.perspective_projection(match[0], match[1], (640, 360), 1530.0, c2c_distance, disparity)
        except:
            print("Cannot unpack non-iterable NoneType object")
        print(match[0], z)

    MoSLib.visual_utils.fps_counter(frame_lcam, cTime, previousTime)
    previousTime = cTime

    cv2.imshow("Left", frame_lcam)
    cv2.imshow("Right", frame_rcam)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

lcam.release()
rcam.release()
cv2.destroyAllWindows()
