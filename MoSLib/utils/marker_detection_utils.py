import cv2
import cv2.aruco as aruco 

video_capture = False
cap = cv2.VideoCapture(1)

def find_aruco(img, marker_size=7, total_markers=250, draw=True):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    key = getattr(aruco,f'DICT_{marker_size}X{marker_size}_{total_markers}')
    arucoDict = aruco.getPredefinedDictionary(key)
    arucoParam = aruco.DetectorParameters()
    bbox, ids, _= aruco.detectMarkers(gray, dictionary=arucoDict, parameters=arucoParam)
    print(ids)
    if draw:
        aruco.drawDetectedMarkers(img, bbox)

    return bbox, ids

while True:
    if video_capture: _, img = cap.read()
    else: 
        img = cv2.imread("Images/aruco.png")
        img = cv2.resize(img, (0,0), fx=0.2, fy=0.2)
    bbox, ids = find_aruco(img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    cv2.imshow("img", img)
