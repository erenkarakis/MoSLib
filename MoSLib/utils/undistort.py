import cv2 as cv
from calibration_utils import calibrate, save_coefficients, load_coefficients

ret, mtx, dist, rvecs, tvecs = calibrate(dirpath="Images/Lcam",prefix="LCam ",image_format="jpg", square_size=0.0186, width=9 ,height=6)
save_coefficients(mtx=mtx, dist=dist, path="Images/Save.xml")
camera_matrix, dist_matrix = load_coefficients(path="Images/Save.xml")
print(camera_matrix)
print()
print(dist_matrix)

img = cv.imread('Images/Lcam/LCam (15).jpg')
h, w = img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

# undistort
mapx, mapy = cv.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w,h), 5)
dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)
# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite('calibresult.png', dst)