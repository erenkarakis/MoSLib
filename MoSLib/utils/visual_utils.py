import cv2, numpy as np


def read_classes_from_file(classesFilePath: str):
    '''Reads the class names from file, creates a color list for bounding boxes. Returns class names list and class colors.
    @param classFilePath Path of the class names file.'''
    np.random.seed(449)
    with open(classesFilePath, "r") as file:
        classes = file.read().splitlines()

        # Colors list for bounding box
        classesColors = np.random.uniform(low=0, high=255, size=(len(classes), 3))
    return classes, classesColors

def create_bounding_box(img: cv2.Mat | np.ndarray, bboxCoordinates: tuple, rectangleThickness: int=2, cornerThickness: int=3, bboxColor: tuple=(255,0,255), cornerColor: tuple=(0,255,0)):
    '''Creates a bounding box around detected class. It also adds diffrent colored corners.
    @param img Image.
    @param bboxCoordinates ((x min, y min), (x max, y max)) location of the bounding box.
    @param rectangleThickness Thickness of the bounding box.
    @param cornerThickness Thickness of the corners.
    @param bboxColor Color of the bounding box.
    @param cornerColor Color of corners.'''

    cv2.rectangle(img, (bboxCoordinates[0][0], bboxCoordinates[0][1]), (bboxCoordinates[1][0], bboxCoordinates[1][1]), bboxColor, rectangleThickness)

    line_width = min(int((bboxCoordinates[1][0] - bboxCoordinates[0][0]) * 0.2), int((bboxCoordinates[1][1] - bboxCoordinates[0][1])* 0.2))

    cv2.line(img, (bboxCoordinates[0][0], bboxCoordinates[0][1]), (bboxCoordinates[0][0] + line_width, bboxCoordinates[0][1]), cornerColor, cornerThickness)
    cv2.line(img, (bboxCoordinates[0][0], bboxCoordinates[0][1]), (bboxCoordinates[0][0], bboxCoordinates[0][1] + line_width), cornerColor, cornerThickness)

    cv2.line(img, (bboxCoordinates[0][0], bboxCoordinates[1][1]), (bboxCoordinates[0][0], bboxCoordinates[1][1] - line_width), cornerColor, cornerThickness)
    cv2.line(img, (bboxCoordinates[0][0], bboxCoordinates[1][1]), (bboxCoordinates[0][0] + line_width, bboxCoordinates[1][1]), cornerColor, cornerThickness)

    cv2.line(img, (bboxCoordinates[1][0], bboxCoordinates[0][1]), (bboxCoordinates[1][0], bboxCoordinates[0][1] + line_width), cornerColor, cornerThickness)
    cv2.line(img, (bboxCoordinates[1][0], bboxCoordinates[0][1]), (bboxCoordinates[1][0] - line_width, bboxCoordinates[0][1]), cornerColor, cornerThickness)

    cv2.line(img, (bboxCoordinates[1][0], bboxCoordinates[1][1]), (bboxCoordinates[1][0], bboxCoordinates[1][1] - line_width), cornerColor, cornerThickness)
    cv2.line(img, (bboxCoordinates[1][0], bboxCoordinates[1][1]), (bboxCoordinates[1][0] - line_width, bboxCoordinates[1][1]), cornerColor, cornerThickness)

def put_text_box(img: cv2.Mat | np.ndarray, textLocation: tuple, text: str="BBox Text: 1.0", textColor: tuple=(255,0,0), font=cv2.FONT_HERSHEY_PLAIN, 
                        fontSize: int=1, textThickness: int=2, recThickness: int=-1, recColor: tuple=(0,0,255)):
    '''Adds the text to the left top corner of detected class's bounding box. It automaticlly adjust the size of the textbox size.
    @param img Image.
    @param bboxCoordinates ((x min, y min), (x max, y max)) location of the bounding box.
    @param text Text.
    @param textColor Text color.
    @param font Text font style.
    @param fontSize Font size'''

    size, _ = cv2.getTextSize(text, font, fontSize, textThickness)
    textWidth, textHeight = size

    cv2.rectangle(img, (max(0, textLocation[0][0]), max(0, textLocation[0][1])), 
                    (max(0, textLocation[0][0]+textWidth), max(40, textLocation[0][1]-(textHeight*2))), recColor, recThickness)

    cv2.putText(img, text, (textLocation[0][0], textLocation[0][1]-int(textHeight/2)), font, fontSize, textColor, textThickness)
    
def fps_counter(img: cv2.Mat | np.ndarray, cTime: float, pTime: float, pt: tuple=(20,30), color: tuple=(255,0,0)):
    '''Shows the fps on the screen
    @param img Image.
    @param cTime Current time.
    @param pTime Previous time.
    @param pt Location of the fps counter on the screen.
    @param color Color'''
    fps = int(1/(cTime-pTime))
    cv2.putText(img, f"FPS: {fps}", pt, cv2.FONT_HERSHEY_PLAIN, 2, color, 2)