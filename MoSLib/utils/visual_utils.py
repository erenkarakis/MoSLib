import cv2, numpy as np
from numpy import ndarray


def read_classes_from_file(classesFilePath: str) -> tuple[tuple, ndarray]:
    """
    Reads the class names from file, creates a color list for bounding boxes.
    
    ### Parameters
        `classFilePath: str`:
            Path of the class names file.
    
    ### Returns
        Class names list with class colors.
    """

    np.random.seed(449)
    with open(classesFilePath, "r") as file:
        classes = file.read().splitlines()

        # Colors list for bounding box
        classesColors = np.random.uniform(low=0, high=255, size=(len(classes), 3))
    return classes, classesColors

def create_bounding_box(img: cv2.Mat | np.ndarray, bboxCoordinates: tuple[tuple[int, int], tuple[int, int]], rectangleThickness: int=2, cornerThickness: int=3, bboxColor: tuple=(255,0,255), cornerColor: tuple=(0,255,0)) -> None:   
    """
    Creates a bounding box around detected class. It also adds diffrent colored corners.
    
    ### Parameters
        `img: cv2.Mat | np.ndarray`:
            img Image/Frame.
        `bboxCoordinates: tuple[tuple[int, int], tuple[int, int]]`:
            Top left and bottom right locations of the box.
        `rectangleThickness: int`:
            Thickness of the bounding box.
        `cornerThickness: int`:
            Thickness of the corners.
        `bboxColor: tuple`:
            Color of the bounding box.
        `cornerColor: tuple`:
            Color of corners.
    
    ### Returns
        None.
    """

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

def put_text_box(img: cv2.Mat | np.ndarray, textLocation: tuple[int, int], text: str="BBox Text: 1.0", textColor: tuple=(255,0,0), font: int=cv2.FONT_HERSHEY_PLAIN, 
                        fontSize: int=1, textThickness: int=2, recThickness: int=-1, recColor: tuple=(0,0,255)) -> None:
    """
    Adds the text to the given point. It automaticlly adjust the size of the textbox.
    
    ### Parameters
        `img: cv2.Mat | np.ndarray`:
            img Image/Frame.
        `textLocaiton: tuple[int, int]`:
            Location of the text.
        `text: str`:
            Text to be shown.
        `textColor: tuple`:
            Color of the text.
        `font: int`:
            Font style of the text.
        `fontSize: int`:
            Size of the text.
        `textThickness: int`:
            Thickness of the text.
        `recThickness: int`:
            Thickness of the rectangle outside the text.
        `recColor: tuple`
            Color of the rectangle outside the text.
    
    ### Returns
        None.
    """

    size, _ = cv2.getTextSize(text, font, fontSize, textThickness)
    textWidth, textHeight = size

    cv2.rectangle(img, (max(0, textLocation[0][0]), max(0, textLocation[0][1])), 
                    (max(0, textLocation[0][0]+textWidth), max(40, textLocation[0][1]-(textHeight*2))), recColor, recThickness)

    cv2.putText(img, text, (textLocation[0][0], textLocation[0][1]-int(textHeight/2)), font, fontSize, textColor, textThickness)
    
def fps_counter(img: cv2.Mat | np.ndarray, cTime: float, pTime: float, fps_txt_pt: tuple=(20,30), txt_color: tuple=(255,0,0)) -> None:
    """
    Adds the fps counter to the screen.
    
    ### Parameters
        `img: cv2.Mat | np.ndarray`:
            Image/Frame.
        `cTime: float`:
            Current time.
        `pTime: float`:
            Previous time.
        `fps_txt_pt: tuple`:
            Location of the fps counter on the screen.
        `txt_color: tuple`:
            Color of the fps text.
        
    ### Returns
        None.
    """

    fps = int(1/(cTime-pTime))
    cv2.putText(img, f"FPS: {fps}", fps_txt_pt, cv2.FONT_HERSHEY_PLAIN, 2, txt_color, 2)

def show_camera_center(img: cv2.Mat | np.ndarray, color: tuple[int, int, int]=(255, 0, 0), line_thickness: int=2, circle_raidus: int=30) -> None:
    """
    Draws a cross and circle to center of the screen.
    
    ### Parameters
        `img: cv2.Mat | np.ndarray`:
            Image/Frame.
        `color: tuple[int, int, int]`:
            Color of the cross and circle.
        `line_thickness: int`:
            Thickness of the cross lines
        
    
    ### Returns
        Class names list with class colors.
    """

    height, width, channels = img.shape
    half_width = int(width / 2)
    half_heigth = int(height / 2)

    # Horizontal line
    cv2.line(img, (0, half_heigth), (width, half_heigth), color, line_thickness)

    # Vertical line
    cv2.line(img, (half_width, 0), (half_width, height), color, line_thickness)

    # Center circle
    cv2.circle(img, (half_width, half_heigth), circle_raidus, color, line_thickness)
