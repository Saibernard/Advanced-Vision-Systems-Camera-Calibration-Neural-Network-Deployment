import cv2
import numpy as np
from convert_trt import *
from yolo import *
from lane import *
import time 

# Initialize camera feed
cap = cv2.VideoCapture(4)

if not (cap.isOpened()):
    print("Could not open video device")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
cap.set(cv2.CAP_PROP_FPS, 60)

def get_distance(x, y):  # input pixel x,y
    Hmount = 13.649806624732923
    intrinsic = np.array(
        [[694.71543085, 0, 449.37540776], [0, 695.54961208, 258.64705743], [0, 0, 1]])  # From calibration

    x_car = intrinsic[1][1] * Hmount / (y - intrinsic[1][2])
    y_car = - x_car * (x - intrinsic[0][2]) / intrinsic[0][0]
    return x_car#, y_car


# ...Run Lane detection/object detection
engine, context = load_trt('./model/model_32.trt')
detect_lane = False

# Get frame
inference_cycle = 0
duration = 0.0
while True:
    ret, image = cap.read()
    if not ret:
        break

    # detect lane
    if detect_lane:
        contours = lane_detection(image)
        cv2.drawContours(image, contours, -1, (0, 255, 0), 3)

    # run inference
    image = image / 255.0
    start_time = time.perf_counter()
    output = inference(image, engine, context)
    end_time = time.perf_counter()
    if inference_cycle <= 100:
        duration += (end_time - start_time)

    inference_cycle += 1
    print(f'duration: {duration/100}')
    # draw bboxs
    for box in output:
        x1, y1 = 3 * int(box[0] - box[2] / 2), 3 * int(box[1] - box[3] / 2)
        x2, y2 = 3 * int(box[0] + box[2] / 2), 3 * int(box[1] + box[3] / 2)
        image = cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        x = (x1 + x2) / 2
        
        distance = get_distance(x, y2) #TODO
        print(f'We are getting a distance of {distance}')
        cv2.putText(
            image,
            str(round(distance, 2)) + "cm",
            (int(box[0]) * 3, int(box[1] + (box[3] / 2)) * 3),
            cv2.FONT_HERSHEY_COMPLEX,
            0.5,
            (0, 255, 0),
            2
        )

    cv2.imshow("Detection Output", image)

    k = cv2.waitKey(1)
    if k % 256 == 27:
        break

cap.release()
cv2.destroyAllWindows()
