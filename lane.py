import cv2
import numpy as np

def filter_yellow_boxes(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # lower and upper bounds for the yellow color
    lower_yellow = np.array([22, 40, 40])
    upper_yellow = np.array([50, 255, 255])

    # Creating a mask for the yellow color
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Applying the mask to the image
    yellow_boxes = cv2.bitwise_and(image, image, mask=mask)

    return yellow_boxes

def lane_detection(image):
    # Resize the image
    image = cv2.resize(image, (960, 540))

    # Filtering yellow boxes
    yellow_boxes = filter_yellow_boxes(image)

    # Convert the filtered image to grayscale
    gray_image = cv2.cvtColor(yellow_boxes, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    blurred_gray = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Apply threshold
    _, thresh = cv2.threshold(blurred_gray, 50, 255, cv2.THRESH_BINARY)

    # Applying morphological operations
    kernel = np.ones((2, 2), np.uint8)
    eroded = cv2.erode(thresh, kernel, iterations=2)
    dilated = cv2.dilate(eroded, kernel, iterations=2)

    # Finding contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filtering contours by area
    min_area = 45  # You can adjust this value based on your specific image
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

    # Drawing contours on the original image
    cv2.drawContours(image, filtered_contours, -1, (0, 255, 0), 2)

    return image


if __name__ == "__main__":
    # Reading the input image
    input_image = cv2.imread("Resources/lane.png")

    # Applying lane detection
    result = lane_detection(input_image)

    # Output
    cv2.imshow("Lane Detection", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
