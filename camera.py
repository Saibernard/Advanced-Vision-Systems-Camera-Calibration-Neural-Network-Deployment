import cv2

# Open the device at the ID 2
cap = cv2.VideoCapture(2)

if not (cap.isOpened()):
    print("Could not open video device")

#Set the resolution and frame rate
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
cap.set(cv2.CAP_PROP_FPS, 60) 

# Capture frame-by-frame
while(True):
    ret, frame = cap.read()

    # Display the resulting frame
    cv2.imshow("preview",frame)
    
    #Waits for a user input to quit the application
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()