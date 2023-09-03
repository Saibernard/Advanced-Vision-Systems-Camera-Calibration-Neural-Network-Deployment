# Import required modules
import cv2
import numpy as np
import os
import glob
  
def calibrate():
    # Define the dimensions of checkerboard
    CHECKERBOARD = (6, 8)
    
    # stop the iteration when specified
    # accuracy, epsilon, is reached or
    # specified number of iterations are completed.
    criteria = (cv2.TERM_CRITERIA_EPS + 
                cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    # Vector for 3D points
    threedpoints = []
    
    # Vector for 2D points
    twodpoints = []
    
    
    #  3D points real world coordinates
    objectp3d = np.zeros((1, CHECKERBOARD[0] 
                        * CHECKERBOARD[1], 
                        3), np.float32)
    objectp3d[0, :, :2] = np.mgrid[0:CHECKERBOARD[0],
                                0:CHECKERBOARD[1]].T.reshape(-1, 2)
    objectp3d = objectp3d * 25 / 8
    prev_img_shape = None
    
    
    # Extracting path of individual image stored
    # in a given directory. Since no path is
    # specified, it will take current directory
    # jpg files alone
    images = glob.glob('./calibration/*.png')
    
    for filename in images:
        image = cv2.imread(filename)
        grayColor = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
        # Find the chess board corners
        # If desired number of corners are
        # found in the image then ret = true
        ret, corners = cv2.findChessboardCorners(
                        grayColor, CHECKERBOARD, 
                        cv2.CALIB_CB_ADAPTIVE_THRESH 
                        + cv2.CALIB_CB_FAST_CHECK + 
                        cv2.CALIB_CB_NORMALIZE_IMAGE)
    
        # If desired number of corners can be detected then,
        # refine the pixel coordinates and display
        # them on the images of checker board
        if ret == True:
            threedpoints.append(objectp3d)
    
            # Refining pixel coordinates
            # for given 2d points.
            corners2 = cv2.cornerSubPix(
                grayColor, corners, (11, 11), (-1, -1), criteria)
    
            twodpoints.append(corners2)
    
            # Draw and display the corners
            image = cv2.drawChessboardCorners(image, 
                                            CHECKERBOARD, 
                                            corners2, ret)
    
        #cv2.imshow('img', image)
        #cv2.waitKey(0)
    
    #cv2.destroyAllWindows()
    
    h, w = image.shape[:2]
    
    
    # Perform camera calibration by
    # passing the value of above found out 3D points (threedpoints)
    # and its corresponding pixel coordinates of the
    # detected corners (twodpoints)
    ret, matrix, distortion, r_vecs, t_vecs = cv2.calibrateCamera(
        threedpoints, twodpoints, grayColor.shape[::-1], None, None)
    
    
    # Displaying required output
    print(" Camera matrix:")
    print(matrix)
    
    print("\n Distortion coefficient:")
    print(distortion)
    
    print("\n Rotation Vectors:")
    print(r_vecs)
    
    print("\n Translation Vectors:")
    print(t_vecs)

calibrate()

def click_event(event, x, y, flags, params): #Get the pixel number of the cone bottom right
  
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
  
        # displaying the coordinates
        # on the Shell
        print(x, ' ', y)
  
        # displaying the coordinates
        # on the image window
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, str(x) + ',' +
                    str(y), (x,y), font,
                    1, (255, 0, 0), 2)
        cv2.imshow('image', img)
  
    # checking for right mouse clicks     
    if event==cv2.EVENT_RBUTTONDOWN:
  
        # displaying the coordinates
        # on the Shell
        print(x, ' ', y)
  
        # displaying the coordinates
        # on the image window
        font = cv2.FONT_HERSHEY_SIMPLEX
        b = img[y, x, 0]
        g = img[y, x, 1]
        r = img[y, x, 2]
        cv2.putText(img, str(b) + ',' +
                    str(g) + ',' + str(r),
                    (x,y), font, 1,
                    (255, 255, 0), 2)
        cv2.imshow('image', img)

'''
# reading the image
img = cv2.imread('./resource/cone_x40cm.png', 1)
  
# displaying the image
cv2.imshow('image', img)
  
# setting mouse handler for the image
# and calling the click_event() function
cv2.setMouseCallback('image', click_event)
  
# wait for a key to be pressed to exit
cv2.waitKey(0)


# close the window
cv2.destroyAllWindows()

#664, 496'''

def calc_Hmount(x_40, y_40): #Assume zcar = 0
    intrinsic = np.array([[694.71543085, 0, 449.37540776], [0, 695.54961208, 258.64705743], [0, 0, 1]]) #From calibration

    return 40*(y_40 - intrinsic[1][2]) / intrinsic[1][1]

print(calc_Hmount(664, 496))

def get_distance(x, y):
    Hmount = 13.649806624732923
    intrinsic = np.array([[694.71543085, 0, 449.37540776], [0, 695.54961208, 258.64705743], [0, 0, 1]]) #From calibration

    x_car = intrinsic[1][1] * Hmount / (y - intrinsic[1][2])
    y_car = - x_car * (x - intrinsic[0][2]) / intrinsic[0][0]
    return x_car, y_car

'''
# reading the image
img = cv2.imread('./resource/cone_unknown.png', 1)
  
# displaying the image
cv2.imshow('image', img)
  
# setting mouse handler for the image
# and calling the click_event() function
cv2.setMouseCallback('image', click_event)
  
# wait for a key to be pressed to exit
cv2.waitKey(0)


# close the window
cv2.destroyAllWindows()

#596, 416'''

print(get_distance(596, 416))