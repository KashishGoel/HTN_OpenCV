# import the necessary packages
import argparse
import datetime
import imutils
import time
import cv2
import numpy as np
from collections import deque
from firebase import firebase
# Include the Dropbox SDK
import dropbox
import tempfile
import base64
import thread
import time
from socketIO_client import SocketIO, LoggingNamespace
import socket


dropbox_key = 'insert_your_key'
dropbox_secret = 'insert_your_secret'
f = open('dropbox_access_token', 'r')
access_token = f.read();
client = dropbox.client.DropboxClient(access_token)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
ap.add_argument("-a", "--min-area", type=int, default=1000, help="minimum area size")
args = vars(ap.parse_args())

# if the video argument is None, then we are reading from webcam
if args.get("video", None) is None:
    camera = cv2.VideoCapture(0)
    time.sleep(0.25)
 
# otherwise, we are reading from a video file
else:
    camera = cv2.VideoCapture(args["video"])
 
# initialize the first frame in the video stream
avg = None
deque_max_size = pow(2, 2)
pts = deque(maxlen=deque_max_size)
firebase_start = 0
firebase_update_rate = 20
firebase = firebase.FirebaseApplication('https://motion.firebaseio.com', None)
avg_people = 0
def send_data_to_firebase(frm):
    frame_small = imutils.resize(frm, width=300)
    cnt = cv2.imencode('.jpeg',frame_small)[1]
    b64 = base64.encodestring(cnt)
    result = firebase.patch('/', {'frame': b64})
   

def send_people_to_firebase(people):
    result = firebase.patch('/', {'people': people})

# loop over the frames of the video
while True:
    # grab the current frame and initialize the occupied/unoccupied
    # text
    (grabbed, frame) = camera.read()
    frame = np.fliplr(frame).copy()
    text = "Unoccupied"

    #upload to the dropbox
    # cv2.imwrite('temp_img.png', frame)
    # f = open("temp_img.png", 'rb')
    # temp_img = tempfile.TemporaryFile()
    # print temp_img.name
    # response = client.put_file('/' + datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p") + '.png', f)
    # print "uploaded:", response


    # if the frame could not be grabbed, then we have reached the end
    # of the video
    if not grabbed:
        break
 
    # resize the frame, convert it to grayscale, and blur it
    """
    frame_small = imutils.resize(frame, width=500)
    cnt = cv2.imencode('.jpeg',frame_small)[1]
    b64 = base64.encodestring(cnt)
    #result = firebase.patch('/', {'frame': b64})
    """
    """
    video_buffer.append(b64)

    # safety check
    if len(video_buffer) > 128:
        video_buffer = []
    """
    #frame_small = frame;
    """
    if firebase_start % firebase_update_rate == 0:
        firebase_start += 1
        cnt = cv2.imencode('.jpeg',frame_small)[1]
        b64 = base64.encodestring(cnt)
        result = firebase.patch('/', {'frame': b64})
    """

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    
    # if the average frame is None, initialize it
    if avg is None:
        print "[INFO] starting background model..."
        avg = gray.copy().astype("float")
        #rawCapture.truncate(0)
        continue

    # accumulate the weighted average between the current frame and
    # previous frames, then compute the difference between the current
    # frame and running average
    cv2.accumulateWeighted(gray, avg, 0.5)

    # compute the absolute difference between the current frame and
    # first frame
    frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(avg))

    thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
 
    # dilate the thresholded image to fill in holes, then find contours
    # on thresholded image
    thresh = cv2.dilate(thresh, None, iterations=2)
    (cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    
    # keep track of main contours
    main_cnts = 0;

    # loop over the contours
    for c in cnts:
        # if the contour is too small, ignore it
        if cv2.contourArea(c) < args["min_area"]:
            continue
 
        main_cnts += 1;
        # compute the bounding box for the contour, draw it on the frame,
        # and update the text
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text = "Occupied"
    

    # Update number of poeple in firebase
    # print main_cnts
    avg_people = .5 * avg_people + .5 * main_cnts
    thread.start_new_thread(send_people_to_firebase, (avg_people,))     

    center = None;
    # only proceed if at least one contour was found
    
    if len(cnts) > 0:
        # find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle and
        # centroid
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
 
        # only proceed if the radius meets a minimum size
        if radius > 10:
            # draw the circle and centroid on the frame,
            # then update the list of tracked points
            cv2.circle(frame, (int(x), int(y)), int(radius),
                (0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)
    
    #Using socket instead of firebase
    #thread.start_new_thread(send_data_to_firebase, (frame,))
    """"""""""""""""""""""""
    frm = frame
    frame_small = imutils.resize(frm, width=100)
    cnt = cv2.imencode('.jpeg',frm)[1]
    b64 = base64.encodestring(cnt)
    s = socket.socket(
            socket.AF_INET, socket.SOCK_STREAM)

    ip = "insert_your_ip";
    port = 58555
    s.connect((ip, port))
    print len(b64)
    newb = b64.ljust(500000)
    print len(newb)
    s.sendall(newb)
    """"""""""""""""""""""""
    break;


    # update the points queue
    pts.appendleft(center)


    # loop over the set of tracked points
    for i in xrange(1, len(pts)):
        # if either of the tracked points are None, ignore
        # them
        if pts[i - 1] is None or pts[i] is None:
            continue
 
        # otherwise, compute the thickness of the line and
        # draw the connecting lines
        thickness = int(np.sqrt(deque_max_size / float(i + 1)) * 2.5)
        cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)
 


    # draw the text and timestamp on the frame
    cv2.putText(frame, "Room Status: {}".format(text), (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
        (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
    cv2.putText(frame, str(main_cnts),
        (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 10, (0, 0, 255), 4)
    
 
    # show the frame and record if the user presses a key
    
    cv2.imshow("Security Feed", frame)
    cv2.imshow("Thresh", thresh)
    cv2.imshow("Frame Delta", frameDelta)

    key = cv2.waitKey(1) & 0xFF
    
    # if the `q` key is pressed, break from the lop
    if key == 27 or key == ord('q'): # escape
        break

 
# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()

