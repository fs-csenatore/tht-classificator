import logging
import cv2
#import cv2.typing does not work in 4.7
import numpy as np
import time
from SettingsFile import xmlSettings

logging.basicConfig(level=logging.INFO)

def get_thresh_mask(frame, #: cv2.typing.MatLike,  #cv2.typing.MatLike does not work in opencv < 4.8
                    lowerBound,
                    upperBound,
                    frame_convert = cv2.COLOR_BGR2HSV,
                ):
    frame = cv2.cvtColor(frame, frame_convert)
    #Blur helps to remove nois in background
    frame = cv2.medianBlur(frame,5)
    mask = cv2.inRange(frame, lowerBound, upperBound)
    return mask

def get_masked_image(frame, #: cv2.typing.MatLike,
                     mask, ##: cv2.typing.MatLike,
                    ):
    img_masked = np.zeros_like(frame, np.uint8)
    imask = np.greater(mask,0)
    img_masked[imask] = frame[imask]
    return img_masked

#Sets Frame-Rate for outgoing stream
def get_frame(cap_gst, fps :int):
    fps_stream = int(round(cap_gst.get(cv2.CAP_PROP_FPS )))
    frames2skip = int(fps_stream / fps)
    logging.debug("frames2skip=%d",frames2skip)
    logging.debug("fps_stream=%d", fps_stream)
    while True:
        frameId = int(round(cap_gst.get(cv2.CAP_PROP_POS_FRAMES)))

        if frameId % frames2skip == 0:
            logging.debug("read frameId=%d",frameId)
            ret, frame  = cap_gst.read()
            return ret, frame
        else:
            logging.debug("drop frameId=%d",frameId)
            cap_gst.grab()

def init_gst():
    cap_gst = cv2.VideoCapture('v4l2src ! video/x-raw, width=1920, height=1080, framerate=5/1, format=YUY2 ! imxvideoconvert_pxp ! video/x-raw, format=BGR ! appsink',
                            cv2.CAP_GSTREAMER)

    wrt_gst = cv2.VideoWriter('appsrc ! video/x-raw, width=1920, height=1080, format=BGR ! videoconvert ! video/x-raw, format=BGRx ! fpsdisplaysink sync=false',
                               cv2.CAP_GSTREAMER,5,(1920,1080),True)

    if not cap_gst.isOpened():
        logging.error('VideoCapture not opened')
        exit(0)
    else:
        logging.debug('VideoCapture is open')

    if not wrt_gst.isOpened():
        logging.error('VideoWriter not opened')
        exit(0)
    else:
        logging.debug('VideoWriter is open')

    return cap_gst, wrt_gst

settings_xml = 'Settings.xml'
Settings = xmlSettings(settings_xml)

try:
    cap_gst, wrt_gst = init_gst()
    while True:
        start_time = time.time()
        Settings.parse(settings_xml)
        thresh_lowerBound, thresh_upperBound = Settings.read_hsv_boundings()
        ret, cap_frame = get_frame(cap_gst,5)
        if not ret:
            logging.error('stream broken')
            break

        #Remove Background
        resized_frame = cv2.resize(cap_frame, (640, 360),interpolation=cv2.INTER_LINEAR)
        mask = get_thresh_mask(resized_frame,thresh_lowerBound,thresh_upperBound)
        resized_frame = get_masked_image(resized_frame, mask)
        
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt = max(contours, key=cv2.contourArea)
        cv2.drawContours(resized_frame, [cnt], 0, (0,255,0),2)
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.intp(box)
        cv2.drawContours(resized_frame, [box], 0, (255,255,0),3)

        #Write Frame
        cap_frame[0:360, 0:640,:] = resized_frame[:,:,:]
        wrt_gst.write(cap_frame)
        end_time = time.time()
        logging.debug("Time in ms =%f",(end_time-start_time)*10**3)

except KeyboardInterrupt:
    cap_gst.release()
    wrt_gst.release()
    Settings.write()
else:
    cap_gst.release()
    wrt_gst.release()