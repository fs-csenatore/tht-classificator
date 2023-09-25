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
    assert fps_stream % fps == 0, "Outgoing frame rate must be an integer divisor"
    frames2skip = int(fps_stream / fps)
    while True:
        frameId = int(round(cap_gst.get(cv2.CAP_PROP_POS_FRAMES)))

        if frameId % frames2skip == 0:
            logging.debug("read frameId=%d",frameId)
            ret, frame  = cap_gst.read()
            return ret, frame
        else:
            logging.debug("drop frameId=%d",frameId)
            cap_gst.grab()

def init_gst(Settings :xmlSettings):
    cap_gst = cv2.VideoCapture('v4l2src ! video/x-raw, width=1920, height=1080, framerate=5/1, format=YUY2 ! imxvideoconvert_pxp ! video/x-raw, format=BGR ! appsink',
                            cv2.CAP_GSTREAMER)

    wrt_gst = cv2.VideoWriter('appsrc ! video/x-raw, width=1920, height=1080, format=GRAY8 ! imxvideoconvert_pxp ! video/x-raw, format=BGRx ! fpsdisplaysink sync=false',
                               cv2.CAP_GSTREAMER,
                               Settings.get_outgoing_framerate(),
                               (1920,1080),
                               False)
    
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

img_path="/home/weston/test"
img_cnt :int =0

cap_gst, wrt_gst = init_gst(Settings)
while True:
    try:
        start_time = time.time()
        Settings.parse(settings_xml)
        thresh_lowerBound, thresh_upperBound = Settings.read_hsv_boundings()
        ret, cap_frame = get_frame(cap_gst,Settings.get_outgoing_framerate())
        if not ret:
            logging.error('stream broken')
            break

        #Remove Background
        resized_frame = cv2.resize(cap_frame, (640, 360),interpolation=cv2.INTER_LINEAR)
        mask = get_thresh_mask(resized_frame,thresh_lowerBound,thresh_upperBound)
        resized_frame = get_masked_image(resized_frame, mask)
        
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cap_frame = cv2.cvtColor(cap_frame,cv2.COLOR_BGR2GRAY)
        if contours:
            #Box for preview_image
            cnt = max(contours, key=cv2.contourArea)
            cv2.drawContours(resized_frame, [cnt], 0, (0,255,0),2)
            rect = cv2.minAreaRect(cnt)
            resized_rect = list(rect)
            resized_rect[1] =  (rect[1][0] * 1.1, rect[1][1] * 1.1)
            rect = tuple(resized_rect)
            box = cv2.boxPoints(rect)
            box = np.intp(box)
            cv2.drawContours(resized_frame, [box], 0, (255,255,0),3)

            #Mask for cap_frame
            mask = np.zeros_like(mask)
            cv2.drawContours(mask, [box],0, (255),cv2.FILLED)
            mask = cv2.resize(mask,(1920, 1080), cv2.INTER_LINEAR)
            imask = np.greater(mask, 0)
            masked_frame = np.zeros_like(cap_frame)
            masked_frame[imask] = cap_frame[imask]

        #rotate rect, so that x:y where x>y
        if rect[1][0]<rect[1][1]:
            angle_rect = list(rect)
            angle_rect[2] = rect[2] - 90
            rect = tuple(angle_rect)
            rect_scaled_size = (rect[1][1]*3, rect[1][0]*3)
        else:
            rect_scaled_size = (rect[1][0]*3, rect[1][1]*3)

        rect_scaled_center = (rect[0][0]*3, rect[0][1]*3)

        #Rotate masked_frame, so that the board is orientated
        rot_mat = cv2.getRotationMatrix2D(rect_scaled_center, rect[2],1)
        masked_frame = cv2.warpAffine(masked_frame, rot_mat, masked_frame.shape[1::-1], flags=cv2.INTER_LINEAR)
        pcb = masked_frame[int(rect_scaled_center[1]-(rect_scaled_size[1]/2)):int(rect_scaled_center[1]+(rect_scaled_size[1]/2)),
                           int(rect_scaled_center[0]-(rect_scaled_size[0]/2)):int(rect_scaled_center[0]+(rect_scaled_size[0]/2))]
        #Write Frame
        masked_frame[0:360, 0:640] = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
        wrt_gst.write(masked_frame)
        end_time = time.time()
        logging.debug("Time in ms =%f",(end_time-start_time)*10**3)

    #TODO: Use another key to create a picture
    except KeyboardInterrupt:
        Settings.write(settings_xml)
        file=str(img_path)+str(img_cnt)+".png"
        cv2.imwrite(file,pcb)
        img_cnt = img_cnt+1
