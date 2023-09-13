import logging
import cv2
#import cv2.typing does not work in 4.7
import numpy as np
import time


logging.basicConfig(level=logging.DEBUG)

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

#Sets Frame-Rate for outgoing streem
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

    wrt_gst = cv2.VideoWriter('appsrc ! video/x-raw, width=640, height=360, format=GRAY8 ! imxvideoconvert_pxp ! video/x-raw, format=BGRx ! fpsdisplaysink',
                               cv2.CAP_GSTREAMER,5,(640,360),False)

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

thresh_lowerBound=np.array([10,20,25]) #10
thresh_upperBound=np.array([90,255,255]) #90

cap_gst, wrt_gst = init_gst()

#outgoing_frame=np.zeros([[1080],[1920],[3]])

while True:
    ret, cap_frame = get_frame(cap_gst,5)
    if not ret:
        logging.error('stream broken')
        break
    start_time = time.time()
    resized_frame = cv2.resize(cap_frame, (640, 360),interpolation=cv2.INTER_NEAREST)
    mask = get_thresh_mask(resized_frame,thresh_lowerBound,thresh_upperBound)
    resized_frame = get_masked_image(resized_frame, mask)
    resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
    end_time = time.time()
    wrt_gst.write(resized_frame)
    logging.debug("Time in ms =%f",(end_time-start_time)*10**3)


cap_gst.release()
wrt_gst.release()
