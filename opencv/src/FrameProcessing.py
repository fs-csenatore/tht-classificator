import logging
import cv2
#import cv2.typing does not work in 4.7
import numpy as np
from SettingsFile import xmlSettings

class FrameProccessing():
    def __init__(self, Settings: xmlSettings):
        self.Settings = Settings
        self.__check_Settings()
        self.__init_gst()
        wrt_frame_res = self.Settings.get_streamwrite_resolution()
        self.wrt_frame = np.uint8(np.zeros((wrt_frame_res[1], wrt_frame_res[0], 3)))

    #Currently only FullHD or HD as Input resolution is supported
    def __check_Settings(self):
        cap_res = self.Settings.get_streamcap_resolution()
        assert cap_res == (1920, 1080) or \
            cap_res == (1280, 720), \
            "Incomming Frame Resolution is not supported"

    def __init_gst(self):
        logging.debug(self.Settings.get_streamcap_gstreamer_string())
        logging.debug(self.Settings.get_streamwrite_gstreamer_string())
        cap_gst = cv2.VideoCapture(
            self.Settings.get_streamcap_gstreamer_string(),
            cv2.CAP_GSTREAMER)

        wrt_gst = cv2.VideoWriter(
            self.Settings.get_streamwrite_gstreamer_string(),
            cv2.CAP_GSTREAMER,
            self.Settings.get_streamwrite_framerate(),
            self.Settings.get_streamwrite_resolution(),
            self.Settings.is_streamwrite_colored())

        assert cap_gst.isOpened(),'VideoCapture not opened'
        logging.debug('VideoCapture is open')

        assert wrt_gst.isOpened(), 'VideoWriter not opened'
        logging.debug('VideoWriter is open')

        self.__cap_gst = cap_gst
        self.__wrt_gst = wrt_gst

    #Sets Frame-Rate for outgoing stream
    def __get_frame(self):
        fps = self.Settings.get_streamwrite_framerate()
        fps_stream = int(round(self.__cap_gst.get(cv2.CAP_PROP_FPS)))
        assert fps_stream % fps == 0, "Outgoing frame rate must be an integer divisor"
        frames2skip = int(fps_stream / fps)
        while True:
            frameId = int(round(self.__cap_gst.get(cv2.CAP_PROP_POS_FRAMES)))

            if frameId % frames2skip == 0:
                logging.debug("read frameId=%d",frameId)
                ret, self.cap_frame  = self.__cap_gst.read()
                return ret
            else:
                logging.debug("drop frameId=%d",frameId)
                self.__cap_gst.grab()

    """
    __streamwrite
    write frame to gstreamer
    """
    def __streamwrite(self):
        if not self.Settings.is_streamwrite_colored():
            self.wrt_frame = cv2.cvtColor(self.wrt_frame,
                                               cv2.COLOR_BGR2GRAY)
        self.__wrt_gst.write(self.wrt_frame)

    """
    __streamcap
    get frame from gstreamer
    """
    def __streamcap(self):
        if not self.__get_frame():
            logging.error('stream is broken')
            assert False, "can't get new frame. Stream is broken"

    """
    get_thresh_mask
    determines which color spectrum is to be considered and creates
    a mask over the desired colors.
    """
    def __get_threshhold_mask(
            self,
            frame, #: cv2.typing.MatLike,  #cv2.typing.MatLike does not work in opencv < 4.8
            lowerBound :tuple,
            upperBound :tuple,
            frame_convert = cv2.COLOR_BGR2HSV,
    ):
        frame = cv2.cvtColor(frame, frame_convert)
        #Blur helps to remove nois in background
        frame = cv2.medianBlur(frame,5)
        mask = cv2.inRange(frame, lowerBound, upperBound)
        return mask

    """
    get_masked_image
    Returns only the color information that is covered by a mask.
    """
    def __get_masked_image(self,
            frame, #: cv2.typing.MatLike,
            mask,  #: cv2.typing.MatLike,
        ):
        img_masked = np.zeros_like(frame, np.uint8)
        imask = np.greater(mask,0)
        img_masked[imask] = frame[imask]
        return img_masked

    def __create_working_frame(self):
        return cv2.resize(self.cap_frame,
                          self.__get_working_framesize(),
                          interpolation=cv2.INTER_LINEAR)

    # like cv2.minAreaRect(cnt) but it includes a litle offset
    def __get_minAreaRect(self, cnt):
        rect = cv2.minAreaRect(cnt)
        resized_rect = list(rect)
        resized_rect[1] =  (rect[1][0] * 1.1, rect[1][1] * 1.1)
        return tuple(resized_rect)

    def __get_working_framesize(self):
        working_frame_res = list(
            self.Settings.get_streamcap_resolution())

        if working_frame_res[0] == 1920:
            working_frame_res[0] = int(working_frame_res[0]/3)
            working_frame_res[1] = int(working_frame_res[1]/3)
        else:
            working_frame_res[0] = int(working_frame_res[0]/2)
            working_frame_res[1] = int(working_frame_res[1]/2)
        return tuple(working_frame_res)

    def __scale_minAreaRect(self, rect):
        tmp_rect = list(rect)
        if self.Settings.get_streamcap_resolution()[0] == 1920:
            tmp_rect[1] = (rect[1][0]*3, rect[1][1]*3)
            tmp_rect[0] = (rect[0][0]*3, rect[0][1]*3)
        else:
            tmp_rect[1] = (rect[1][0]*2, rect[1][1]*2)
            tmp_rect[0] = (rect[0][0]*2, rect[0][1]*2)

        return tuple(tmp_rect)

    def __rotate_minAreaRect(self, rect):
        tmp_rect = list(rect)
        #rotate rect, so that x:y where x>y
        if rect[1][0]<rect[1][1]:
            tmp_rect[2] = rect[2] - 90
            tmp_rect[1] = (rect[1][1], rect[1][0])
        else:
            tmp_rect[1] = (rect[1][0], rect[1][1])
        return tuple(tmp_rect)

    def __do_preprocess(self):
        #Get a Frame to work with
        self.working_frame = self.__create_working_frame()
        mask = self.__get_threshhold_mask(self.working_frame,
                                      self.Settings.get_hsv_boundings()[0],
                                      self.Settings.get_hsv_boundings()[1])
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                                cv2.CHAIN_APPROX_SIMPLE)

        #TODO: Support more then one Board
        if contours:
            #create Boxing
            cnt = max(contours, key=cv2.contourArea)
            rect = self.__get_minAreaRect(cnt)
            scaled_rect = self.__scale_minAreaRect(rect)
            scaled_box = cv2.boxPoints(scaled_rect)
            scaled_box = np.intp(scaled_box)
            self.scaled_box = scaled_box
            x, y, w, h = cv2.boundingRect(scaled_box)

            #Get bounding Object
            self.bounded_object = self.cap_frame[
                y:y+h,
                x:x+w,
                :].copy()
            scaled_rect = self.__rotate_minAreaRect(scaled_rect)

            #assume. that the center of bbox is same as in minAreaRect
            object_center = tuple(i/2 for i in self.bounded_object.shape[0:2])
            rot_mat = cv2.getRotationMatrix2D(object_center, scaled_rect[2],1)
            self.object_img = cv2.warpAffine(self.bounded_object, rot_mat, self.bounded_object.shape[1::-1], flags=cv2.INTER_LINEAR)

            #TODO: Board cutout

            #when debug is enabled
            if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
                self.calibrate_frame = self.__get_masked_image(self.working_frame, mask)
                cv2.drawContours(self.calibrate_frame, [cnt], 0, (0,255,0),2)
                box = np.int0(cv2.boxPoints(rect))
                cv2.drawContours(self.calibrate_frame, [box], 0, (255,255,0),3)
                x, y, w, h = cv2.boundingRect(box)
                cv2.rectangle(self.cap_frame, (x, y), (x + w, y + h), (255,0,255), 8)

        else:
            return

    def read_frame(self):
        self.__streamcap()

    def write_frame(self):
        self.__streamwrite()

    """
    update
    write outgoing frame, get incoming frame and run preprocessing
    """
    def update(self):
        self.__streamwrite()
        self.__streamcap()
        self.__do_preprocess()

    def __del__(self):
        self.__cap_gst.release()
        self.__wrt_gst.release()