"""
The purpose is that an object with certain color spectra
(e.g. the green of a PCB) is detected in the image of the camera.
The object should always be cut out and aligned.
"""

import logging
import cv2
#import cv2.typing does not work in 4.7
import numpy as np
from THTClassificator.SettingsFile import FrameSettings
import pickle

class FrameProccessing():
    def __init__(self):
        self.Settings = FrameSettings()
        self.__check_Settings()
        self.__init_gst()
        wrt_frame_res = self.Settings.get_streamwrite_resolution()
        self.wrt_frame = np.zeros((wrt_frame_res[1], wrt_frame_res[0], 3), np.uint8)


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
        #fps_stream is not the real FPS comming from source.
        #It is the fixed value comming from v4l2-ctl --list-formats-ext
        fps_stream = int(round(self.__cap_gst.get(cv2.CAP_PROP_FPS)))
        frames2skip = round(fps_stream / fps)
        while True:
            frameId = int(round(self.__cap_gst.get(cv2.CAP_PROP_POS_FRAMES)))

            if frameId % frames2skip == 0:
                logging.debug("read frameId=%d",frameId)
                ret, self.cap_frame  = self.__cap_gst.read()
                return ret
            else:
                logging.debug("drop frameId=%d",frameId)
                self.__cap_gst.grab()


    def __streamwrite(self):
        """
        __streamwrite
        write frame to gstreamer
        """
        if not self.Settings.is_streamwrite_colored():
            self.wrt_frame = cv2.cvtColor(self.wrt_frame,
                                               cv2.COLOR_BGR2GRAY)
        self.__wrt_gst.write(self.wrt_frame)


    def __streamcap(self):
        """
        __streamcap
        get frame from gstreamer
        """
        if not self.__get_frame():
            logging.error('stream is broken')
            assert False, "can't get new frame. Stream is broken"

        #Due to distortion, and performance issues the image is limited to the center area.
        # Therefore, the image is cropped to 4:3 format (25% less data).
        self.cap_frame = self.cap_frame[:,int(self.cap_frame.shape[1]*0.125):int(self.cap_frame.shape[1]*0.875),:]

        #undistort Image
        if self.Settings.is_distortion_enabled() and \
                self.Settings.get_distortion_file() != None:
            
            dist_data = self.__get_distortion_data()
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(dist_data[0], dist_data[1], self.cap_frame.shape[1::-1], 1, self.cap_frame.shape[1::-1])
            self.cap_frame = cv2.undistort(self.cap_frame, dist_data[0], dist_data[1], None, newcameramtx)


    def __get_threshhold_mask(
            self,
            frame, #: cv2.typing.MatLike,  #cv2.typing.MatLike does not work in opencv < 4.8
            lowerBound :tuple,
            upperBound :tuple,
            frame_convert = cv2.COLOR_BGR2HSV,
    ):
        """
        determines which color spectrum is to be considered and creates
        a mask over the desired colors.
        """
        frame = cv2.cvtColor(frame, frame_convert)
        #Blur helps to remove nois in background
        frame = cv2.blur(frame, (20,20))
        mask = cv2.inRange(frame, lowerBound, upperBound)
        return mask


    def __get_masked_image(self,
            frame, #: cv2.typing.MatLike,
            mask,  #: cv2.typing.MatLike,
        ):
        """
        Returns only the color information that is covered by a mask.
        """
        img_masked = np.zeros_like(frame, np.uint8)
        imask = np.greater(mask,0)
        img_masked[imask] = frame[imask]
        return img_masked


    def __create_working_frame(self):
        return cv2.resize(self.cap_frame,
                          self.__get_working_framesize(),
                          interpolation=cv2.INTER_NEAREST)


    # like cv2.minAreaRect(cnt) but it includes a litle offset
    def __get_minAreaRect(self, cnt):

        #Rect is a little oversized to get full PCB-border
        scale_factor = 1.1

        rect = cv2.minAreaRect(cnt)
        resized_rect = list(rect)
        resized_rect[1] =  (int(rect[1][0] * scale_factor),
                             int(rect[1][1] * scale_factor))
        return tuple(resized_rect)


    def __get_working_framesize(self):
        '''
        Get a framesize to work with.
        it scalse the captured frame down with a ratio of 1/3 per axis
        '''
        working_frame_res = list(
            self.Settings.get_streamcap_resolution())

        working_frame_res[0] = int(working_frame_res[0]*0.75/3)
        working_frame_res[1] = int(working_frame_res[1]/3)
        return tuple(working_frame_res)


    def __scale_minAreaRect(self, rect):
        '''
        Scale minareaRect from Working frame size to capture frame size
        '''
        tmp_rect = list(rect)
        working_frame_res = self.Settings.get_streamcap_resolution()
        if working_frame_res[0] == 1920 or \
            working_frame_res[0] == 1280:
            tmp_rect[1] = (rect[1][0]*3, rect[1][1]*3)
            tmp_rect[0] = (rect[0][0]*3, rect[0][1]*3)

        return tuple(tmp_rect)


    def __rotate_rect(self, rect, bounded_object):
        """
        Rotate bounding-box-img and minarearect, so that x:y where x>y
        """
        tmp_rect = list(rect)
        if rect[1][1]<rect[1][0] and rect[2] > 45:
            bounded_object = cv2.rotate(bounded_object, cv2.ROTATE_90_COUNTERCLOCKWISE)
            tmp_rect[2] = rect[2] - 90

        elif rect[1][0]<rect[1][1] and rect[2] < 45:
            bounded_object = cv2.rotate(bounded_object, cv2.ROTATE_90_COUNTERCLOCKWISE)

        elif rect[1][0]<rect[1][1] and rect[2] > 45:
            tmp_rect[2] = rect[2] - 90
            tmp_rect[1] = (rect[1][1], rect[1][0])

        if rect[1][0]<rect[1][1]:
            tmp_rect[1] = (rect[1][1], rect[1][0])

        return tuple(tmp_rect), bounded_object


    #returnvalue is a tuple (mtx, dist, rvecs, tvecs)
    def __get_distortion_data(self):
        with open(self.Settings.get_distortion_file(), 'rb') as file:
            data = pickle.load(file)

        return data


    def __do_preprocess(self):
        #Get a Frame to work with
        self.working_frame = self.__create_working_frame()
        mask = self.__get_threshhold_mask(self.working_frame,
                                      self.Settings.get_hsv_boundings()[0],
                                      self.Settings.get_hsv_boundings()[1])
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                                cv2.CHAIN_APPROX_SIMPLE)

        if hasattr(self,'bounded_object'):
            del self.bounded_object
        if hasattr(self,'object_img'):
            del self.object_img
        if hasattr(self,'bounded_object'):
            del self.bounded_object
        if hasattr(self,'object_img'):
            del self.object_img

        #TODO: Support more then one Board
        if contours:
            #create Boxing
            cnt = max(contours, key=cv2.contourArea)
            rect = self.__get_minAreaRect(cnt)
            scaled_rect = self.__scale_minAreaRect(rect)
            scaled_box = cv2.boxPoints(scaled_rect)
            scaled_box = np.intp(scaled_box)
            x, y, w, h = cv2.boundingRect(scaled_box)

            #Get bounding Object
            offset = 10

            #is the board completly in frame
            if max(0,y-offset) == 0 or \
                min(y+h+offset,self.cap_frame.shape[0]) == self.cap_frame.shape[0] or \
                max(0,x-offset) == 0 or min(x+w+offset,self.cap_frame.shape[1]) == self.cap_frame.shape[1]:
                logging.debug("object is truncated")
                pass

            #get board within bounding box
            self.bounded_object = self.cap_frame[
                max(0,y-offset):min(y+h+offset,self.cap_frame.shape[0]),
                max(0,x-offset):min(x+w+offset,self.cap_frame.shape[1]),
                :]

            #Rotate board
            if hasattr(self.bounded_object, 'shape') and self.bounded_object.shape[0] > 0 and self.bounded_object.shape[1] > 0:
                scaled_rect, self.bounded_object = self.__rotate_rect(scaled_rect, self.bounded_object)

                #assume. that the center of bbox is same as in minAreaRect
                object_center = tuple(i/2 for i in self.bounded_object.shape[0:2])
                rot_mat = cv2.getRotationMatrix2D(object_center, scaled_rect[2],1)
                object_img = cv2.warpAffine(self.bounded_object, rot_mat, self.bounded_object.shape[1::-1], flags=cv2.INTER_LINEAR)
                #Cut Board from rotated image
                object_img = object_img[int(object_center[0]-(scaled_rect[1][1]/2)):int(object_center[0]+(scaled_rect[1][1]/2)),
                           int(object_center[1]-(scaled_rect[1][0]/2)):int(object_center[1]+(scaled_rect[1][0]/2))]

                #Format image with 1:1. The Image size is variable
                if object_img.shape[0] > 100 and object_img.shape[1] > 100:
                    max_size = max(object_img.shape[0], object_img.shape[1])
                    self.object_img = np.zeros((max_size, max_size, 3),np.uint8)
                    self.object_img[0:object_img.shape[0],0:object_img.shape[1],:] = object_img
                    self.object_img = cv2.resize(self.object_img,(640,640), interpolation=cv2.INTER_LINEAR_EXACT)

        #when debug is enabled
        if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
            self.calibrate_frame = self.__get_masked_image(self.working_frame, mask)
            if 'cnt' in locals():
                cv2.drawContours(self.calibrate_frame, [cnt], 0, (0,255,0),2)
                box = np.int0(cv2.boxPoints(rect))
                cv2.drawContours(self.calibrate_frame, [box], 0, (255,255,0),3)
                x, y, w, h = cv2.boundingRect(box)


    def read_frame(self):
        self.__streamcap()


    def write_frame(self):
        self.__streamwrite()


    def update(self):
        """
        update
        write outgoing frame, get incoming frame and run preprocessing
        """
        self.__streamwrite()
        self.__streamcap()
        self.__do_preprocess()

    def __del__(self):
        self.__cap_gst.release()
        self.__wrt_gst.release()
