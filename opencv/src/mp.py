import logging
import multiprocessing as mp
from multiprocessing import shared_memory
from queue import Empty
import numpy as np
from FSBoard import MED3_rev100, Boards
from FrameProcessing import FrameProccessing
from SettingsFile import xmlSettings
import cv2
from queue import Empty
from dataclasses import dataclass

#Process Signals  
class STOPFLAG(): pass
class SAVEVOC(): pass
class ACK(): pass
class PUT(): pass

#SSHKEYBOARD
@dataclass
class Keyboard:
    f5: bool
    f6: bool
    f7: bool

def process_classification(queue: mp.Queue, shm_name: str, lock: mp.Lock, board: Boards, log_level: int):
    shm = shared_memory.SharedMemory(name=shm_name)
    img_buf = np.ndarray((320,320,3), dtype="uint8", buffer=shm.buf)
    logging.basicConfig(level=log_level)

    img_buf[:,:,:] = np.ones((320,320,3))*255

    while True:
        try:
            signal = queue.get()
        except Empty:
            logging.error("Close Classification process")
            break

        if isinstance(signal, STOPFLAG):
            logging.debug("got stop signal\n Exit Classification process")
            break
        
        img = np.zeros((320,320,3), dtype=np.uint8)
        with lock:
            img[:,:,:] = img_buf[:,:,:]
        
        match board:
            case Boards.MED3_REV100:
                currentBoard = MED3_rev100(img)

        if isinstance(signal, SAVEVOC):
            logging.debug("create Image")
            currentBoard.save_voc('/home/weston/dataset_train')

def process_preprocess(settings_path: str, queue: mp.Queue, shm_name: str, lock: mp.Lock, log_level: int):
    shm = shared_memory.SharedMemory(name=shm_name)
    img_buf = np.ndarray((320,320,3), dtype="uint8", buffer=shm.buf)
    frame_index = 0

    #Start GStreamer
    Settings= xmlSettings(settings_path)
    img_processing = FrameProccessing(Settings)

    while True:
        img_processing.update()

        if hasattr(img_processing, "object_img"):
            frame_index = frame_index + 1

            #first of all, use only every 5th object
            if frame_index % 5 == 0:
                with lock:
                    img_buf[:,:,:] = img_processing.object_img[:,:,:]
                
                queue.put(PUT())

        #Draw Rectangle around object
        if hasattr(img_processing, 'scaled_box'):
            x, y, w, h = cv2.boundingRect(img_processing.scaled_box)
            cv2.drawContours(img_processing.cap_frame, [img_processing.scaled_box], 0, (255,255,0),2)
            cv2.rectangle(img_processing.cap_frame, (x, y), (x + w, y + h), (0,0,255), 4)

        #Show Captured Frame
        img_processing.wrt_frame[:,
            img_processing.wrt_frame.shape[1]-img_processing.cap_frame.shape[1]:,
            :] = img_processing.cap_frame

        #when debug is enabled
        if log_level == logging.DEBUG:

            #Show Calibration Frame
            if hasattr(img_processing, 'calibrate_frame'):
                try:
                    shape1 = img_processing.calibrate_frame.shape
                    img_processing.wrt_frame[0:shape1[0], 0:shape1[1], :] = img_processing.calibrate_frame
                except:
                    logging.error("Could not display calibrate_frame")

            #Show rotated object
            if hasattr(img_processing, 'object_img'):
                try:
                    shape2 = img_processing.object_img.shape
                    board = MED3_rev100(img_processing.object_img)
                    board.draw_boundingbox()
                    img_processing.wrt_frame[0:shape2[0], img_processing.wrt_frame.shape[1]-shape2[1]:, :] = board.image
                except:
                    logging.error("Could not display object_img")

            img_processing.Settings.parse(settings_path)