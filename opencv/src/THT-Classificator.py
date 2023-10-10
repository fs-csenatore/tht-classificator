import os
import threading
import logging
import numpy as np
import cv2
import time
import argparse
from SettingsFile import xmlSettings
from FrameProcessing import FrameProccessing
from sshkeyboard import listen_keyboard
from dataclasses import dataclass
import multiprocessing as mp
from multiprocessing import shared_memory
from FSBoard import MED3_rev100
import enum
from queue import Empty

class Boards(enum.Enum):
    MED3_REV100 = 1

@dataclass
class Keyboard:
    f5: bool
    f6: bool
    f7: bool

key_pressed = Keyboard(False, False, False)

def press(key):
    if key=='f5':
        key_pressed.f5 = True
    elif key=='f6':
        key_pressed.f6 = True
    elif key=='f7':
        key_pressed.f7 = True

def release(key):
    if key=='f5':
        key_pressed.f5 = False
    elif key=='f6':
        key_pressed.f6 = False
    elif key=='f7':
        key_pressed.f7 = False

def listen_keyboard_wrapper():
    listen_keyboard(
        on_press=press,
        on_release=release,
        delay_second_char=2,
        delay_other_chars=0.1,
    )

class STOPFLAG(): pass
class SAVEVOC(): pass

def process_classification(queue: mp.Queue, shm_name: str, lock: mp.Lock, board: Boards, log_level: int):
    shm = shared_memory.SharedMemory(name=shm_name)
    img_stack = np.ndarray((320,320,3), dtype="uint8", buffer=shm.buf)
    logging.basicConfig(level=log_level)

    img_stack[:,:,:] = np.ones((320,320,3))*255

    while True:
        try:
            index = queue.get(1)
        except Empty:
            logging.error("Close Classification process")
            break

        if isinstance(index, STOPFLAG):
            logging.debug("got stop signal\n Exit Classification process")
            break

        img = np.zeros((320,320,3), dtype=np.uint8)
        with lock:
            img[:,:,:] = img_stack[:,:,:]

        match board:
            case Boards.MED3_REV100:
                currentBoard = MED3_rev100(img)

        if isinstance(index, SAVEVOC):
            logging.debug("create Image")
            currentBoard.save_voc('/home/weston/dataset_train')

def main():

    #create working dir
    home_path = os.path.expanduser("~")
    working_path = home_path + '/.tht-classificator'
    if not os.path.exists(working_path):
        os.makedirs(working_path)

    dataset_path = working_path + '/data_set'
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)

    imgcnt=0

    #parse Programm arguments
    parser = argparse.ArgumentParser(description='THT-Classificator: Erkenne und bewerte THT-Steckverbinder')
    parser.add_argument('-b', '--board', type=str, help='Which board is evaluated?', default='MED3_rev1.00',choices=['MED3_rev1.00'])
    parser.add_argument('-s', '--settings', type=str, help='Path to Settings-File', default=working_path + '/Settings.xml')
    parser.add_argument('-d', '--debug', action='store_true', help='print debugging messages')
    parser.add_argument('--maintain', action='store_true', help='Enable Maintain-Mode. It is used to create VOC datasets')
    args = parser.parse_args()

    #Set debug mode
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
        logging.debug('Enable debugging messages')
    else:
        logging.basicConfig(level=logging.INFO)

    if args.board == "MED3_rev1.00":
        board = Boards.MED3_REV100

    #Prepair multiprocessing for classification
    if args.maintain:
        sh_array_lock = mp.Lock()
        shared_img = shared_memory.SharedMemory(create=True, size=(320*320*3))
        img_stack = np.ndarray((320,320,3), dtype="uint8", buffer=shared_img.buf)
        class_queue = mp.Queue()
        frame_index = 0
        classProcess = mp.Process(target=process_classification, args=(class_queue, shared_img.name, sh_array_lock, board, logging.getLogger().getEffectiveLevel()))
        classProcess.start()

    #Start GStreamer
    Settings= xmlSettings(args.settings)
    img_processing = FrameProccessing(Settings)

    #Set Keyboard polling
    if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
        thread = threading.Thread(target=listen_keyboard_wrapper)
        thread.start()

    #MAIN loop
    while True:
        img_processing.update()

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
        if logging.getLogger().getEffectiveLevel() == logging.DEBUG:

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

                #when maintain is enabled
                if args.maintain:
                    if key_pressed.f5:
                        logging.debug("extand Dataset")
                        class_queue.put(SAVEVOC())
                        key_pressed.f5 = False

                    frame_index = frame_index + 1
                    if frame_index % 5 == 0:
                        with sh_array_lock:
                            img_stack[:,:,:] = img_processing.object_img[:,:,:]

                            class_queue.put(frame_index)

            img_processing.Settings.parse(args.settings)

if __name__ == '__main__':
    main()
