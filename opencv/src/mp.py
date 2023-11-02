import logging
import multiprocessing as mp
from multiprocessing import shared_memory
from queue import Empty
import numpy as np
from FSBoard import MED3_rev100, Boards
from FrameProcessing import FrameProccessing
from SettingsFile import xmlSettings
import cv2
from dataclasses import dataclass
import tflite_runtime.interpreter as tflite
import os.path
from datetime import datetime
import traceback
import FSBoard

#Process Signals  
class STOPFLAG(): pass
class SAVEVOC(): pass
class ACK(): pass
class PUT(): pass
class doAI(): pass

#SSHKEYBOARD
@dataclass
class Keyboard:
    f5: bool
    f6: bool
    f7: bool


def process_classification(queue_in: mp.Queue, queue_out: mp.Queue, shm_name: str, lock: mp.Lock, board: Boards, log_level: int):    
    #Init shared buffer
    shm = shared_memory.SharedMemory(name=shm_name)
    img_buf = np.ndarray((2,640,640,3), dtype="uint8", buffer=shm.buf)
    
    #check working dir
    home_path = os.path.expanduser("~")
    working_path = home_path + '/.tht-classificator'
    if not os.path.exists(working_path):
        os.makedirs(working_path)

    if board == Boards.MED3_REV100:
        model_file = 'MED3_ssd_mobilenet_v2_640x640_fpnlite_vela.tflite'
    else:
        logging.error("Board not defined!")
        queue_out.put(STOPFLAG())
        queue_in.close()
        queue_out.close()
        return 1

    if os.path.isfile(working_path + '/' + model_file):
        model_file = working_path + '/' + model_file
    else:
        logging.error('Vela-Model not found!')
        queue_out.put(STOPFLAG())
        queue_in.close()
        queue_out.close()
        return 1
  
    match board:
        case FSBoard.Boards.MED3_REV100:
            label_map_file = working_path + '/tflite_label_map.txt'

            currentBoard = FSBoard.MED3_rev100(model_file, label_map_file, 0)

    try:
        while True:
            #To Communicate with other Processes, we use Queues.
            #Each Queue Entry that is put, is a uniqe class 
            signal = queue_in.get()
            if isinstance(signal, STOPFLAG):
                logging.debug("got stop signal\n Exit Classification process")
                break
            
            img = np.zeros((640,640,3), dtype=np.uint8)
            with lock:
                img[:,:,:] = img_buf[0][:,:,:]
            
            currentBoard.set_img(img)
            currentBoard.get_results()
            
            with lock:
                img_buf[1][:,:,:] = currentBoard.get_result_image()
            
            queue_out.put(PUT())

            #Make a screenshot
            if isinstance(signal, SAVEVOC):
                print("create Image")
                #img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                dataset_rootpath = '/home/weston/dataset_train'
                imagesets_path = dataset_rootpath + "/ImageSets"
                cv2.imwrite(imagesets_path + '/MED3-REV1.00_' + timestamp + '.jpg', img)
            
            #TODO: Implement inference with doAI Singal
            if isinstance(signal, doAI):
                pass
    except:
        traceback.print_exc()
        logging.error('exception uccoured')
        logging.error("Close Classification process")
    
    queue_out.put(STOPFLAG())
    shm.close()
    queue_in.close()
    queue_out.close()


def process_preprocess(settings_path: str, queue_in: mp.Queue, queue_out: mp.Queue, shm_name: str, lock: mp.Lock, log_level: int):
    #Init shared buffer
    shm = shared_memory.SharedMemory(name=shm_name)
    img_buf = np.ndarray((2,640,640,3), dtype="uint8", buffer=shm.buf)
    tmp_image = np.zeros((640,640,3),np.uint8)
    frame_index = 0
    od_every_n_frame = 5

    #Start GStreamer
    Settings= xmlSettings(settings_path)
    img_processing = FrameProccessing(Settings)

    try:
        while True:
            img_processing.update()

            #Send Board Img to Classification Process
            if hasattr(img_processing, "object_img"):
                frame_index = frame_index + 1

                if frame_index % od_every_n_frame == 1:
                    with lock:
                        img_buf[0][:,:,:] = img_processing.object_img[:,:,:]
                    
                    queue_out.put(PUT())

            #Draw Rectangle around object
            if hasattr(img_processing, 'scaled_box'):
                x, y, w, h = cv2.boundingRect(img_processing.scaled_box)
                cv2.drawContours(img_processing.cap_frame, [img_processing.scaled_box], 0, (255,255,0),2)
                cv2.rectangle(img_processing.cap_frame, (x, y), (x + w, y + h), (0,0,255), 4)


            #Show Captured Frame
            if log_level == logging.DEBUG:
                img_processing.wrt_frame[:,
                    img_processing.wrt_frame.shape[1]-img_processing.cap_frame.shape[1]:,
                    :] = img_processing.cap_frame
            else:
                img_processing.wrt_frame[:,
                    :img_processing.cap_frame.shape[1],
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
                        #Show object_img
                        shape2 = img_processing.object_img.shape
                        img_processing.wrt_frame[0:shape2[0], img_processing.wrt_frame.shape[1]-shape2[1]:, :] = img_processing.object_img[:,:,:]

                    except:
                        traceback.print_exc() 
                        logging.error("Could not display object_img")

                img_processing.Settings.parse(settings_path)

            #Show AI result
            with lock:
                tmp_image = img_buf[1][:,:,:]

            tmp_image = cv2.resize(tmp_image, (700, 700))
            img_processing.wrt_frame[img_processing.wrt_frame.shape[0]-700:, img_processing.wrt_frame.shape[1]-700:, :] = tmp_image[:,:,:]
    except:
        traceback.print_exc()
        queue_out.put(STOPFLAG())
        shm.close()
        queue_in.close()
        queue_out.close()