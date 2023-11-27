import logging
import multiprocessing as mp
from multiprocessing import shared_memory
import numpy as np
from THTClassificator.FrameProcessing import FrameProccessing
import cv2
from dataclasses import dataclass
import os.path
import traceback
from THTClassificator import FSBoard
from THTClassificator.SettingsFile import TFLITESettings
from typing import Union
from queue import Empty


class run(): pass
class done(): pass

#Process Signals
class Signals():
    def __init__(self, state: Union[run , done]):
        self.state = state

class STOPFLAG():pass
        
class SAVEVOC(Signals):
    def __init__(self, state: Union[run , done] = run()):
        super().__init__(state)

class doAI(Signals):
    def __init__(self, state: Union[run , done] = run()):
        super().__init__(state)

class doFrame(Signals):
    def __init__(self, state: Union[run , done] = run()):
        super().__init__(state)

#SSHKEYBOARD
@dataclass
class Keyboard:
    f5: bool
    f6: bool
    f7: bool


def process_classification(queue_in: mp.Queue, queue_out: mp.Queue, shm_name: str, lock: mp.Lock, board: FSBoard.Boards, log_level: int):    
    settings  = TFLITESettings()
    
    #Init shared buffer
    shm = shared_memory.SharedMemory(name=shm_name)
    img_buf = np.ndarray(settings.get_shared_buf_array_dim(), dtype="uint8", buffer=shm.buf)

    if board == FSBoard.Boards.MED3_REV100:
        model_file = settings.get_model_path()
        label_map_file = settings.get_label_path()
    else:
        logging.error("Board not defined!")
        queue_out.put(STOPFLAG())
        queue_in.close()
        queue_out.close()
        return 1

    if not os.path.isfile(model_file):
        logging.error('Vela-Model not found!')
        queue_out.put(STOPFLAG())
        queue_in.close()
        queue_out.close()
        return 1

    if not os.path.isfile(label_map_file):
        logging.error('Label map not found!')
        queue_out.put(STOPFLAG())
        queue_in.close()
        queue_out.close()
        return 1
  
    match board:
        case FSBoard.Boards.MED3_REV100:

            currentBoard = FSBoard.MED3_rev100(model_file, label_map_file, settings.get_delegate())


    try:
        while True:
            #To Communicate with other Processes, we use Queues.
            #Each Queue Entry that is put, is a uniqe class 
            signal = queue_in.get()
            if isinstance(signal, STOPFLAG):
                logging.debug("got stop signal\n Exit Classification process")
                break
            
            #RUN AI-Inference
            if isinstance(signal, doAI):
                img = np.zeros(settings.get_shared_buf_array_dim()[1:], dtype=np.uint8)
                with lock:
                    img[:,:,:] = img_buf[0][:,:,:]
                
                currentBoard.set_img(img)
                currentBoard.get_results()
                
                with lock:
                    img_buf[1][:,:,:] = currentBoard.get_result_image()
                
                queue_out.put(doAI(done()))

            #Save input-tensor
            if isinstance(signal, SAVEVOC):
                if not os.path.exists(settings.get_dataset_path()):
                    os.makedirs(settings.get_dataset_path())
                currentBoard.make_screenshot(settings.get_dataset_path())
                queue_out.put(SAVEVOC(done()))
    except:
        traceback.print_exc()
        logging.error('exception uccoured')
        logging.error("Close Classification process")
    
    queue_out.put(STOPFLAG())
    shm.close()
    queue_in.close()
    queue_out.close()


def process_preprocess(queue_in: mp.Queue, queue_out: mp.Queue, shm_name: str, lock: mp.Lock, log_level: int):
    settings = TFLITESettings()
    #Init shared buffer
    shm = shared_memory.SharedMemory(name=shm_name)
    img_buf = np.ndarray(settings.get_shared_buf_array_dim(), dtype="uint8", buffer=shm.buf)
    tmp_image = np.zeros(settings.get_model_input_size()[1:], dtype=np.uint8)

    #Start GStreamer
    try:
        img_processing = FrameProccessing()

        while True:
            try:
                signal = queue_in.get(False)
            except Empty:
                signal = None

            
            if isinstance(signal, STOPFLAG):
                break
            
            if isinstance(signal, doFrame):
                skip_preprocess = False
            else:
                skip_preprocess = True

            if img_processing.update(skip_preprocess):
                if skip_preprocess == False:
                    logging.debug("try doFrame again")
                    queue_in.put(doFrame())

            #Send Board Img to Classification Process
            if hasattr(img_processing, "object_img"):
                img_processing.object_img = cv2.resize(img_processing.object_img, settings.get_model_input_size()[1:3], cv2.INTER_LINEAR_EXACT)
                with lock:
                    img_buf[0][:,:,:] = img_processing.object_img[:,:,:]
                queue_out.put(doFrame(done()))

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

                #Show rotated object:
                with lock:
                    shape2 = max(shape1[0], shape1[1])
                    img_prev=cv2.resize(img_buf[0],(shape2,shape2))
                
                shape2 = img_prev.shape
                img_processing.wrt_frame[shape1[0]:shape1[0]+shape2[0], 0:shape2[1], :] = img_prev[:,:,:]
                 


                img_processing.Settings.load()

            #Show AI result
            if isinstance(signal, doFrame):
                with lock:
                    tmp_image = img_buf[1][:,:,:]
                    tmp_image = cv2.resize(tmp_image, (800, 800))
            
            img_processing.wrt_frame[img_processing.wrt_frame.shape[0]-800:, img_processing.wrt_frame.shape[1]-800:, :] = tmp_image[:,:,:]
    
    except:
        traceback.print_exc()
        queue_out.put(STOPFLAG())
    
    shm.close()
    queue_in.close()
    queue_out.close()