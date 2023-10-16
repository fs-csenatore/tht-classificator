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
import tflite_runtime.interpreter as tflite
import os.path
from FSBoard import Boards
import time
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

def detect_objects(interpreter: tflite.Interpreter, image, threshold=0.5):
    """Returns a list of detection results, each a dictionary of object info."""
    
    input_data = np.expand_dims(image, axis=0)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    #Normalize input data
    floating_model = input_details[0]['dtype'] == np.float32
    if floating_model:
        input_data = (np.float32(input_data) - 127.5) / 127.5

    # Check output layer name to determine if this model was created with TF2 or TF1,
    # because outputs are ordered differently for TF2 and TF1 models
    if ('StatefulPartitionedCall' in output_details[0]['name']):
        # This is a TF2 model
        boxes_idx, classes_idx, scores_idx, count_idx = 1, 3, 0, 2
    else:
        # This is a TF1 model
        boxes_idx, classes_idx, scores_idx, count_idx = 0, 1, 2, 3

    start_time = time.time()
    # Perform the actual detection by running the model with the image as input
    try:
        interpreter.set_tensor(input_details[0]['index'], input_data)
    except ValueError:
        print('ValueError in set_tensor')
        raise Exception("Object detection crashed")
    
    try:
        interpreter.invoke()
    except ValueError:
        print('ValueError in invoke')
        raise Exception("Object detection crashed")
    end_time = time.time()
    logging.debug("Detection-Runtime in ms =%f",(end_time-start_time)*10**3)
    
    # Retrieve detection results
    boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[0]
    scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0]
    classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0]
    count = int(interpreter.get_tensor(output_details[count_idx]['index'])[0])

    results = list()
    print(count)
    for i in range(count):
        if scores[i] >= threshold:
            result = {
                'bounding_box': boxes[i],
                'class_id': classes[i],
                'score': scores[i],
            }
            results.append(result)
    return results

def process_classification(queue_in: mp.Queue, queue_out: mp.Queue, shm_name: str, lock: mp.Lock, board: Boards, log_level: int):    
    #check working dir
    home_path = os.path.expanduser("~")
    working_path = home_path + '/.tht-classificator'
    if not os.path.exists(working_path):
        os.makedirs(working_path)

    if board == Boards.MED3_REV100:
        model_file = 'med3_detect_efficientdet_lite0_vela.tflite'
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
    
    ext_delegate = [tflite.load_delegate('/usr/lib/libethosu_delegate.so')]
    interpreter = tflite.Interpreter(model_file,experimental_delegates=ext_delegate)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    print("Input details: ", input_details)
    output_details = interpreter.get_output_details()
    print("Output details: ", output_details)
    print(input_details[0]['index'])

    shm = shared_memory.SharedMemory(name=shm_name)
    img_buf = np.ndarray((2,320,320,3), dtype="uint8", buffer=shm.buf)
    try:
        while True:
            signal = queue_in.get()
            if isinstance(signal, STOPFLAG):
                logging.debug("got stop signal\n Exit Classification process")
                break
            
            img = np.zeros((320,320,3), dtype=np.uint8)
            with lock:
                img[:,:,:] = img_buf[0][:,:,:]
            
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            match board:
                case Boards.MED3_REV100:
                    currentBoard = MED3_rev100(img)         
            
            results = detect_objects(interpreter,currentBoard.image)

            """
            bounding_box contains two points y1, x1 and y2, x2.
            The Values are standardized and must be scaled to image.shape
            """
            for obj in results:
                logging.debug(obj)
                y1, x1, y2, x2 = obj['bounding_box']
                y1 = int(y1 * currentBoard.image.shape[0])
                y2 = int(y2 * currentBoard.image.shape[0])
                x1 = int(x1 * currentBoard.image.shape[1])
                x2 = int(x2 * currentBoard.image.shape[1])
            
                color = (255,0,0)
                class_id = int(obj['class_id'])
                cv2.rectangle(currentBoard.image, (x1, y1), (x2, y2), color, 2)
                # Make adjustments to make the label visible for all objects
                y = y1 - 15 if y1 - 15 > 15 else y1 + 15
                label = "{}: {:.0f}%".format(currentBoard.detection_labels[class_id], obj['score'] * 100)
                cv2.putText(currentBoard.image, label, (x1, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            currentBoard.image = cv2.cvtColor(currentBoard.image, cv2.COLOR_RGB2BGR)
            with lock:
                img_buf[1][:,:,:] = currentBoard.image[:,:,:]
            
            queue_out.put(PUT())

            if isinstance(signal, SAVEVOC):
                print("create Image")
                currentBoard.save_voc('/home/weston/dataset_train')
            
            if isinstance(signal, doAI):
                pass
    except:
        logging.error('exception uccoured')
        logging.error("Close Classification process")
    
    queue_out.put(STOPFLAG())
    shm.close()
    queue_in.close()
    queue_out.close()


def process_preprocess(settings_path: str, queue_in: mp.Queue, queue_out: mp.Queue, shm_name: str, lock: mp.Lock, log_level: int):
    shm = shared_memory.SharedMemory(name=shm_name)
    img_buf = np.ndarray((2,320,320,3), dtype="uint8", buffer=shm.buf)
    frame_index = 0

    #Start GStreamer
    Settings= xmlSettings(settings_path)
    img_processing = FrameProccessing(Settings)

    tmp_image = np.zeros((320,320,3)).astype(np.uint8)
    try:
        while True:
            img_processing.update()

            if hasattr(img_processing, "object_img"):
                frame_index = frame_index + 1

                #first of all, use only every 5th object
                if frame_index % 1 == 0:
                    with lock:
                        img_buf[0][:,:,:] = img_processing.object_img[:,:,:]
                    
                    queue_out.put(PUT())

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
                        #Show object_img
                        shape2 = img_processing.object_img.shape
                        img_processing.wrt_frame[0:shape2[0], img_processing.wrt_frame.shape[1]-shape2[1]:, :] = img_processing.object_img[:,:,:]

                        #Show AI result
                        try:
                            signal = queue_in.get(False)
                        except Empty:
                            pass

                        if isinstance(signal, PUT):
                            with lock:
                                tmp_image = img_buf[1][:,:,:]

                        img_processing.wrt_frame[shape2[0]:shape2[0]*2, img_processing.wrt_frame.shape[1]-shape2[1]:, :] = tmp_image[:,:,:]
                    except:
                        logging.error("Could not display object_img")

                img_processing.Settings.parse(settings_path)
    except:
        queue_out.put(STOPFLAG())
        shm.close()
        queue_in.close()
        queue_out.close()