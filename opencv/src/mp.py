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
from datetime import datetime
import traceback

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

def normalize_input_tensor(tensor: np.ndarray):
    return (np.float32(tensor) - 127.5) / 127.5

def determine_output_index(output_details):
    # Check output layer name to determine if this model was created with TF2 or TF1
    if ('StatefulPartitionedCall' in output_details[0]['name']):
        # This is a TF2 model
        boxes_idx, classes_idx, scores_idx, count_idx = 1, 3, 0, 2
    else:
        # This is a TF1 model
        boxes_idx, classes_idx, scores_idx, count_idx = 0, 1, 2, 3

    return  boxes_idx, classes_idx, scores_idx, count_idx

def detect_objects(interpreter: tflite.Interpreter, image, threshold=0.0):
    """Returns a list of detection results, each a dictionary of object info."""
    
    #model_input shape is (1xheightxwidthxdepth) -> we must expand the img data
    input_tensor= np.expand_dims(image, axis=0)

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    if input_details[0]['dtype'] == np.float32:
        input_tensor = normalize_input_tensor(input_tensor)

    boxes_idx, classes_idx, scores_idx, count_idx = \
        determine_output_index(output_details)

    start_time = time.time()
    # Perform the actual detection by running the model with the image as input
    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.invoke()
    end_time = time.time()
    logging.debug("Detection-Runtime in ms =%f",(end_time-start_time)*10**3)
    
    # Retrieve detection results
    boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[0]
    scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0]
    classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0]
    count = int(interpreter.get_tensor(output_details[count_idx]['index'])[0])

    #Get all detected objects and return the highest score from each class_id
    results = list()
    for i in range(count):
        if scores[i] >= threshold:
            result = {
                'bounding_box': boxes[i],
                'class_id': classes[i],
                'score': scores[i],
            }
            results.append(result)

    results = sorted(results, key=lambda x: x['score'], reverse=True)

    objects = list()
    obj_idx = list()
    for obj in results:
        if obj['class_id'] not in obj_idx:
            obj_idx.append(obj['class_id'])
            objects.append(obj)

    return objects

def process_classification(queue_in: mp.Queue, queue_out: mp.Queue, shm_name: str, lock: mp.Lock, board: Boards, log_level: int):    
    #Init shared buffer
    shm = shared_memory.SharedMemory(name=shm_name)
    img_buf = np.ndarray((2,320,320,3), dtype="uint8", buffer=shm.buf)
    
    #check working dir
    home_path = os.path.expanduser("~")
    working_path = home_path + '/.tht-classificator'
    if not os.path.exists(working_path):
        os.makedirs(working_path)

    if board == Boards.MED3_REV100:
        model_file = 'MED3_ssd_mobilenet_v2_320x320_fpnlite_vela.tflite'
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
    
    #To run on Ethos-U or GPU, we need delegates.
    #To run on CPU, remove delegate attribute in tflite.Interpreter
    ext_delegate = [tflite.load_delegate('/usr/lib/libethosu_delegate.so')]
    interpreter = tflite.Interpreter(model_file,experimental_delegates=ext_delegate)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    if log_level == logging.DEBUG:
        print("Input details: ", input_details)
        print("Output details: ", output_details)
        
    match board:
        case Boards.MED3_REV100:
            currentBoard = MED3_rev100(working_path + '/tflite_label_map.txt')

    try:
        while True:
            #To Communicate with other Processes, we use Queues.
            #Each Queue Entry that is put, is a uniqe class 
            signal = queue_in.get()
            if isinstance(signal, STOPFLAG):
                logging.debug("got stop signal\n Exit Classification process")
                break
            
            img = np.zeros((320,320,3), dtype=np.uint8)
            with lock:
                img[:,:,:] = img_buf[0][:,:,:]
            
            
            currentBoard.set_img(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            
            results = detect_objects(interpreter,currentBoard.image, threshold=0.3)

            """
            bounding_box contains two points y1, x1 and y2, x2.
            The Values are standardized and must be scaled to image.shape
            """
            for obj in results:
                y1, x1, y2, x2 = obj['bounding_box']
                y1 = int(y1 * currentBoard.image.shape[0])
                y2 = int(y2 * currentBoard.image.shape[0])
                x1 = int(x1 * currentBoard.image.shape[1])
                x2 = int(x2 * currentBoard.image.shape[1])
            
                class_id = int(obj['class_id'])
                color = currentBoard.labels_color[class_id]
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

            #Make a screenshot
            if isinstance(signal, SAVEVOC):
                print("create Image")
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
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
    img_buf = np.ndarray((2,320,320,3), dtype="uint8", buffer=shm.buf)
    tmp_image = np.zeros((320,320,3)).astype(np.uint8)
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

            #Show AI result
            with lock:
                tmp_image = img_buf[1][:,:,:]

            tmp_image = cv2.resize(tmp_image, (700, 700))
            img_processing.wrt_frame[img_processing.wrt_frame.shape[0]-700:, img_processing.wrt_frame.shape[1]-700:, :] = tmp_image[:,:,:]


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
    except:
        queue_out.put(STOPFLAG())
        shm.close()
        queue_in.close()
        queue_out.close()