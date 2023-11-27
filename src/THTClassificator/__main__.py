import os
import threading
import logging
import argparse
from sshkeyboard import listen_keyboard
import multiprocessing
from multiprocessing import shared_memory
from THTClassificator.FSBoard import Boards
import THTClassificator.mp as mp
import sys
import ctypes
import time
from queue import Empty
import traceback 
import os
import shutil
from THTClassificator.SettingsFile import TFLITESettings


key_pressed = mp.Keyboard(False, False, False)

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

# inspired by https://github.com/mosquito/crew/blob/master/crew/worker/thread.py
def kill_thread(
        thread: threading.Thread, exception: BaseException=KeyboardInterrupt
) -> None:
    if not thread.is_alive():
        return

    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(
        ctypes.c_long(thread.ident), ctypes.py_object(exception)
    )

    if res == 0:
        raise ValueError('nonexistent thread id')
    elif res > 1:
        ctypes.pythonapi.PyThreadState_SetAsyncExc(thread.ident, None)
        raise SystemError('PyThreadState_SetAsyncExc failed')

    while thread.is_alive():
        time.sleep(0.01)

def main():
    #ensures that working dir exists
    home_path = os.path.expanduser("~")
    working_path = home_path + '/.tht-classificator'
    if not os.path.exists(working_path):
        os.makedirs(working_path)

    #ensures that bin files are available
    pkl_found = False
    tflite_found = False
    xml_found = False

    for fname in os.listdir(working_path):
        if fname.endswith('.pkl'):
            pkl_found = True
        if fname.endswith('.tflite'):
            tflite_found = True
        if fname.endswith('.xml'):
            xml_found = True
    pkg_path = os.path.dirname(__file__)
    bin_path = os.path.join(pkg_path, 'bin')
    for fname in os.listdir(bin_path):
        if not pkl_found and fname.endswith('.pkl'):
            shutil.copy(os.path.join(bin_path, fname), working_path)
        elif not tflite_found and fname.endswith('.tflite'):
            shutil.copy(os.path.join(bin_path, fname), working_path)
    
    if not xml_found:
        shutil.copy(os.path.join(pkg_path, 'SettingsFile', 'Settings.xml'),
                    working_path)

    #parse Programm arguments
    parser = argparse.ArgumentParser(description='THT-Classificator: Erkenne und bewerte THT-Steckverbinder')
    parser.add_argument('-b', '--board', type=str, help='Which board is evaluated?', default='MED3_rev1.00',choices=['MED3_rev1.00'])
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

    #Set Keyboard polling
    thread = threading.Thread(target=listen_keyboard_wrapper)
    thread.start()

    settings = TFLITESettings()
    sh_buff_lock = multiprocessing.Lock()
    shared_img = shared_memory.SharedMemory(create=True, size=settings.get_shared_buf_size())

    #Prepair multiprocessing for classification
    class_queue_in = multiprocessing.Queue()
    class_queue_out = multiprocessing.Queue()
    classProcess = multiprocessing.Process(target=mp.process_classification, args=(class_queue_in, class_queue_out, shared_img.name, sh_buff_lock, board, logging.getLogger().getEffectiveLevel()))
    classProcess.start()

    #Prepair multiprocessing for Img-Processing
    prep_queue_in = multiprocessing.Queue()
    prep_queue_out = multiprocessing.Queue()
    prepProcess = multiprocessing.Process(target=mp.process_preprocess, args=(prep_queue_in, prep_queue_out, shared_img.name, sh_buff_lock, logging.getLogger().getEffectiveLevel()))
    prepProcess.start()
    prep_queue_in.put(mp.doFrame())

    try:
        while True:
            try:
                prep_signal = prep_queue_out.get(False)
            except Empty:
                prep_signal = None

            try:
                class_signal = class_queue_out.get(False)
            except Empty:
                class_signal = None

            #Handle Stop Flags   
            if isinstance(class_signal, mp.STOPFLAG):
                prep_queue_in.put(mp.STOPFLAG())
                break
            
            if isinstance(prep_signal, mp.STOPFLAG):
                class_queue_in.put(mp.STOPFLAG())
                break

            #return Classifier Results
            if isinstance(class_signal, mp.doAI) and isinstance(class_signal.state, mp.done):
                prep_queue_in.put(mp.doFrame())
                logging.debug("MAIN says doFrame to PrepProcess")

            
            #Save Image in dataset
            if key_pressed.f5:
                class_queue_in.put(mp.SAVEVOC())
                logging.debug("MAIN says SAVEVOC to ClassProcess")

                key_pressed.f5 = False
            
            if isinstance(class_signal, mp.SAVEVOC) and isinstance(class_signal.state, mp.done):
                logging.info("Image saved for VOC Dataset")

            #start Classifier Process
            if isinstance(prep_signal, mp.doFrame) and isinstance(prep_signal.state, mp.done):
                class_queue_in.put(mp.doAI())
                logging.debug("MAIN says DoAI to ClassProcess")

    except:
        logging.info("Exception uccored")
        traceback.print_exc()
        class_queue_in.put(mp.STOPFLAG)
        prep_queue_in.put(mp.STOPFLAG)


    logging.info("closing THT-Classificator")
    classProcess.join()
    prepProcess.join()
    kill_thread(thread)
    shared_img.unlink()
    sys.exit(1)

if __name__ == '__main__':
    main()
