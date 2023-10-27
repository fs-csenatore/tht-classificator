import os
import threading
import logging
import argparse
from sshkeyboard import listen_keyboard
import multiprocessing
from multiprocessing import shared_memory
from FSBoard import  Boards
import mp
import sys
import ctypes
import time
from queue import Empty
import traceback 


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

    #create working dir
    home_path = os.path.expanduser("~")
    working_path = home_path + '/.tht-classificator'
    if not os.path.exists(working_path):
        os.makedirs(working_path)

    dataset_path = working_path + '/data_set'
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)

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

    if hasattr(args, "settings"):
        settings_path = args.settings
    else:
        logging.error("Settings-Path is required")
        exit(1)

    #Set Keyboard polling
    thread = threading.Thread(target=listen_keyboard_wrapper)
    thread.start()

    sh_buff_lock = multiprocessing.Lock()
    shared_img = shared_memory.SharedMemory(create=True, size=(2*320*320*3))

    #Prepair multiprocessing for classification
    class_queue_in = multiprocessing.Queue()
    class_queue_out = multiprocessing.Queue()
    classProcess = multiprocessing.Process(target=mp.process_classification, args=(class_queue_in, class_queue_out, shared_img.name, sh_buff_lock, board, logging.getLogger().getEffectiveLevel()))
    classProcess.start()

    #Prepair multiprocessing for Img-Processing
    prep_queue_in = multiprocessing.Queue()
    prep_queue_out = multiprocessing.Queue()
    prepProcess = multiprocessing.Process(target=mp.process_preprocess, args=(settings_path, prep_queue_in, prep_queue_out, shared_img.name, sh_buff_lock, logging.getLogger().getEffectiveLevel()))
    prepProcess.start()

    try:
        while True:
            prep_signal = prep_queue_out.get()
        
            try:
                class_signal = class_queue_out.get(False)
            except Empty:
                class_signal = None
                
            if isinstance(prep_signal, mp.PUT):
                class_queue_in.put(1)

            if key_pressed.f5:
                class_queue_in.put(mp.SAVEVOC())
                key_pressed.f5 = False
            
            if isinstance(class_signal, mp.STOPFLAG):
                break
            
            if isinstance(class_signal, mp.PUT):
                prep_queue_in.put(mp.PUT())

            if isinstance(prep_signal, mp.STOPFLAG):
                break

    except:
        logging.info("Exception uccored")
        traceback.print_exc()


    logging.info("closing")
    classProcess.terminate()
    prepProcess.terminate()
    shared_img.unlink()
    kill_thread(thread)
    sys.exit(1)

if __name__ == '__main__':
    main()
