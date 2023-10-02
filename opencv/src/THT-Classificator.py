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
    args = parser.parse_args()

    #Set debug mode
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
        logging.debug('Enable debugging messages')
    else:
        logging.basicConfig(level=logging.INFO)

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
                    img_processing.wrt_frame[0:shape2[0], img_processing.wrt_frame.shape[1]-shape2[1]:, :] = img_processing.object_img[:,:,:]
                except:
                    logging.error("Could not display object_img")
                if key_pressed.f5:
                    #TODO: Automated dataset-creation
                    imgcnt = imgcnt + 1
                    cv2.imwrite(args.board + 'rot' + str(imgcnt) + '.png', img_processing.object_img)
                    print('saved image as test.png')
                    key_pressed.f5 = False

            img_processing.Settings.parse(args.settings)


if __name__ == '__main__':
    main()
