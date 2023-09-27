import os
import logging
import numpy as np
import cv2
import time
import argparse
from SettingsFile import xmlSettings
from FrameProcessing import FrameProccessing

def main():

    #create working dir
    home_path = os.path.expanduser("~")
    working_path = home_path + '/.tht-classificator'
    if not os.path.exists(working_path):
        os.makedirs(working_path)

    #parse Programm arguments
    parser = argparse.ArgumentParser(description='THT-Classificator: Erkenne und bewerte THT-Steckverbinder')
    parser.add_argument('-b', '--board', type=str, help='Which board is evaluated?', default='armstoneA9',choices=['armstoneA9'])
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

    #MAIN loop
    while True:
        img_processing.update()
        img_processing.wrt_frame = img_processing.cap_frame

        #Draw Rectangle
        if hasattr(img_processing, 'scaled_box'):
            x, y, w, h = cv2.boundingRect(img_processing.scaled_box)
            cv2.drawContours(img_processing.wrt_frame, [img_processing.scaled_box], 0, (255,255,0),2)
            cv2.rectangle(img_processing.wrt_frame, (x, y), (x + w, y + h), (0,0,255), 4)

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
                    img_processing.wrt_frame[0:shape2[0], 1920-shape2[1]:, :] = img_processing.object_img[:,:,:]
                except:
                    logging.error("Could not display object_img")
            img_processing.Settings.parse(args.settings)

if __name__ == '__main__':
    main()
