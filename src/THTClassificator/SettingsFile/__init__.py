import xml.etree.ElementTree as xmlET
from os.path import isfile
import os


class xmlSettings(xmlET.ElementTree):

    def __init__(self):
        super().__init__(xmlET.Element('Settings'))
        self.workdir = os.path.join(os.path.expanduser("~"),
                                        '.tht-classificator')
        self.xml_file =  os.path.join(self.workdir,
                                        'Settings.xml')
        assert isfile(self.xml_file)
        self.parse(self.xml_file)


    def set_default_value(self):
        std_xml_file = os.path.join(os.path.dirname(__file__),
                                    'Settings.xml',)
        self.parse(std_xml_file)


    def load(self):
        self.parse(self.xml_file)




class FrameSettings(xmlSettings):
    def __init__(self):
        super().__init__()


    def get_hsv_boundings(self):
        lowerH = int(self.findtext('HSV/lowerBound/H'))
        lowerS = int(self.findtext('HSV/lowerBound/S'))
        lowerV = int(self.findtext('HSV/lowerBound/V'))

        upperH = int(self.findtext('HSV/upperBound/H'))
        upperS = int(self.findtext('HSV/upperBound/S'))
        upperV = int(self.findtext('HSV/upperBound/V'))

        return (lowerH, lowerS, lowerV ), (upperH, upperS, upperV)


    def get_streamwrite_framerate(self):
        return int(self.findtext('StreamWrite/framerate'))


    def get_streamcap_framerate(self):
        return int(self.findtext('StreamCap/framerate'))


    def __get_streamwrite_width(self):
        return int(self.findtext('StreamWrite/Frame-Width'))


    def __get_streamcap_width(self):
        return int(self.findtext('StreamCap/Frame-Width'))


    def __get_streamwrite_height(self):
        return int(self.findtext('StreamWrite/Frame-Height'))


    def __get_streamcap_height(self):
        return int(self.findtext('StreamCap/Frame-Height'))


    def __get_streamwrite_format(self):
        return self.findtext('StreamWrite/Frame-Format')


    def __get_streamcap_format(self):
        return self.findtext('StreamCap/Frame-Format')


    def __get_streamcap_rotation(self):
        return self.findtext('StreamCap/rotation')


    def get_streamcap_gstreamer_string(self):
        string = "v4l2src ! video/x-raw, "
        string = string + "width=" + str(self.__get_streamcap_width()) + ", "
        string = string + "height=" + str(self.__get_streamcap_height()) + ", "
        string = string + "format=" + self.__get_streamcap_format() + ", "
        string = string + "framerate=" + str(self.get_streamcap_framerate()) + "/1 "
        string = string + "! queue ! imxvideoconvert_pxp rotation=" + self.__get_streamcap_rotation() + " "
        string = string + "! video/x-raw, format=BGR ! queue ! appsink"
        return string


    def get_streamwrite_gstreamer_string(self):
        string = "appsrc ! video/x-raw, "
        string = string + "width=" + str(self.__get_streamwrite_width()) + ", "
        string = string + "height=" + str(self.__get_streamwrite_height()) + ", "
        string = string + "format=" + self.__get_streamwrite_format() + " "
        string = string + "! queue ! videoconvert ! video/x-raw, format=BGRx ! queue ! fpsdisplaysink sync=false"
        return string


    def get_streamwrite_resolution(self):
        return (self.__get_streamwrite_width(),
                     self.__get_streamwrite_height())


    def get_streamcap_resolution(self):
        return (self.__get_streamcap_width(),
                     self.__get_streamcap_height())


    def is_streamwrite_colored(self):
        if self.__get_streamwrite_format() == "GRAY8":
            return False
        else:
            return True


    def get_distortion_file(self):
        return os.path.join(self.workdir, self.findtext('StreamCap/dist-file'))


    def is_distortion_enabled(self):
        if self.findtext('StreamCap/dist-en') == "True":
            return True
        else:
            return False




class TFLITESettings(xmlSettings):
    def __init__(self):
        super().__init__()


    def get_model_path(self):
        return os.path.join(self.workdir, self.findtext('tflite/inference/model-file'))


    def get_label_path(self):
        return os.path.join(self.workdir, self.findtext('tflite/inference/label-map'))


    def get_delegate(self):
        return int(self.findtext('tflite/inference/delegate'))
    
    
    def get_dataset_path(self):
        return os.path.join(os.path.expanduser("~"), self.findtext('tflite/dataset-path'))