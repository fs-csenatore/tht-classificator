import xml.etree.ElementTree as xmlET
from os.path import isfile, expanduser
import logging

class xmlSettings(xmlET.ElementTree):

    def __init__(self, xml_file :str):
        super().__init__(xmlET.Element('Settings'))
        if isfile(xml_file):
            self.parse(xml_file)
        else:
            self.set_default_value()
            xmlET.indent(self,space="\t", level=0)
            self.write(xml_file,encoding="unicode", xml_declaration=True, method="xml")

    def set_default_value(self):
        hsv = xmlET.SubElement(self.getroot(), "HSV")
        lower_bound = xmlET.SubElement(hsv, "lowerBound")
        upper_bound = xmlET.SubElement(hsv, "upperBound")
        xmlET.SubElement(lower_bound, "H").text = "10"
        xmlET.SubElement(lower_bound, "S").text = "30"
        xmlET.SubElement(lower_bound, "V").text = "30"
        xmlET.SubElement(upper_bound, "H").text = "130"
        xmlET.SubElement(upper_bound, "S").text = "255"
        xmlET.SubElement(upper_bound, "V").text = "255"

        StreamCap = xmlET.SubElement(self.getroot(), "StreamCap")
        xmlET.SubElement(StreamCap, "framerate").text = "5"
        xmlET.SubElement(StreamCap, "Frame-Width").text ="1920"
        xmlET.SubElement(StreamCap, "Frame-Height").text = "1080"
        xmlET.SubElement(StreamCap, "Frame-Format").text = "YUY2"
        xmlET.SubElement(StreamCap, 'rotation').text = '2'
        xmlET.SubElement(StreamCap, 'dist-file').text = expanduser("~")+ '/.tht-classificator/distortion.pkl'

        StreamWrite = xmlET.SubElement(self.getroot(), "StreamWrite")
        xmlET.SubElement(StreamWrite, "framerate").text = "5"
        xmlET.SubElement(StreamWrite, "Frame-Width").text ="1920"
        xmlET.SubElement(StreamWrite, "Frame-Height").text = "1080"
        xmlET.SubElement(StreamWrite, "Frame-Format").text = "GRAY8"

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
        return self.findtext('StreamCap/dist-file')
