
import xml.etree.ElementTree as xmlET
from os.path import isfile

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

        framerate = xmlET.SubElement(self.getroot(), "framerate")
        xmlET.SubElement(framerate, "in").text = "5"
        xmlET.SubElement(framerate, "out").text = "5"            

    def read_hsv_boundings(self):
        lowerH = int(self.findtext('HSV/lowerBound/H'))
        lowerS = int(self.findtext('HSV/lowerBound/S'))
        lowerV = int(self.findtext('HSV/lowerBound/V'))

        upperH = int(self.findtext('HSV/upperBound/H'))
        upperS = int(self.findtext('HSV/upperBound/S'))
        upperV = int(self.findtext('HSV/upperBound/V'))

        return (lowerH, lowerS, lowerV ), (upperH, upperS, upperV)

    def set_hsv_boundings(self, lowerBound :tuple, upperBound :tuple):
        self.find('HSV/lowerBound/H').text = str(lowerBound[0])
        self.find('HSV/lowerBound/S').text = str(lowerBound[1])
        self.find('HSV/lowerBound/V').text = str(lowerBound[2])

        self.find('HSV/upperBound/H').text = str(upperBound[0])
        self.find('HSV/upperBound/H').text = str(upperBound[1])
        self.find('HSV/upperBound/H').text = str(upperBound[2])

    def get_outgoing_framerate(self):
        return int(self.findtext('framerate/out'))
    
    def get_ingoing_framerate(self):
        return int(self.findtext('framerate/in'))
    

