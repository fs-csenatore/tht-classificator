import numpy as np
import cv2
from dataclasses import dataclass
import xml.etree.ElementTree as xmlET
from datetime import datetime

@dataclass
class boundingbox:
    x: int
    y: int
    w: int
    h: int

def xml_create_object_label(root: xmlET.Element,
                            label: str):
        object = xmlET.SubElement(root, "object")
        name = xmlET.SubElement(object, "name")
        name.text = label

        return root, object

def xml_init(img_label: str):

    root= xmlET.Element("annotation")
    root, _  = xml_create_object_label(root, img_label)

    return root

def xml_add_bbox_label(root: xmlET.Element, label: str, bbox: boundingbox):
    root, object = xml_create_object_label(root, label)
    bndbox = xmlET.SubElement(object, "bndbox")
    xmin = xmlET.SubElement(bndbox, "xmin")
    ymin = xmlET.SubElement(bndbox, "ymin")
    xmax = xmlET.SubElement(bndbox, "xmax")
    ymax = xmlET.SubElement(bndbox, "ymax")
    xmin.text = str(bbox.x)
    xmax.text = str(bbox.x + bbox.w)
    ymin.text = str(bbox.y)
    ymax.text = str(bbox.y + bbox.h)

    return root




class MED3_rev100():

    def __init__(self, med3_img: np.ndarray ):
        self.image = med3_img.copy()

    "Save Image and create a VOC xml file"
    def save_voc(self, dataset_rootpath: str):
        self.__define_bbox()
        annotation_path = dataset_rootpath + "/Annotations"
        imagesets_path = dataset_rootpath + "/ImageSets"
        xml_root = xml_init("MED3-REV1.00")
        xml_root, _ = xml_create_object_label(xml_root, self.rotation_label)
        xml_root = xml_add_bbox_label(xml_root, self.J09_J11_label, self.J09_J11)
        xml_root = xml_add_bbox_label(xml_root, self.J6_label, self.J6)
        xml_root = xml_add_bbox_label(xml_root, self.J12_label, self.J12)
        xml_root = xml_add_bbox_label(xml_root, self.J14_label, self.J14)
        xml_root = xml_add_bbox_label(xml_root, self.J13_label, self.J13)
        xml_root = xml_add_bbox_label(xml_root, self.J2_label, self.J2)
        xml_root = xml_add_bbox_label(xml_root, self.J3_label, self.J3)
        xml_root = xml_add_bbox_label(xml_root, self.C59_label, self.C59)
        xml_root = xml_add_bbox_label(xml_root, self.J4_label, self.J4)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        cv2.imwrite(imagesets_path + '/MED3-REV1.00_' + timestamp + '.jpg', self.image)
        xml_tree = xmlET.ElementTree(xml_root)
        xmlET.indent(xml_tree,space="\t", level=0)
        xml_tree.write(annotation_path + '/MED3-REV1.00_' + timestamp + '.xml',
                       encoding="unicode",
                       xml_declaration=True,
                       method="xml")

    def __define_bbox(self):
        bbox_xml = xmlET.parse('/home/root/.tht-classificator/med3-rev1-bbox.xml')
        bbox_xml_root = bbox_xml.getroot()
        self.J09_J11_label = bbox_xml_root.findtext('J09_J11/label')
        self.J09_J11 = boundingbox(int(bbox_xml_root.findtext('J09_J11/x')),
                                  int(bbox_xml_root.findtext('J09_J11/y')),
                                  int(bbox_xml_root.findtext('J09_J11/w')),
                                  int(bbox_xml_root.findtext('J09_J11/h')))
        self.J6_label = bbox_xml_root.findtext('J6/label')
        self.J6 = boundingbox(int(bbox_xml_root.findtext('J6/x')),
                                  int(bbox_xml_root.findtext('J6/y')),
                                  int(bbox_xml_root.findtext('J6/w')),
                                  int(bbox_xml_root.findtext('J6/h')))
        self.J12_label = bbox_xml_root.findtext('J12/label')
        self.J12 = boundingbox(int(bbox_xml_root.findtext('J12/x')),
                                  int(bbox_xml_root.findtext('J12/y')),
                                  int(bbox_xml_root.findtext('J12/w')),
                                  int(bbox_xml_root.findtext('J12/h')))
        self.J14_label = bbox_xml_root.findtext('J14/label')
        self.J14 = boundingbox(int(bbox_xml_root.findtext('J14/x')),
                                  int(bbox_xml_root.findtext('J14/y')),
                                  int(bbox_xml_root.findtext('J14/w')),
                                  int(bbox_xml_root.findtext('J14/h')))
        self.J13_label = bbox_xml_root.findtext('J13/label')
        self.J13 = boundingbox(int(bbox_xml_root.findtext('J13/x')),
                                  int(bbox_xml_root.findtext('J13/y')),
                                  int(bbox_xml_root.findtext('J13/w')),
                                  int(bbox_xml_root.findtext('J13/h')))
        self.J2_label = bbox_xml_root.findtext('J2/label')
        self.J2 = boundingbox(int(bbox_xml_root.findtext('J2/x')),
                                  int(bbox_xml_root.findtext('J2/y')),
                                  int(bbox_xml_root.findtext('J2/w')),
                                  int(bbox_xml_root.findtext('J2/h')))
        self.J3_label = bbox_xml_root.findtext('J3/label')
        self.J3 = boundingbox(int(bbox_xml_root.findtext('J3/x')),
                                  int(bbox_xml_root.findtext('J3/y')),
                                  int(bbox_xml_root.findtext('J3/w')),
                                  int(bbox_xml_root.findtext('J3/h')))
        self.C59_label = bbox_xml_root.findtext('C59/label')
        self.C59 = boundingbox(int(bbox_xml_root.findtext('C59/x')),
                                  int(bbox_xml_root.findtext('C59/y')),
                                  int(bbox_xml_root.findtext('C59/w')),
                                  int(bbox_xml_root.findtext('C59/h')))
        self.J4_label = bbox_xml_root.findtext('J4/label')
        self.J4 = boundingbox(int(bbox_xml_root.findtext('J4/x')),
                                  int(bbox_xml_root.findtext('J4/y')),
                                  int(bbox_xml_root.findtext('J4/w')),
                                  int(bbox_xml_root.findtext('J4/h')))
        self.rotation_label = bbox_xml_root.findtext('rot')

    def draw_boundingbox(self):
        self.__define_bbox()
        cv2.rectangle(self.image,
                       (self.J09_J11.x, self.J09_J11.y),
                       ((self.J09_J11.x + self.J09_J11.w),
                        (self.J09_J11.y + self.J09_J11.h)),
                       (0,0,255), 1)
        cv2.rectangle(self.image,
                       (self.J6.x, self.J6.y),
                       ((self.J6.x + self.J6.w),
                        (self.J6.y + self.J6.h)),
                       (0,0,255), 1)
        cv2.rectangle(self.image,
                       (self.J12.x, self.J12.y),
                       ((self.J12.x + self.J12.w),
                        (self.J12.y + self.J12.h)),
                       (0,0,255), 1)
        cv2.rectangle(self.image,
                       (self.J14.x, self.J14.y),
                       ((self.J14.x + self.J14.w),
                        (self.J14.y + self.J14.h)),
                       (0,0,255), 1)
        cv2.rectangle(self.image,
                       (self.J13.x, self.J13.y),
                       ((self.J13.x + self.J13.w),
                        (self.J13.y + self.J13.h)),
                       (0,0,255), 1)
        cv2.rectangle(self.image,
                       (self.J2.x, self.J2.y),
                       ((self.J2.x + self.J2.w),
                        (self.J2.y + self.J2.h)),
                       (0,0,255), 1)
        cv2.rectangle(self.image,
                       (self.J3.x, self.J3.y),
                       ((self.J3.x + self.J3.w),
                        (self.J3.y + self.J3.h)),
                       (0,0,255), 1)
        cv2.rectangle(self.image,
                       (self.C59.x, self.C59.y),
                       ((self.C59.x + self.C59.w),
                        (self.C59.y + self.C59.h)),
                       (0,0,255), 1)
        cv2.rectangle(self.image,
                       (self.J4.x, self.J4.y),
                       ((self.J4.x + self.J4.w),
                        (self.J4.y + self.J4.h)),
                       (0,0,255), 1)
