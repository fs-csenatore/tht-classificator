import numpy as np
import cv2
from dataclasses import dataclass
import xml.etree.ElementTree as xmlET
import enum
import random

class Boards(enum.Enum):
    MED3_REV100 = 1

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

def get_labels(label_map_path: str):
    "returns a dict with lables and ids from a tflite_label"
    with open(label_map_path, 'r') as file:
        lines = file.readlines()

    label_dict = {}
    for index, line in enumerate(lines):
        line = line.strip()
        label_dict[index] = line
    
    return label_dict

def random_color():
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    return (r, g, b)

class MED3_rev100():

    def __init__(self, label_map_path: str):
        self.detection_labels = get_labels(label_map_path)
        self.labels_color = list()
        for _ in self.detection_labels:
            self.labels_color.append(random_color())

    def set_img(self, med3_img: np.ndarray):
        self.image = med3_img.copy()
