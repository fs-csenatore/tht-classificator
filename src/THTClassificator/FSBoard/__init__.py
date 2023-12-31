import numpy as np
import cv2
from dataclasses import dataclass
import xml.etree.ElementTree as xmlET
import enum
import random
import tflite_runtime.interpreter as tflite
import time
import logging
from datetime import datetime
from THTClassificator.SettingsFile import TFLITESettings

class Boards(enum.Enum):
    MED3_REV100 = 1

@dataclass
class boundingbox:
    ymin: int
    xmin: int
    ymax: int
    xmax: int

class pascal_voc(xmlET.ElementTree):
    def __init__(self, img_name:str):
        super().__init__(xmlET.Element("annotation"))
        
        folder = xmlET.SubElement(self.getroot(), "folder")
        folder.text = "ImageSet"
        
        filename = xmlET.SubElement(self.getroot(), "filename")
        filename.text = img_name

        source = xmlET.SubElement(self.getroot(), "source")
        database = xmlET.SubElement(source, "database")
        database.text = "Unknown"

        size = xmlET.SubElement(self.getroot(), "size")
        width = xmlET.SubElement(size, "width")
        height = xmlET.SubElement(size, "height")
        depth = xmlET.SubElement(size, "depth")
        width.text = str(640)
        height.text = str(640)
        depth.text = str(3)

        segmented = xmlET.SubElement(self.getroot(), "segmented")
        segmented.text = str(0)

    def __xml_create_object_label(self, label: str):
        object = xmlET.SubElement(self.getroot(), "object")
        name = xmlET.SubElement(object, "name")
        name.text = label

        return object

    def xml_add_bbox_label(self, label: str, bbox: boundingbox):
        object = self.__xml_create_object_label(label)
        bndbox = xmlET.SubElement(object, "bndbox")
        xmin = xmlET.SubElement(bndbox, "xmin")
        ymin = xmlET.SubElement(bndbox, "ymin")
        xmax = xmlET.SubElement(bndbox, "xmax")
        ymax = xmlET.SubElement(bndbox, "ymax")
        xmin.text = str(bbox.xmin)
        xmax.text = str(bbox.xmax)
        ymin.text = str(bbox.ymin)
        ymax.text = str(bbox.ymax)

    def write_xml(self, filename: str):
        xmlET.indent(self,space="\t", level=0)
        self.write(filename,encoding="unicode", xml_declaration=True, method="xml")

class object_detection:
    def __init__(self):
        """
        create a interpreter for object detection.
        """
        settings = TFLITESettings()
        if settings.get_delegate() == 1:
            ext_delegate = [tflite.load_delegate('/usr/lib/libethosu_delegate.so')]
        else:
            ext_delegate = None

        self.__interpreter = tflite.Interpreter(
            settings.get_model_path(),
            experimental_delegates=ext_delegate
        )
        self.__interpreter.allocate_tensors()

        self.input_details = self.__interpreter.get_input_details()
        self.output_details = self.__interpreter.get_output_details()

        # Check output layer name to determine if this model was created with TF2 or TF1
        if ('StatefulPartitionedCall' in self.output_details[0]['name']):
            # This is a TF2 model
            self.__boxes_idx, self.__classes_idx, self.__scores_idx, self.__count_idx = 1, 3, 0, 2
        else:
            # This is a TF1 model
            self.__boxes_idx, self.__classes_idx, self.__scores_idx, self.__count_idx = 0, 1, 2, 3

        #define image preprocess function
        if settings.get_input_type() == "float32-VGG":
            self.__preprocess_input_tensor=self.__vgg_input_tensor
        elif settings.get_input_type() == "float32-1":
            self.__preprocess_input_tensor=self.__normalize_input_tensor
        else:
            self.__preprocess_input_tensor=self.__normalize_input_tensor

    def __normalize_input_tensor(self, tensor: np.ndarray):
        return (np.float32(tensor) - 127.5) / 127.5
    
    def __vgg_input_tensor(self , tensor: np.ndarray ):
        if tensor.shape[3] == 3:
            channel_means = [np.float32(123.68) , np.float32(116.779), np.float32(103.939)]
            return tensor.astype(np.float32) - [[channel_means]]
        else:
            return tensor.astype(np.float32)

    def detect_objects(self, image: np.ndarray, threshold=0.0):
        """Returns a list of detection results, each a dictionary of object info."""

        #model_input shape is (1xheightxwidthxdepth) -> we must expand the img data
        input_tensor= np.expand_dims(image, axis=0)

        #Normalize Input when type is float32, else assume uint8.
        if self.input_details[0]['dtype'] == np.float32:
            input_tensor = self.__preprocess_input_tensor(input_tensor)

        start_time = time.time()

        # Perform the actual detection
        self.__interpreter.set_tensor(self.input_details[0]['index'], input_tensor)
        self.__interpreter.invoke()

        end_time = time.time()
        logging.debug("Detection-Runtime in ms =%f",(end_time-start_time)*10**3)

        # Retrieve detection results
        boxes = self.__interpreter.get_tensor(self.output_details[self.__boxes_idx]['index'])[0]
        scores = self.__interpreter.get_tensor(self.output_details[self.__scores_idx]['index'])[0]
        classes = self.__interpreter.get_tensor(self.output_details[self.__classes_idx]['index'])[0]
        count = int(self.__interpreter.get_tensor(self.output_details[self.__count_idx]['index'])[0])

        #Get all detected objects and return the highest score from each class_id
        results = list()
        for i in range(count):
            if scores[i] >= threshold:
                bbox_ymin , bbox_xmin, bbox_ymax , bbox_xmax = boxes[i]
                bndbox = self.__get_absolute_bndbox(bbox_ymin , bbox_xmin, bbox_ymax , bbox_xmax)
                result = {
                    'bounding_box': bndbox,
                    'class_id': classes[i],
                    'score': int(scores[i]*100),
                }
                results.append(result)

        #get only the highest result in label
        results = sorted(results, key=lambda x: x['score'], reverse=True)
        objects = list()
        obj_idx = list()
        for obj in results:
            if obj['class_id'] not in obj_idx:
                obj_idx.append(obj['class_id'])
                objects.append(obj)

        return objects

    def __get_absolute_bndbox(self, ymin: float, xmin: float, ymax: float, xmax: float):
        bndbox = boundingbox(0,0,0,0)
        bndbox.ymin = int(ymin * self.image.shape[0])
        bndbox.xmin = int(xmin * self.image.shape[1])
        bndbox.ymax = int(ymax * self.image.shape[0])
        bndbox.xmax = int(xmax * self.image.shape[1])
        return bndbox

class boards:
    def get_labels(self, label_map_file: str):
        "create a dict with lables and ids from a tflite_label_map"
        with open(label_map_file, 'r') as file:
            lines = file.readlines()

        self.detection_labels = dict()
        for index, line in enumerate(lines):
            line = line.strip()
            self.detection_labels[index] = line
        
        #define a color for each label
        #Needed for debug purpose
        self.labels_color = list()
        for _ in self.detection_labels:
            self.labels_color.append(self.__random_color())
        
    def __random_color(self):
        r = random.randint(120, 255)
        g = random.randint(0, 100)
        b = random.randint(120, 255)
        return (r, g, b)
    
    def __good_color(self):
        r = 0
        g = 255
        b = 0
        return (r, g ,b)

    def __faulty_color(self):
        r = 255
        g = 0
        b = 0
        return (r, g, b)

    def set_img(self, img: np.ndarray):
        self.image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.result_image = self.image.copy()
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
        self.img_name = self.board_name + '_' + timestamp + ".jpg"
        self.xml_name = self.board_name + '_' + timestamp + ".xml"

        self.voc = pascal_voc(self.img_name)

    def get_img(self):
        return cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)

    def get_result_image(self):
        return cv2.cvtColor(self.result_image, cv2.COLOR_RGB2BGR)
    
    def __draw_rectangle(self, bndbox: boundingbox, color: tuple):
        ymin = bndbox.ymin
        xmin = bndbox.xmin
        ymax = bndbox.ymax
        xmax = bndbox.xmax
        cv2.rectangle(self.result_image, (xmin, ymin), (xmax, ymax), color, 2)

    def __draw_text(self, bndbox: boundingbox, label: str, label_score: int, color: tuple):
        # Put label next to bndbox
        y = bndbox.ymin - 5 if bndbox.ymin - 5 > 5 else bndbox.ymin + 5

        label = "{}: {:0d}%".format(label, int(label_score))
        cv2.putText(self.result_image, label, (bndbox.xmin, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def __draw_object(self, bndbox: boundingbox, label: str, score: int, color: tuple):
        self.__draw_rectangle(bndbox, color)
        self.__draw_text(bndbox, label, score, color)
        self.voc.xml_add_bbox_label(label, bndbox)
    
    def draw_good_object(self, object: dict):
        bndbox = object['bounding_box']
        label = self.detection_labels[object['class_id']]
        score = object['score']
        self.__draw_object(bndbox, label, score, self.__good_color())

    def draw_faulty_object(self, object: dict):
        bndbox = object['bounding_box']
        label = self.detection_labels[object['class_id']]
        score = object['score']
        self.__draw_object(bndbox, label, score,self.__faulty_color())
    
    def draw_undef_object(self, object: dict):
        bndbox = object['bounding_box']
        label: str = self.detection_labels[object['class_id']]
        label = label[:label.rfind('_')]
        label = label + "_undefined"
        self.__draw_object(bndbox, label, 0.0,self.__faulty_color())
        

class MED3_rev100(object_detection, boards):

    def __init__(self):
        super().__init__()
        self.board_name = "MED3-rev1.00"
        #Get defined labels from tflite label map
        settings = TFLITESettings()
        self.get_labels(settings.get_label_path())

        #Group labels according to reference identifier
        self.__J6_label_ids = list()
        self.__J12_label_ids = list()
        self.__J14_label_ids = list()
        self.__J11_label_ids = list()
        self.__J10_label_ids = list()
        self.__J09_label_ids = list()
        self.__J3_label_ids = list()
        self.__J2_label_ids = list()
        self.__C59_label_ids = list()
        self.__J4_label_ids = list()
        self.__J13_label_ids = list()
        self.__MED3_label_ids = list()

        for id, label in self.detection_labels.items():
            if label.startswith('J6'):
                self.__J6_label_ids.append(id)
            elif label.startswith('J12'):
                self.__J12_label_ids.append(id)
            elif label.startswith('J14'):
                self.__J14_label_ids.append(id)
            elif label.startswith('J11'):
                self.__J11_label_ids.append(id)
            elif label.startswith('J10'):
                self.__J10_label_ids.append(id)
            elif label.startswith('J09'):
                self.__J09_label_ids.append(id)
            elif label.startswith('J3'):
                self.__J3_label_ids.append(id)
            elif label.startswith('J2'):
                self.__J2_label_ids.append(id)
            elif label.startswith('C59'):
                self.__C59_label_ids.append(id)
            elif label.startswith('J4'):
                self.__J4_label_ids.append(id)
            elif label.startswith('J13'):
                self.__J13_label_ids.append(id)
            elif label.startswith('MED3'):
                self.__MED3_label_ids.append(id)

    def get_results(self):

        detected_objects = self.detect_objects(self.image)
        
        J6_objs = list()
        J12_objs = list()
        J14_objs = list()
        J11_objs = list()
        J10_objs = list()
        J09_objs = list()
        J3_objs = list()
        J2_objs = list()
        C59_objs = list()
        J4_objs = list()
        J13_objs = list()
        med3_objs = list()

        self.board_dict = {"J6": J6_objs,
                      "J12": J12_objs,
                      "J14": J14_objs,
                      "J11": J11_objs,
                      "J10": J10_objs,
                      "J09": J09_objs,
                      "J3": J3_objs,
                      "J2": J2_objs,
                      "C59": C59_objs,
                      "J4": J4_objs,
                      "J13": J13_objs,
                      "MED3": med3_objs}

        #Sort Object according to reference identifier
        for obj in detected_objects:
            skip = False
            for id in self.__J6_label_ids:
                if obj['class_id'] == id:
                    J6_objs.append(obj)
                    skip = True
                    break
            
            if skip:
                continue

            for id in self.__J12_label_ids:
                if obj['class_id'] == id:
                    J12_objs.append(obj)
                    skip = True
                    break
            
            if skip:
                continue

            for id in self.__J14_label_ids:
                if obj['class_id'] == id:
                    J14_objs.append(obj)
                    skip = True
                    break
            
            if skip:
                continue

            for id in self.__J11_label_ids:
                if obj['class_id'] == id:
                    J11_objs.append(obj)
                    skip = True
                    break
            
            if skip:
                continue

            for id in self.__J10_label_ids:
                if obj['class_id'] == id:
                    J10_objs.append(obj)
                    skip = True
                    break
            
            if skip:
                continue

            for id in self.__J09_label_ids:
                if obj['class_id'] == id:
                    J09_objs.append(obj)
                    skip = True
                    break
            
            if skip:
                continue

            for id in self.__J3_label_ids:
                if obj['class_id'] == id:
                    J3_objs.append(obj)
                    skip = True
                    break

            if skip:
                continue

            for id in self.__J2_label_ids:
                if obj['class_id'] == id:
                    J2_objs.append(obj)
                    skip = True
                    break

            if skip:
                continue

            for id in self.__C59_label_ids:
                if obj['class_id'] == id:
                    C59_objs.append(obj)
                    skip = True
                    break

            if skip:
                continue

            for id in self.__J4_label_ids:
                if obj['class_id'] == id:
                    J4_objs.append(obj)
                    skip = True
                    break

            if skip:
                continue

            for id in self.__J13_label_ids:
                if obj['class_id'] == id:
                    J13_objs.append(obj)
                    skip = True
                    break
            if skip:
                continue

            for id in self.__MED3_label_ids:
                if obj['class_id'] == id:
                    med3_objs.append(obj)
                    skip = True
                    break

        #Determine if a board is OK or if faults have been detected.
        for key, ref_objs in self.board_dict.items():

            #Check if we have Labels with similar scores
            #These labels are likely to contain incorrect classifications.
            if len(ref_objs) > 1:
                max_score = 0.0
                sndmax_score = 0.0
                
                for obj in ref_objs:
                    if obj['score'] >= max_score:
                        sndmax_score = max_score
                        max_score = obj['score']

                    diff_score = max_score - sndmax_score
                
                if diff_score < 30:
                    self.draw_undef_object(obj)
                elif self.detection_labels[obj['class_id']].endswith('_ok'):
                    self.draw_good_object(obj)
                else:
                    self.draw_faulty_object(obj)

            elif len(ref_objs) == 1:
                if self.detection_labels[ref_objs[0]['class_id']].endswith('_ok') \
                            or self.detection_labels[ref_objs[0]['class_id']].startswith("MED3"):
                    self.draw_good_object(ref_objs[0])
                else:
                    self.draw_faulty_object(ref_objs[0])
            else:
                msg = key + " not found"
                logging.debug(msg)

    def make_screenshot(self, path: str):
        cv2.imwrite(path + '/' + self.img_name, cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR))
        self.voc.write_xml(path + '/' + self.xml_name)

