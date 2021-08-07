import configparser 
import numpy as np
import json

config = configparser.ConfigParser()
config.read("./config.cfg")

INPUT_DIR = config["SOURCE_SINK"]["INPUT_DIR"]
TYPE = config["SOURCE_SINK"]["TYPE"]
OUTPUT_DIR = config["SOURCE_SINK"]["OUTPUT_DIR"]

YOLO_DETECTION_CONFIG_PATH = config["YOLO_DETECTION"]["CONFIG_PATH"]
YOLO_DETECTION_WEIGHT_PATH = config["YOLO_DETECTION"]["WEIGHT_PATH"]
YOLO_DETECTION_DATA_FILE_PATH = config["YOLO_DETECTION"]["DATA_FILE_PATH"]
YOLO_MODEL_VERSION = config["YOLO_DETECTION"]["MODEL_VERSION"]


YOLO_DETECTION_THRESHOLD = float(config["ALGO_PARAMS"]["THRESHOLD"])
YOLO_DETECTION_BATCH_SIZE = int(config["ALGO_PARAMS"]["BATCH_SIZE"])
YOLO_DETECTION_SAVE_LABELS = int(config["ALGO_PARAMS"]["SAVE_LABELS"])
YOLO_DETECTION_DONT_SHOW = int(config["ALGO_PARAMS"]["DONT_SHOW"])
YOLO_DETECTION_EXT_OUTPUT = int(config["ALGO_PARAMS"]["EXT_OUTPUT"])

IOU_CALCULATION_FIELDS = config["IOU_CALCULATION"]["FIELDS"]
IOU_CALCULATION_FIELDS = json.loads(IOU_CALCULATION_FIELDS)
IOU_CALCULATION_THRESHOLD = float(config["IOU_CALCULATION"]["IOU_THRESHOLD"])
VARY_IOU_THRESHOLD = config["IOU_CALCULATION"]["VARY_IOU_THRESHOLD"]
SAVE_ANNOTATIONS = config["IOU_CALCULATION"]["SAVE_ANNOTATIONS"]


