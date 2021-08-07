import time
import cv2
import numpy as np
import os
import glob
import xml.etree.ElementTree as ET
import math
from config_parser import *
from sklearn.utils.linear_assignment_ import linear_assignment

class data_class(object):

    hw_mp_consolidated_metric= {}

    def __init__(self, Id, frame,img_path,iou_fields,nw_detections,excel_dir,iou_thr,annotation_path,network_w_h,Image_org,Image_mod):
        self.Id = int(Id)
        self.Frame = frame
        self.Img_name= ""
        self.Img_path = img_path
        self.Predicted_filtered = []
        self.Nw_io_shape = network_w_h 
        self.Predicted_detections = nw_detections
        self.Iou_fields = iou_fields
        self.Iou_threshold = iou_thr       
        self.Gt_img_shape = ""
        self.Gt_coordinates = []
        self.Gt_Filtered = []
        self.Xml_path = ""
        self.Excel_dir = excel_dir
        self.Iou = []
        self.temp_metric=[]
        self.total_detection_count = {}
        self.total_gt_count = {}
        self.annotation_path = annotation_path
        self.new_iou_field = iou_fields + ["hw","mp"]
        self.hw_mp_metric = {}
        self.temp_arg = 0
        self.Org_img = Image_org
        self.Square_img = Image_mod

    def read_xml(self):
        string_split = self.Img_path.split("/")
        image_name = string_split[(len(string_split)-1)].split(".")
        self.Img_name = image_name
        replace_value = str(string_split[(len(string_split)-1)])
        temp_path =(self.Img_path).replace(replace_value,"")
        self.xml_path = glob.glob(os.path.join(temp_path,"xml",str(image_name[0])+".xml"))
        content = self.read_content(str(self.xml_path[0]))
        self.Gt_coordinates = content
        img_gt_size = content[2:]
        self.Gt_img_shape = img_gt_size
        gt_field_coordinates = {}
        for x in range(len(content[1])):
            for i in range(len(self.Iou_fields)):
                temp_list = []
                field_name = content[1][x][0]
                if field_name == self.Iou_fields[i]:
                    field_coordinates = content[1][x] 
                    temp_list.append(field_coordinates[1:])
                    gt_field_coordinates[str(field_name)] = temp_list

                # if str(field_name) == "hw" or str(field_name) == "mp":
                #     try:
                #         #[filed_name, xmin, ymin, xmax, ymax]
                #         Image = self.Org_img
                #         crop_img = Image[field_coordinates[2]:field_coordinates[4], field_coordinates[1]:field_coordinates[3]]
                #         save_path = "./cropped_img/"+str(field_name)+"/"+str(time.time())+".jpeg"
                #         cv2.imwrite(save_path,crop_img)
                #     except:
                #         pass

                if str(field_name) == "hw" or str(field_name) == "mp":
                    field_coordinates = content[1][x] 
                    temp_list.append(field_coordinates[1:])
                    if str(field_name) in gt_field_coordinates.keys():
                        gt_field_coordinates[str(field_name)].append(field_coordinates[1:])
                    else:
                        gt_field_coordinates[str(field_name)] = temp_list
                    break

        self.Gt_Filtered = gt_field_coordinates 
        
    def read_content(self,xml_file: str):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        list_with_all_boxes = []
        size = root.find('size')
        width = size.find('width').text
        height = size.find('height').text
        for boxes in root.iter('object'):
            filename = root.find('filename').text
            filed_name, ymin, xmin, ymax, xmax = None, None, None, None, None
            ymin = int(boxes.find("bndbox/ymin").text)
            xmin = int(boxes.find("bndbox/xmin").text)
            ymax = int(boxes.find("bndbox/ymax").text)
            xmax = int(boxes.find("bndbox/xmax").text)
            filed_name = str(boxes.find("name").text)
            list_with_single_boxes = [filed_name, xmin, ymin, xmax, ymax]
            list_with_all_boxes.append(list_with_single_boxes)
        return filename, list_with_all_boxes, width, height
    
    def convertBack(self,x, y, w, h):
        xmin = int(round(x - (w / 2)))
        xmax = int(round(x + (w / 2)))
        ymin = int(round(y - (h / 2)))
        ymax = int(round(y + (h / 2)))
        return xmin, ymin, xmax, ymax
    
    def filter_detections(self):
        predicted_field_coordinates = {}
        # delta_y = float(int(self.Gt_img_shape[1])/int(self.Nw_io_shape[1]))
        # delta_x = float(int(self.Gt_img_shape[0])/int(self.Nw_io_shape[0]))
        delta_y = float(int(self.Square_img.shape[1])/int(self.Nw_io_shape[1]))
        delta_x = float(int(self.Square_img.shape[0])/int(self.Nw_io_shape[0]))
        for i in range(len(self.Predicted_detections)):
            field_name = self.Predicted_detections[i][0]
            for x in range(len(self.Iou_fields)):
                if field_name == self.Iou_fields[x]:
                    field_coordinates = []
                    temp_list=[]
                    field_coordinates_1 =self.Predicted_detections[i][2]  
                    xmin, ymin, xmax, ymax = self.convertBack(float(field_coordinates_1[0]), float(field_coordinates_1[1]), float(field_coordinates_1[2]), float(field_coordinates_1[3]))
                    field_coordinates = [int(xmin*delta_x), int(ymin*delta_y), int(xmax*delta_x), int(ymax*delta_y)]
                    if str(field_name) in predicted_field_coordinates.keys():
                        
                        predicted_field_coordinates[str(field_name)].append(field_coordinates)
                    else:
                        temp_list.append(field_coordinates)
                        predicted_field_coordinates[str(field_name)] = temp_list 


                if field_name == "mp" or field_name == "hw":
                    field_coordinates = []
                    temp_list=[]
                    field_coordinates_1 =self.Predicted_detections[i][2]  
                    xmin, ymin, xmax, ymax = self.convertBack(float(field_coordinates_1[0]), float(field_coordinates_1[1]), float(field_coordinates_1[2]), float(field_coordinates_1[3]))
                    field_coordinates = [int(xmin*delta_x), int(ymin*delta_y), int(xmax*delta_x), int(ymax*delta_y)]
                    if str(field_name) in predicted_field_coordinates.keys():
                        
                        predicted_field_coordinates[str(field_name)].append(field_coordinates)
                    else:
                        temp_list.append(field_coordinates)
                        predicted_field_coordinates[str(field_name)] = temp_list
                    break

        self.Predicted_filtered = predicted_field_coordinates

    def bb_intersection_over_union(self,boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou

    def total_count_calc(self):
        for i in range(len(self.new_iou_field)):
            if self.new_iou_field[i] in self.Predicted_filtered.keys():
                if self.new_iou_field[i] in self.total_detection_count.keys():
                    self.total_detection_count[self.new_iou_field[i]] += len(self.Predicted_filtered[self.new_iou_field[i]])
                else:
                    self.total_detection_count[self.new_iou_field[i]]  = len(self.Predicted_filtered[self.new_iou_field[i]])

            if self.new_iou_field[i] in self.Gt_Filtered.keys():
                if self.new_iou_field[i] in self.total_gt_count.keys():
                    self.total_gt_count[self.new_iou_field[i]] += len(self.Gt_Filtered[self.new_iou_field[i]])
                else:
                    self.total_gt_count[self.new_iou_field[i]]  = len(self.Gt_Filtered[self.new_iou_field[i]])

    def calculate_metric(self):
        for i in range(len(self.Iou_fields)):
            temp_metric_list=[0,0,0,0] #tp fp tn fn
            if self.Iou_fields[i] in self.Predicted_filtered.keys() and self.Iou_fields[i] in self.Gt_Filtered.keys() :
                if len(self.Gt_Filtered[self.Iou_fields[i]]) == len(self.Predicted_filtered[self.Iou_fields[i]]):
                    iou=self.bb_intersection_over_union(self.Gt_Filtered [self.Iou_fields[i]][0],self.Predicted_filtered[self.Iou_fields[i]][0])
                    # self.Iou.append(iou)
                    if iou >= self.Iou_threshold:
                        temp_metric_list[0] = int(temp_metric_list[0]) +1 #+tp   
                        self.Iou.append(iou)                    
                    else:
                        temp_metric_list[1] = int(temp_metric_list[1])+1 #+fp
                        temp_metric_list[3] = int(temp_metric_list[3])+1 #+fn
                        self.Iou.append(0)

                elif len(self.Gt_Filtered[self.Iou_fields[i]]) < len(self.Predicted_filtered[self.Iou_fields[i]]):   
                    temp_iou = []                    
                    for x in range(len(self.Predicted_filtered[self.Iou_fields[i]])):
                        iou=self.bb_intersection_over_union(self.Gt_Filtered [self.Iou_fields[i]][0],self.Predicted_filtered[self.Iou_fields[i]][x])
                        temp_iou.append(iou)
                    max_iou = max(temp_iou)
                    if max_iou >= self.Iou_threshold:
                        temp_metric_list[0] = int(temp_metric_list[0])+1 #+tp
                        temp_metric_list[1] = int(temp_metric_list[1])+ int((len(self.Predicted_filtered[self.Iou_fields[i]])-1)) #fp + (len(pred)-1)
                        self.Iou.append(max_iou)
                    else:
                        temp_metric_list[1] = int(temp_metric_list[1])+int(len(self.Predicted_filtered[self.Iou_fields[i]])) #+fp
                        temp_metric_list[3] = int(temp_metric_list[3])+1 #+fn
                        self.Iou.append(0)
                    # self.Iou.append(max_iou)
            elif self.Iou_fields[i] not in self.Predicted_filtered.keys() and self.Iou_fields[i] not in self.Gt_Filtered.keys():
                self.Iou.append(0)
                temp_metric_list[2] = int(temp_metric_list[2])+1  #+tn            
            elif self.Iou_fields[i] in self.Predicted_filtered.keys() and self.Iou_fields[i] not in self.Gt_Filtered.keys() :
                self.Iou.append(0)
                for x in range(len(self.Predicted_filtered[self.Iou_fields[i]])):
                    temp_metric_list[1] = int(temp_metric_list[1])+len(self.Predicted_filtered[self.Iou_fields[i]]) #fp + len(pred)
            
            elif self.Iou_fields[i] not in self.Predicted_filtered.keys() and self.Iou_fields[i] in self.Gt_Filtered.keys():
                self.Iou.append(0)
                temp_metric_list[3] = int(temp_metric_list[3])+1  #+fn
            self.temp_metric.append(temp_metric_list)
    
    def save_annotated(self):
        delta_y = float(int(self.Square_img.shape[1])/int(self.Nw_io_shape[1]))
        delta_x = float(int(self.Square_img.shape[0])/int(self.Nw_io_shape[0]))
        #dispImage = cv2.imread(self.Img_path) 
        dispImage = self.Org_img
        for detection in self.Predicted_detections:
            cx, cy, w, h =  detection[2][0], detection[2][1], detection[2][2], detection[2][3]
            if detection[0] == "payeename" or detection[0] == "payerdetails" or detection[0] == "checknumber" or detection[0] == "lar" or detection[0] == "car" or detection[0] == "memo" or detection[0] == "date" or detection[0] == "bankdetails" or detection[0] == "micr":
                afterPoint = 2
                cx=math.floor(cx * 10 ** afterPoint) / 10 ** afterPoint
                cy=math.floor(cy * 10 ** afterPoint) / 10 ** afterPoint
                w=math.floor(w * 10 ** afterPoint) / 10 ** afterPoint
                h=math.floor(h * 10 ** afterPoint) / 10 ** afterPoint
                xmin = int(round(cx - (w / 2)) * delta_x)
                xmax = int(round(cx + (w / 2)) * delta_x)
                ymin = int(round(cy - (h / 2)) * delta_y)
                ymax = int(round(cy + (h / 2)) * delta_y)
                pt1 = (xmin, ymin)
                pt2 = (xmax, ymax)
                
                cv2.putText(dispImage,detection[0],pt1,cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 0),1,cv2.LINE_AA )
                dispImage = cv2.rectangle(dispImage, pt1, pt2, (0, 0, 255), 1)

        annotation_img_path = self.annotation_path+"/"+ self.Img_name[0]+"."+self.Img_name[1]
        cv2.imwrite(annotation_img_path,dispImage)

    def association(self,iou_matrix, iou_threshold):
        matched_indices = linear_assignment(-iou_matrix) #minimization problem

        unmatched_detections = []
        for d in range(0, iou_matrix.shape[0]):
            if d not in matched_indices[:, 0]:
                unmatched_detections.append(d)
        unmatched_trackers = []
        for t in range(0, iou_matrix.shape[1]):
            if t not in matched_indices[:, 1]:
                unmatched_trackers.append(t)
        # filter out matched with low IOU
        matches = []
        for m in matched_indices:
            if iou_matrix[m[0], m[1]] < iou_threshold:
                unmatched_detections.append(m[0])
                unmatched_trackers.append(m[1])
            else:
                matches.append(m.reshape(1, 2))
        if len(matches) == 0:
            matches = np.empty((0, 2), dtype=int)
        else:
            matches = np.concatenate(matches, axis=0)
        return matches, np.array(unmatched_trackers), np.array(unmatched_detections)
    
    def hw_mp_analysis(self):
        fields = ["hw","mp"]
        for i in range (len(fields)):
            temp_dict ={}
            temp_list =[]
            temp_metric_list=[0,0,0,0] #tp fp tn fn
            if fields[i] not in self.Predicted_filtered.keys() or fields[i] not in self.Gt_Filtered.keys(): 
                if fields[i] not in self.Predicted_filtered.keys() and fields[i] not in self.Gt_Filtered.keys():
                    temp_metric_list[2] +=1 #tn
                elif fields[i] not in self.Predicted_filtered.keys() and  len(self.Gt_Filtered[fields[i]]) != 0:
                    temp_metric_list[3] +=int(len(self.Gt_Filtered[fields[i]])) #fn
                elif len(self.Predicted_filtered[fields[i]]) != 0 and  fields[i] not in self.Gt_Filtered.keys():
                    temp_metric_list[1] +=int(len(self.Predicted_filtered[fields[i]])) #fp
            if fields[i]  in self.Predicted_filtered.keys() and fields[i] in self.Gt_Filtered.keys(): 
                iou_matrix = np.zeros((len(self.Gt_Filtered[fields[i]]),len(self.Predicted_filtered[fields[i]])),'float32')

                for j in range (len(self.Gt_Filtered[fields[i]])):
                    for k in range(len(self.Predicted_filtered[fields[i]])):
                        iou_matrix[j,k] = self.bb_intersection_over_union(self.Gt_Filtered[fields[i]][j],self.Predicted_filtered[fields[i]][k])
                
                matches, not_match_est, not_match_gt = self.association(iou_matrix, self.Iou_threshold)

                if len(matches) !=0:
                    temp_metric_list[0] += len(matches)   #tp fp tn fn
                if len(not_match_est) != 0:
                    temp_metric_list[1] += len(not_match_est)
                if len(not_match_gt)!= 0:
                    temp_metric_list[3] += len(not_match_gt)

                for p in range (len(matches)):
                    temp_list.append(iou_matrix[matches[p][0],matches[p][1]])

            temp_dict["metric"] = temp_metric_list
            temp_dict["iou"] = temp_list

            self.hw_mp_metric[fields[i]] = temp_dict

            if fields[i] not in data_class.hw_mp_consolidated_metric.keys():

                data_class.hw_mp_consolidated_metric[fields[i]] = temp_dict
            else:
                if "metric" in data_class.hw_mp_consolidated_metric[fields[i]].keys():
                    res_list = [sum(x) for x in zip(data_class.hw_mp_consolidated_metric[fields[i]]["metric"],temp_metric_list)]
                    data_class.hw_mp_consolidated_metric[fields[i]]["metric"] = res_list

                else:
                    data_class.hw_mp_consolidated_metric[fields[i]]["metric"] = temp_metric_list

                if "iou" in data_class.hw_mp_consolidated_metric[fields[i]].keys():
                    data_class.hw_mp_consolidated_metric[fields[i]]["iou"] += temp_list

                else:
                    data_class.hw_mp_consolidated_metric[fields[i]]["iou"] = temp_list

    def clean_dict(self):
        data_class.hw_mp_consolidated_metric = {}

    def run_post_processing(self):
        self.read_xml()
        self.filter_detections()
        self.calculate_metric()
        self.total_count_calc()
        self.hw_mp_analysis()

        if SAVE_ANNOTATIONS.lower() == "yes":
            try:
                self.save_annotated()
            except Exception as e:
                print("Save annotation error")

        
        
        
    