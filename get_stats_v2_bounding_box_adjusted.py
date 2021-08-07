import os
import numpy as np
from sklearn.utils.linear_assignment_ import linear_assignment
import pickle
import xlwt
from xlwt import Workbook
import argparse
import pandas as pd

def iou(bb_test, bb_gt):
    # Computes IUO between two bboxes in the form [x1,y1,x2,y2]
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
              + (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1]) - wh)
    return o

def association(iou_matrix, iou_threshold):
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


def compute_stats(test_dataset_folder,iou_threshold = 0.5, pkl_filename = 'stats.pkl', excel_filename = 'stats.xlsx'):

    print("Test set directory name is", test_dataset_folder)
    print("iou threshold for a detection is ", iou_threshold)
    print("pkl file name is ", pkl_filename)
    print("Excel file name is ", excel_filename)
    # exit()

    date_format = xlwt.XFStyle()
    date_format.num_format_str = 'dd/mm/yyyy'
    style1 = "font: color black; font: bold 1; align: wrap 1"
    style2 = "font: color black; align: wrap 1"
    style3 = "font: color black; align: wrap 1;pattern: pattern solid, fore_color red;"
    style4 = "font: color black;  font: bold 1; align: wrap 1;pattern: pattern solid, fore_color yellow;"

    style10 = "font: color red; font: height 320;font: bold 1; align: wrap 1"
    # Workbook is created
    wb = Workbook()
    # add_sheet is used to create sheet.
    Stats = wb.add_sheet('Stats')
    Stats.write(0,0, "Total images", xlwt.Style.easyxf(style1))
    Stats.write(1,0, "Total mask labels", xlwt.Style.easyxf(style1))
    Stats.write(2,0, "Total no mask labels", xlwt.Style.easyxf(style1))
    Stats.write(3,0, "Total no face labels", xlwt.Style.easyxf(style1))
    Stats.write(4,0, "Total labels", xlwt.Style.easyxf(style1))

    Stats.write(6,0, "IoU threshold", xlwt.Style.easyxf(style1))
    Stats.write(7, 0, "Mask TP", xlwt.Style.easyxf(style1))
    Stats.write(8, 0, "No mask TP", xlwt.Style.easyxf(style1))
    Stats.write(9, 0, "TN", xlwt.Style.easyxf(style1))

    Stats.write(11, 0, "IoU threshold", xlwt.Style.easyxf(style1))
    Stats.write(12, 0, "Mask FP", xlwt.Style.easyxf(style1))
    Stats.write(13, 0, "No mask FP", xlwt.Style.easyxf(style1))
    Stats.write(14, 0, "Mask FN", xlwt.Style.easyxf(style1))
    Stats.write(15, 0, "No mask FN", xlwt.Style.easyxf(style1))
    Stats.write(16, 0, "Mask MC", xlwt.Style.easyxf(style1))
    Stats.write(17, 0, "No mask MC", xlwt.Style.easyxf(style1))

    Stats.write(19, 0, "IoU threshold", xlwt.Style.easyxf(style1))
    Stats.write(20, 0, "Accuracy", xlwt.Style.easyxf(style4))

    Stats.write(22, 0, "Algo threshold", xlwt.Style.easyxf(style1))

    Stats.write(2, 5, "Mask", xlwt.Style.easyxf(style1))
    Stats.write(2, 6, "No Mask", xlwt.Style.easyxf(style1))
    Stats.write(2, 7, "No Face", xlwt.Style.easyxf(style1))
    Stats.write(3, 4, "Mask", xlwt.Style.easyxf(style1))
    Stats.write(4, 4, "No Mask", xlwt.Style.easyxf(style1))
    Stats.write(5, 4, "No Face", xlwt.Style.easyxf(style1))

    # Get images names
    gt_folder_path = test_dataset_folder + 'input/ground-truth/'
    all_filenames = []
    for (dirpath, dirnames, filenames) in os.walk(gt_folder_path):
        all_filenames += [os.path.join(dirpath, file) for file in filenames]
    # print(all_filenames)
    est_folder_path = test_dataset_folder + 'input/detection-results/'
    mask_tp = 0
    no_mask_tp = 0
    mask_mc = 0
    no_mask_mc = 0
    mask_fp = 0
    no_mask_fp = 0
    mask_fn = 0
    no_mask_fn = 0
    true_neg = 0

    mask_tp_list = []
    no_mask_tp_list = []
    mask_mc_list = []
    no_mask_mc_list = []
    mask_fp_list = []
    no_mask_fp_list = []
    mask_fn_list = []
    no_mask_fn_list = []
    true_neg_list = []

    total_number_images = 0
    total_mask_labels = 0
    total_no_mask_labels = 0
    total_no_face_labels = 0
    for file in all_filenames:
        total_number_images += 1
        gt_filename = file
        # print(gt_filename)
        gt_txt_filename = gt_filename
        gt_filename_only = gt_filename.split("/")[-1]
        est_txt_filename = est_folder_path + gt_filename_only
        gt_txt = open(gt_txt_filename)
        est_txt = open(est_txt_filename)
        no_of_gt_detections = sum(1 for line in gt_txt)
        no_of_est_detections = sum(1 for line in est_txt)
        gt_txt.seek(0)
        est_txt.seek(0)
        iou_matrix = np.zeros((no_of_gt_detections,no_of_est_detections),'float32')
        # print(iou_matrix)
        class_match_matrix = 100*np.ones((no_of_gt_detections, no_of_est_detections), 'float32')
        gt_class_labels = []
        # print(class_match_matrix)
        for i in range(0, no_of_gt_detections):
            gt_labels = gt_txt.readline().split(" ")
            gt_class = gt_labels[0]
            gt_class_labels.append(gt_class)
            gt_box = gt_labels[1:5]
            gt_x1, gt_y1, gt_x2, gt_y2 = int(gt_box[0]), int(gt_box[1]), int(gt_box[2]), int(gt_box[3])
            gt_box_coord = [gt_x1, gt_y1, gt_x2, gt_y2]
            est_txt.seek(0)
            est_class_labels = []
            for j in range(0, no_of_est_detections):
                est_labels = est_txt.readline().split(" ")
                est_class = est_labels[0]
                est_class_labels.append(est_class)
                est_box = est_labels[2:6]
                est_x1, est_y1, est_x2, est_y2 = int(est_box[0]), int(est_box[1]), int(est_box[2]), int(est_box[3])
                est_box_coord = [est_x1, est_y1, est_x2, est_y2]
                iou_boxes = iou(gt_box_coord, est_box_coord)
                iou_matrix[i, j] = iou_boxes
                if gt_class == est_class:
                    class_match_matrix[i, j] = 1
                else:
                    class_match_matrix[i, j] = 0
        for i in range(0,len(gt_class_labels)):
            if gt_class_labels[i] == 'mask':
                total_mask_labels += 1
            elif gt_class_labels[i] == 'nomask':
                total_no_mask_labels += 1
        if len(gt_class_labels) == 0:
            total_no_face_labels += 1

        # print(iou_matrix)
        # print(class_match_matrix)
        gt_txt.seek(0)
        est_txt.seek(0)
        if no_of_gt_detections == 0 and no_of_est_detections !=0:
            for k in range(0,no_of_est_detections):
                est_labels = est_txt.readline().split(" ")
                est_class = est_labels[0]
                if est_class == 'mask':
                    mask_fp += 1
                    mask_fp_list.append(gt_filename)
                if est_class == 'nomask':
                    no_mask_fp += 1
                    no_mask_fp_list.append(gt_filename)
        gt_txt.seek(0)
        est_txt.seek(0)


        if no_of_gt_detections != 0 and no_of_est_detections == 0:
            for k in range(0,no_of_gt_detections):
                gt_labels = gt_txt.readline().split(" ")
                gt_class = gt_labels[0]
                if gt_class == 'mask':
                    mask_fn += 1
                    mask_fn_list.append(gt_filename)
                if gt_class == 'nomask':
                    no_mask_fn += 1
                    no_mask_fn_list.append(gt_filename)
        gt_txt.seek(0)
        est_txt.seek(0)
        if no_of_gt_detections == 0 and no_of_est_detections == 0:
            true_neg += 1
            true_neg_list.append(gt_filename)

        if no_of_gt_detections != 0 and no_of_est_detections != 0:
            matches, not_match_est, not_match_gt = association(iou_matrix, iou_threshold)
            for i, j in matches:
                if class_match_matrix[i][j] == 1:
                    if gt_class_labels[i] == 'mask':
                        mask_tp += 1
                        mask_tp_list.append(gt_filename)
                    elif gt_class_labels[i] == 'nomask':
                        no_mask_tp += 1
                        no_mask_tp_list.append(gt_filename)
                else:
                    if gt_class_labels[i] == 'mask':
                        mask_mc += 1
                        mask_mc_list.append(gt_filename)
                    elif gt_class_labels[i] == 'nomask':
                        no_mask_mc += 1
                        no_mask_mc_list.append(gt_filename)

            for i in not_match_gt:
                if gt_class_labels[i] == 'mask':
                    mask_fn += 1
                    mask_fn_list.append(gt_filename)
                elif gt_class_labels[i] == 'nomask':
                    no_mask_fn += 1
                    no_mask_fn_list.append(gt_filename)
            for j in not_match_est:
                if est_class_labels[j] == 'mask':
                    mask_fp += 1
                    mask_fp_list.append(gt_filename)
                elif est_class_labels[j] == 'nomask':
                    no_mask_fp += 1
                    no_mask_fp_list.append(gt_filename)

    total_number_of_labels = total_mask_labels + total_no_mask_labels + total_no_face_labels
    total_number_of_pred = mask_tp + no_mask_tp + true_neg + mask_fp + no_mask_fp + mask_fn + no_mask_fn + mask_mc + no_mask_mc
    accuracy = (mask_tp + no_mask_tp + true_neg) / total_number_of_pred
    Values = [[mask_tp, mask_mc, mask_fn],[no_mask_mc, no_mask_tp, no_mask_fn],[mask_fp, no_mask_fp, true_neg]] 
    confusion_matrix = pd.DataFrame(Values, columns = ['Mask' , 'No-Mask', 'No-Face'], index = ['Mask', 'No-Mask', 'No-Face'])
    print(confusion_matrix) 
    print("total number of images: ", total_number_images)
    print("total number of  mask labels: ", total_mask_labels)
    print("total number of no mask labels: ", total_no_mask_labels)
    print("total number of no face: ", total_no_face_labels)
    print("total number of labels: ", total_number_of_labels)
    print("accuracy:", accuracy*100)
    print("\n\nmask_tp: ", mask_tp)
    print("no_mask_tp: ", no_mask_tp)
    print("mask_fp: ", mask_fp)
    print("no_mask_fp: ", no_mask_fp)
    print("mask_fn: ", mask_fn)
    print("no_mask_fn: ", no_mask_fn)
    print("mask_mc: ", mask_mc)
    print("no_mask_mc: ", no_mask_mc)
    print("tn: ", true_neg)

    Stats.write(0, 1, total_number_images, xlwt.Style.easyxf(style2))
    Stats.write(1, 1, total_mask_labels, xlwt.Style.easyxf(style2))
    Stats.write(2, 1, total_no_mask_labels, xlwt.Style.easyxf(style2))
    Stats.write(3, 1, total_no_face_labels, xlwt.Style.easyxf(style2))
    Stats.write(4, 1, total_number_of_labels, xlwt.Style.easyxf(style2))

    Stats.write(6, 1, iou_threshold, xlwt.Style.easyxf(style2))
    Stats.write(7, 1, mask_tp, xlwt.Style.easyxf(style2))
    Stats.write(8, 1, no_mask_tp, xlwt.Style.easyxf(style2))
    Stats.write(9, 1, true_neg, xlwt.Style.easyxf(style2))

    Stats.write(11, 1, iou_threshold, xlwt.Style.easyxf(style2))
    Stats.write(12, 1, mask_fp, xlwt.Style.easyxf(style2))
    Stats.write(13, 1, no_mask_fp, xlwt.Style.easyxf(style2))
    Stats.write(14, 1, mask_fn, xlwt.Style.easyxf(style2))
    Stats.write(15, 1, no_mask_fn, xlwt.Style.easyxf(style2))
    Stats.write(16, 1, mask_mc, xlwt.Style.easyxf(style2))
    Stats.write(17, 1, no_mask_mc, xlwt.Style.easyxf(style2))

    Stats.write(19, 1, iou_threshold, xlwt.Style.easyxf(style2))
    Stats.write(20, 1, accuracy*100, xlwt.Style.easyxf(style4))

    Stats.write(3, 5, mask_tp, xlwt.Style.easyxf(style2))
    Stats.write(3, 6, mask_mc, xlwt.Style.easyxf(style2))
    Stats.write(3, 7, mask_fn, xlwt.Style.easyxf(style2))
    Stats.write(4, 5, no_mask_mc, xlwt.Style.easyxf(style2))
    Stats.write(4, 6, no_mask_tp, xlwt.Style.easyxf(style2))
    Stats.write(4, 7, no_mask_fn, xlwt.Style.easyxf(style2))
    Stats.write(5, 5, mask_fp, xlwt.Style.easyxf(style2))
    Stats.write(5, 6, no_mask_fp, xlwt.Style.easyxf(style2))
    Stats.write(5, 7, true_neg, xlwt.Style.easyxf(style2))


    #save_parameters = [iou_threshold, test_dataset_folder, total_number_images, total_mask_labels, total_no_mask_labels, total_no_face_labels, total_number_of_labels, accuracy, mask_tp, no_mask_tp, mask_mc, no_mask_mc, mask_fp, no_mask_fp, mask_fn, no_mask_fn, true_neg, mask_tp_list, no_mask_tp_list, mask_mc_list, no_mask_mc_list, mask_fp_list, no_mask_fp_list, mask_fn_list, no_mask_fn_list, true_neg_list]

    save_parameters = {'iou_threshold': iou_threshold, 'test_dataset_folder': test_dataset_folder,
                            'total_number_images': total_number_images, 'total_mask_labels': total_mask_labels,
                            'total_no_mask_labels': total_no_mask_labels, 'total_no_face_labels': total_no_face_labels,
                            'total_number_of_labels': total_number_of_labels, 'accuracy': accuracy, 'mask_tp': mask_tp,
                            'no_mask_tp': no_mask_tp, 'mask_mc': mask_mc, 'no_mask_mc': no_mask_mc, 'mask_fp': mask_fp,
                            'no_mask_fp': no_mask_fp, 'mask_fn': mask_fn, 'no_mask_fn': no_mask_fn, 'true_neg': true_neg,
                            'mask_tp_list': mask_tp_list, 'no_mask_tp_list': no_mask_tp_list, 'mask_mc_list': mask_mc_list,
                            'no_mask_mc_list': no_mask_mc_list, 'mask_fp_list': mask_fp_list,
                            'no_mask_fp_list': no_mask_fp_list, 'mask_fn_list': mask_fn_list,
                            'no_mask_fn_list': no_mask_fn_list, 'true_neg_list': true_neg_list, 'confusion_matrix': confusion_matrix}

    with open(pkl_filename, 'wb') as f:
        pickle.dump(save_parameters, f)
    wb.save(excel_filename)
    return save_parameters

if __name__ == "__main__":
    a = "/home/shivakumarka/Documents/Projects/MaskDetection/Code/evaluation/test_set_9/"
    compute_stats(a,0.5,'test_set_9.pkl','test_set_9.xlsx')
