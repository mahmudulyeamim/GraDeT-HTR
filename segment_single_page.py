from os.path import join
import numpy as np
import operator
import cv2
from math import *
import os.path
import shutil
from BN_DRISHTI.yolov5.detect import load_model, infer
import time
import torch

ROTATE_LINE_DSKEW = "BN_DRISHTI/content/DSkew/"
ROTATE_LINE_HAUGHLINE = "BN_DRISHTI/content/HaughLine_Affine/"
ROTATED_LINE_BY_HOUGHLINE_AFFINE = "BN_DRISHTI/content/Rotated_line_by_HaughLine_Affine/"

YOLO_DETECTIONS = "BN_DRISHTI/yolov5/runs/"
YOLO_FIRST_DETECTION_LABELS = "BN_DRISHTI/yolov5/runs/detect/exp/labels/"
YOLO_SECOND_DETECTION_LABELS = "BN_DRISHTI/yolov5/runs/detect/exp2/labels"
YOLO_THIRD_DETECTION_LABELS = "BN_DRISHTI/yolov5/runs/detect/exp3/labels/"

SORTED_LINE_AFTER_FIRST_DETECTION = "BN_DRISHTI/content/sorted_line_after_1st_detection/"
SORTED_WORD_DETECTION = 'BN_DRISHTI/content/sorted_Word_detection/'
INITIAL_LINE_SEGMENTATION = "BN_DRISHTI/content/initial_line_segmantation/"
FINAL_LINE_SEGMENTATION = "BN_DRISHTI/content/final_line_segmentation/"
FINAL_WORD_SEGMENTATION = "BN_DRISHTI/content/final_word_segmentation/"
SECOND_LINE_DETECTION_LABELS = "BN_DRISHTI/content/2nd line detection for rotated images (labels)/" 
SECOND_LINE_DETECTION_FOR_ROTATED_IMAGES = "BN_DRISHTI/content/2nd line detection for rotated images/"

ORIGINAL_LINE_IMAGES = "BN_DRISHTI/content/Original line images"


def clean_workspace(
        root_path='',
        base_dir="BN_DRISHTI/content",
        image_extensions=('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'),
        extra_dirs=None
):
    """
    Cleans up the workspace by:
      1. Removing any folders named after image files in base_dir.
      2. Removing a fixed list of processing directories.
    """

    base_dir = os.path.join(root_path, base_dir)

    if extra_dirs is None:
        extra_dirs = [
            ROTATE_LINE_DSKEW,
            ROTATE_LINE_HAUGHLINE,
            ROTATED_LINE_BY_HOUGHLINE_AFFINE,
            FINAL_LINE_SEGMENTATION,
            FINAL_WORD_SEGMENTATION,
            INITIAL_LINE_SEGMENTATION,
            SORTED_WORD_DETECTION,
            SORTED_LINE_AFTER_FIRST_DETECTION,
            YOLO_DETECTIONS,
            SECOND_LINE_DETECTION_FOR_ROTATED_IMAGES,
            SECOND_LINE_DETECTION_LABELS,
            ORIGINAL_LINE_IMAGES,
        ]

    # 1. Remove folders named after image files
    for entry in os.listdir(base_dir):
        lower = entry.lower()
        if lower.endswith(image_extensions):
            label = os.path.splitext(entry)[0]
            folder = os.path.join(base_dir, label)
            if os.path.isdir(folder):
                shutil.rmtree(folder)
                print(f"Removed folder: {folder}")

    # 2. Remove extra processing directories
    for d in extra_dirs:
        fullpath = d
        if os.path.isdir(fullpath):
            shutil.rmtree(fullpath)
            print(f"Removed directory: {fullpath}")


def line_sort(lines):
    sort_lines = {}
    for line in lines:
        img_lb = line.split('.')[0]
        lb = [int(i) for i in img_lb.split('_')]
        new_lb = ['0' + str(r) if r < 10 else str(r) for r in lb]
        if len(new_lb) == 3:
            items = int(new_lb[0] + new_lb[1] + new_lb[2])
        if len(new_lb) == 4:
            items = int(new_lb[0] + new_lb[1] + new_lb[2] + new_lb[3])
        sort_lines[items] = line
    sort_lines = dict(sorted(sort_lines.items()))
    new_lines = list(sort_lines.values())
    return new_lines


def yolo_load_model(image_size, half=False, device="cpu", weights='BN_DRISHTI/model/line_model_best.pt'):
    mcfg = load_model(weights=weights, imgsz=image_size, half=half, device=device)
    return mcfg

def yolo_detection(model_config, img_path, conf):
    infer(
        mcfg=model_config,
        conf_thres=conf,
        source=img_path,
        save_txt=True,
        save_conf=True,
    )

class Line_sort:
    def __init__(self, txt_files, txt_loc, sort_label, flag):
        self.txt_files = txt_files
        self.txt_loc = txt_loc
        self.sort_label = sort_label
        self.flag = flag
        self.read_file()

    def read_file(self):
        files = self.txt_files
        os.mkdir(self.sort_label)
        for file in files:
            txt_file = []
            file_loc = self.txt_loc + file
            with open(file_loc, 'r', encoding='utf-8', errors='ignore') as lines:
                for line in lines:
                    token = line.split()

                    _, x, y, w, h, conf = map(float, line.split(' '))
                    if self.flag == 0:  # 1st line detection lavel
                        if w > 0.50 and conf < 0.50:
                            continue
                        else:
                            txt_file.append(token)
                    else:  # Word detection lavel
                        txt_file.append(token)

            if self.flag == 0:  # 1st line detection lavel
                sorted_txt_file = sorted(txt_file, key=operator.itemgetter(2))
            else:  # Word detection lavel
                sorted_txt_file = sorted(txt_file, key=operator.itemgetter(1))

            self.file_write(sorted_txt_file, file)

    def file_write(self, txt_file, file_name):
        loc = self.sort_label + file_name
        with open(loc, 'w') as f:
            c = 0
            for line in txt_file:
                for l in line:
                    c += 1
                    if c == len(line):
                        f.write('%s' % l)
                    else:
                        f.write('%s ' % l)
                f.write("\n")
                c = 0


def sort_detection_label(txt_loc, sort_label, flag):
    txt_files = os.listdir(txt_loc)
    obj = Line_sort(txt_files, txt_loc, sort_label, flag)


def line_segmantation_1(img, img_lb, label, segmented_img_path, flag=0):
    pred_lb = os.listdir(label)
    print(pred_lb)
    pred_lb2 = str(pred_lb[0])
    pred_lb3 = label + pred_lb[0]

    dir = segmented_img_path
    os.mkdir(dir)
    img1 = cv2.imread(img)
    dh, dw, _ = img1.shape
    txt_lb = open(pred_lb3, 'r')
    txt_lb_data = txt_lb.readlines()
    txt_lb.close()
    img_lb2 = img_lb.split('.')[0]

    k = 1
    for dt in txt_lb_data:
        if flag != 0:
            _, x, y, w, h = map(float, dt.split(' '))
        else:
            _, x, y, w, h, conf = map(float, dt.split(' '))

        if w > 0.50 and w < 0.80 and flag == 0:
            x = 0.5
            w = 1.0
        l = int((x - w / 2) * dw)
        r = int((x + w / 2) * dw)
        t = int((y - h / 2) * dh)
        b = int((y + h / 2) * dh)

        crop = img1[t:b, l:r]
        cv2.imwrite("{}/{}_{}.jpg".format(dir, img_lb2, k), crop)
        k += 1


class ImgCorrect():
    def __init__(self, img):
        self.img = img
        self.h, self.w, self.channel = self.img.shape
        # print("Original images h & w -> | w: ",self.w, "| h: ",self.h)
        if self.w <= self.h:
            self.scale = 700 / self.w
            self.img = cv2.resize(self.img, (0, 0), fx=self.scale, fy=self.scale, interpolation=cv2.INTER_NEAREST)
        else:
            self.scale = 700 / self.h
            self.img = cv2.resize(self.img, (0, 0), fx=self.scale, fy=self.scale, interpolation=cv2.INTER_NEAREST)
        self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

    def img_lines(self):
        ret, binary = cv2.threshold(self.gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # rectangular structure
        binary = cv2.dilate(binary, kernel)  # dilate
        edges = cv2.Canny(binary, 50, 200)

        self.lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=20)

        if self.lines is None:
            print("Line segment not found")
            return None

        lines1 = self.lines[:, 0, :]  # Extract as 2D

        imglines = self.img.copy()
        for x1, y1, x2, y2 in lines1[:]:
            cv2.line(imglines, (x1, y1), (x2, y2), (0, 255, 0), 3)
        return imglines

    def search_lines(self):
        lines = self.lines[:, 0, :]  # extract as 2D

        number_inexist_k = 0
        sum_pos_k45 = number_pos_k45 = 0
        sum_pos_k90 = number_pos_k90 = 0
        sum_neg_k45 = number_neg_k45 = 0
        sum_neg_k90 = number_neg_k90 = 0
        sum_zero_k = number_zero_k = 0

        for x in lines:
            if x[2] == x[0]:
                number_inexist_k += 1
                continue
            degree = degrees(atan((x[3] - x[1]) / (x[2] - x[0])))
            if 0 < degree < 45:
                number_pos_k45 += 1
                sum_pos_k45 += degree
            if 45 <= degree < 90:
                number_pos_k90 += 1
                sum_pos_k90 += degree
            if -45 < degree < 0:
                number_neg_k45 += 1
                sum_neg_k45 += degree
            if -90 < degree <= -45:
                number_neg_k90 += 1
                sum_neg_k90 += degree
            if x[3] == x[1]:
                number_zero_k += 1

        max_number = max(number_inexist_k, number_pos_k45, number_pos_k90, number_neg_k45, number_neg_k90,
                         number_zero_k)

        if max_number == number_inexist_k:
            return 90
        if max_number == number_pos_k45:
            return sum_pos_k45 / number_pos_k45
        if max_number == number_pos_k90:
            return sum_pos_k90 / number_pos_k90
        if max_number == number_neg_k45:
            return sum_neg_k45 / number_neg_k45
        if max_number == number_neg_k90:
            return sum_neg_k90 / number_neg_k90
        if max_number == number_zero_k:
            return 0

    def rotate_image(self, degree):
        """
        Positive angle counterclockwise rotation
        :param degree:
        :return:
        """
        if -45 <= degree <= 0:
            degree = degree  # #negative angle clockwise
        if -90 <= degree < -45:
            degree = 90 + degree  # positive angle counterclockwise
        if 0 < degree <= 45:
            degree = degree  # positive angle counterclockwise
        if 45 < degree <= 90:
            degree = degree - 90  # negative angle clockwise
        print("DSkew angle: ", degree)

        # degree = degree - 90
        height, width = self.img.shape[:2]
        heightNew = int(width * fabs(sin(radians(degree))) + height * fabs(
            cos(radians(degree))))  # This formula refers to the previous content
        widthNew = int(height * fabs(sin(radians(degree))) + width * fabs(cos(radians(degree))))

        matRotation = cv2.getRotationMatrix2D((width / 2, height / 2), degree, 1)  # rotate degree counterclockwise
        matRotation[0, 2] += (widthNew - width) / 2
        # Because after rotation, the origin of the coordinate system is the upper left corner of the new image, so it needs to be converted according to the original image
        matRotation[1, 2] += (heightNew - height) / 2

        # Affine transformation, the background color is filled with white
        imgRotation = cv2.warpAffine(self.img, matRotation, (widthNew, heightNew), borderValue=(255, 255, 255))

        # Padding
        pad_image_rotate = cv2.warpAffine(self.img, matRotation, (widthNew, heightNew), borderValue=(0, 255, 0))
        return imgRotation


def dskew(line_path, img):
    img_loc = line_path + img
    im = cv2.imread(img_loc)

    # Padding
    bg_color = [255, 255, 255]
    pad_img = cv2.copyMakeBorder(im, 100, 100, 100, 100, cv2.BORDER_CONSTANT, value=bg_color)

    imgcorrect = ImgCorrect(pad_img)
    lines_img = imgcorrect.img_lines()

    if lines_img is None:
        rotate = imgcorrect.rotate_image(0)
    else:
        degree = imgcorrect.search_lines()
        rotate = imgcorrect.rotate_image(degree)

    return rotate


# Degree conversion
def DegreeTrans(theta):
    res = theta / np.pi * 180
    return res


# Rotate the image degree counterclockwise (original size)
def rotateImage(src, degree):
    # The center of rotation is the center of the image
    h, w = src.shape[:2]
    # Calculate the two-dimensional rotating affine transformation matrix
    RotateMatrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), degree, 1)

    # Affine transformation, the background color is filled with GREEN so that the rotation can be easily understood
    rotate1 = cv2.warpAffine(src, RotateMatrix, (w, h), borderValue=(0, 255, 0))
    # Affine transformation, the background color is filled with white
    rotate = cv2.warpAffine(src, RotateMatrix, (w, h), borderValue=(255, 255, 255))

    # Padding
    bg_color = [255, 255, 255]
    pad_image_rotate = cv2.copyMakeBorder(rotate, 100, 100, 100, 100, cv2.BORDER_CONSTANT, value=bg_color)

    return pad_image_rotate


# Calculate angle by Hough transform
def CalcDegree(srcImage, canny_img):
    lineimage = srcImage.copy()
    lineimg = srcImage.copy()
    # Detect straight lines by Hough transform
    # The fourth parameter is the threshold, the greater the threshold, the higher the detection accuracy
    try:
        lines = cv2.HoughLines(canny_img, 1, np.pi / 180, 200)
        # Due to different images, the threshold is not easy to set, because the threshold is set too high, so that the line cannot be detected, the threshold is too low, the line is too much, the speed is very slow
        theta_sum = 0
        rho_sum = 0
        sum_x1 = sum_x2 = sum_y1 = sum_y2 = 0
        # Draw each line segment in turn
        for i in range(len(lines)):
            for rho, theta in lines[i]:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(round(x0 + 1000 * (-b)))
                y1 = int(round(y0 + 1000 * a))
                x2 = int(round(x0 - 1000 * (-b)))
                y2 = int(round(y0 - 1000 * a))
                # Only select the smallest angle as the rotation angle
                sum_x1 += x1
                sum_x2 += x2
                sum_y1 += y1
                sum_y2 += y2
                rho_sum += rho
                theta_sum += theta
                cv2.line(lineimage, (x1, y1), (x2, y2), (0, 0, 255), 1, cv2.LINE_AA)

        pt1 = (sum_x1 // len(lines), sum_y1 // len(lines))
        pt2 = (sum_x2 // len(lines), sum_y2 // len(lines))

        average = theta_sum / len(lines)
        angle = DegreeTrans(average) - 90
        print("Skewed Angle: ", angle)
        average_rho = rho_sum / len(lines)

        cv2.line(lineimg, pt1, pt2, (0, 0, 255), 2)

        return angle
    except:
        angle = 0.0
        return angle


def ready_for_rotate(line_path, img):
    img_loc = line_path + img
    image = cv2.imread(img_loc)

    org_width = image.shape[1]
    org_height = image.shape[0]

    img1 = image
    im_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(im_gray, 50, 150, apertureSize=3)

    degree = CalcDegree(image, edges)

    if degree == 0.0:
        rotate = dskew(line_path, img)

        filename1 = ROTATE_LINE_DSKEW + img
        cv2.imwrite(filename1, rotate)
        filename = ROTATED_LINE_BY_HOUGHLINE_AFFINE + img
        cv2.imwrite(filename, rotate)
    else:
        rotate = rotateImage(image, degree)

        filename2 = ROTATE_LINE_HAUGHLINE + img
        cv2.imwrite(filename2, rotate)
        filename = ROTATED_LINE_BY_HOUGHLINE_AFFINE + img
        cv2.imwrite(filename, rotate)


def rotate_lines(first_detection):
    line_path = first_detection
    line_dir = line_sort(os.listdir(line_path))

    for img in line_dir:
        ready_for_rotate(line_path, img)


def find_undetected_images(img, label, undetected_images_path=[]):
    img_path = img
    detect_lb_path = label
    undetect_img_path = FINAL_LINE_SEGMENTATION

    def take_valid_img(images):
        image = []
        valid_img_ext = ["jpg", "JPG", "jpeg", "JPEG", "png", "PNG"]
        for img in images:
            try:
                ext = img.split('.')[1]
                if ext not in valid_img_ext:
                    continue
                else:
                    image.append(img)
            except:
                continue
        return image

    img1 = os.listdir(img_path)
    img = take_valid_img(img1)
    detect_lb = os.listdir(detect_lb_path)

    def find_undetect_img(img, detect_lb):
        img_lb = [im.split('.')[0] for im in img]
        dt_lb = [dt.split('.')[0] for dt in detect_lb]
        undt_lb = list(set(img_lb).difference(dt_lb))
        undetect_img = []
        detect_img = []
        for lb in undt_lb:
            for im in img:
                im_lb = im.split('.')[0]
                if lb == im_lb:
                    undetect_img.append(im)
                else:
                    detect_img.append(im)
        print("Undetect image: ", undetect_img)
        write_image(undetect_img)

    def write_image(undt_img):
        for im in undt_img:
            filename = undetect_img_path + im
            img = cv2.imread(img_path + im)
            cv2.imwrite(filename, img)
            undetected_images_path.append(filename)

    find_undetect_img(img, detect_lb)
    
    return undetected_images_path


def crop_image(bb_data, destination, image, img_lb, dh, dw):
    x = float(bb_data[1])
    y = float(bb_data[2])
    w = float(bb_data[3])
    h = float(bb_data[4])

    # x = 0.5
    # w  = 1.0
    l = int((x - w / 2) * dw)
    r = int((x + w / 2) * dw)
    t = int((y - h / 2) * dh)
    b = int((y + h / 2) * dh)

    crop = image[t:b, l:r]
    filename = destination + img_lb
    cv2.imwrite(filename, crop)
    print("Segmented successfully!\n")


def line_segmantation_2(img, img_path, label, label_path, segmented_img_path):
    dir = segmented_img_path
    img1 = cv2.imread(img_path)
    dh, dw, _ = img1.shape
    txt_lb = open(label_path, 'r')
    txt_lb_data = txt_lb.readlines()
    txt_lb.close()
    img_name = img

    max_w = 0
    data1 = []
    for line in txt_lb_data:
        token = line.split()
        data1.append(token)

    if len(data1) == 1:
        bb_data = data1[0]
        wdth = float(bb_data[3])
        if wdth > 0.4:
            crop_image(bb_data, dir, img1, img_name, dh, dw, )
        else:
            filename = dir + img_name
            cv2.imwrite(filename, img1)
    elif len(data1) == 2:
        bb_data1 = data1[0]
        bb_data2 = data1[1]
        w1 = float(bb_data1[3])
        w2 = float(bb_data2[3])
        c1 = float(bb_data1[5])
        c2 = float(bb_data2[5])
        if w1 <= 0.5 and w2 <= 0.5:
            if c1 >= 0.8 and c2 >= 0.8:
                sorted_bb_data = sorted(data1, key=operator.itemgetter(5))
                bb_data = sorted_bb_data[-1]
                crop_image(bb_data, dir, img1, img_name, dh, dw, )
            else:
                filename = dir + img_name
                cv2.imwrite(filename, img1)
        else:
            sorted_bb_data = sorted(data1, key=operator.itemgetter(3))
            bb_data = sorted_bb_data[-1]
            crop_image(bb_data, dir, img1, img_name, dh, dw, )
    elif len(data1) == 3:
        sorted_bb_data = sorted(data1, key=operator.itemgetter(2))
        bb_data = sorted_bb_data[1]
        crop_image(bb_data, dir, img1, img_name, dh, dw, )
    else:
        sorted_bb_data = sorted(data1, key=operator.itemgetter(3))
        bb_data = sorted_bb_data[-1]
        crop_image(bb_data, dir, img1, img_name, dh, dw, )


def word_segmentation(line_images, word_labels):
    line_img = os.listdir(line_images)
    word_label = os.listdir(word_labels)

    for i in word_label:
        for j in line_img:
            fn_i = i.split(".")
            fn_j = j.split(".")
            if fn_i[0] == fn_j[0]:
                dir = FINAL_WORD_SEGMENTATION + fn_i[0]
                os.mkdir(dir)

                img = cv2.imread(line_images + j)
                dh, dw, _ = img.shape
                txt_lb = open(word_labels + i, 'r')
                txt_lb_data = txt_lb.readlines()
                txt_lb.close()
                img_lb = fn_i[0]

                k = 1
                for dt in txt_lb_data:
                    # _, x, y, w, h = map(float, dt.split(' '))
                    _, x, y, w, h, conf = map(float, dt.split(' '))
                    l = int((x - w / 2) * dw)
                    r = int((x + w / 2) * dw)
                    t = int((y - h / 2) * dh)
                    b = int((y + h / 2) * dh)
                    if w > 0.50:
                        continue
                    crop = img[t:b, l:r]
                    cv2.imwrite("{}/{}_{}.jpg".format(dir, img_lb, k), crop)
                    k += 1

def trim_original_image(rotate, org_w, org_h):
    org_width = org_w
    org_height = org_h

    img1 = rotate
    width = img1.shape[1]
    height = img1.shape[0]
    print("Original height -> ",org_height)
    print("Original width -> ",org_width)

    start_row = 60
    end_row = height - 60

    start_col = 60
    end_col = width - 60
    img_new = img1[start_row:end_row, start_col:end_col]

    width1 = img_new.shape[1]
    height1 = img_new.shape[0]
    print("New height -> ",height1)
    print("New width -> ",width1)
        
    return img_new


def load_segmentation_models(
    line_weights='BN_DRISHTI/model_weights/line_model_best.pt',
    word_weights='BN_DRISHTI/model_weights/word_model_best.pt',
    half=False,
    device="cpu",
):
    line_model_config = yolo_load_model(image_size=(640,640), weights=line_weights, half=half, device=device)
    word_model_config = yolo_load_model(image_size=(640,640), weights=word_weights, half=half, device=device)
    
    return line_model_config, word_model_config

def run_segmentation_model(
    image_path,  # path to image
    image_label, # just the image label i.e. 1_1.jpg
    line_model_config,
    word_model_config,
    final_word_segmentation = "BN_DRISHTI/content/final_word_segmentation/",
    root_path = ''
):
    global FINAL_WORD_SEGMENTATION
    FINAL_WORD_SEGMENTATION = final_word_segmentation

    global ROTATE_LINE_DSKEW, ROTATE_LINE_HAUGHLINE, ROTATED_LINE_BY_HOUGHLINE_AFFINE, YOLO_DETECTIONS, YOLO_FIRST_DETECTION_LABELS, YOLO_SECOND_DETECTION_LABELS
    global YOLO_THIRD_DETECTION_LABELS, SORTED_LINE_AFTER_FIRST_DETECTION, SORTED_WORD_DETECTION, INITIAL_LINE_SEGMENTATION
    global FINAL_LINE_SEGMENTATION, SECOND_LINE_DETECTION_LABELS
    global SECOND_LINE_DETECTION_FOR_ROTATED_IMAGES, ORIGINAL_LINE_IMAGES

    ROTATE_LINE_DSKEW = os.path.join(root_path, ROTATE_LINE_DSKEW)
    ROTATE_LINE_HAUGHLINE = os.path.join(root_path, ROTATE_LINE_HAUGHLINE)
    ROTATED_LINE_BY_HOUGHLINE_AFFINE = os.path.join(root_path, ROTATED_LINE_BY_HOUGHLINE_AFFINE)

    YOLO_DETECTIONS = os.path.join(root_path, YOLO_DETECTIONS)
    YOLO_FIRST_DETECTION_LABELS = os.path.join(root_path, YOLO_FIRST_DETECTION_LABELS)
    YOLO_SECOND_DETECTION_LABELS = os.path.join(root_path, YOLO_SECOND_DETECTION_LABELS)
    YOLO_THIRD_DETECTION_LABELS = os.path.join(root_path, YOLO_THIRD_DETECTION_LABELS)

    SORTED_LINE_AFTER_FIRST_DETECTION = os.path.join(root_path, SORTED_LINE_AFTER_FIRST_DETECTION)
    SORTED_WORD_DETECTION = os.path.join(root_path, SORTED_WORD_DETECTION)
    INITIAL_LINE_SEGMENTATION = os.path.join(root_path, INITIAL_LINE_SEGMENTATION)
    FINAL_LINE_SEGMENTATION = os.path.join(root_path, FINAL_LINE_SEGMENTATION)
    FINAL_WORD_SEGMENTATION = os.path.join(root_path, FINAL_WORD_SEGMENTATION)
    SECOND_LINE_DETECTION_LABELS = os.path.join(root_path, SECOND_LINE_DETECTION_LABELS)
    SECOND_LINE_DETECTION_FOR_ROTATED_IMAGES = os.path.join(root_path, SECOND_LINE_DETECTION_FOR_ROTATED_IMAGES)

    ORIGINAL_LINE_IMAGES = os.path.join(root_path, ORIGINAL_LINE_IMAGES)
    
    clean_workspace(root_path=root_path)
    
    # 1st line detection
    yolo_detection(
        model_config=line_model_config,
        img_path=image_path,
        conf=0.30,
    )

    # Sorting Labels of 1st detection on the basis of y...
    txt_loc = YOLO_FIRST_DETECTION_LABELS
    new_sort_label = SORTED_LINE_AFTER_FIRST_DETECTION
    sort_detection_label(txt_loc, new_sort_label, flag=0)
    
    sorted_label = SORTED_LINE_AFTER_FIRST_DETECTION
    filename = INITIAL_LINE_SEGMENTATION
    line_segmantation_1(image_path, image_label, sorted_label, filename)
    
    os.mkdir(ROTATED_LINE_BY_HOUGHLINE_AFFINE)
    os.mkdir(ROTATE_LINE_DSKEW)
    os.mkdir(ROTATE_LINE_HAUGHLINE)

    rotate_lines(filename)

    # 2nd line detection
    rotated_img_path = ROTATED_LINE_BY_HOUGHLINE_AFFINE
    
    yolo_detection(
        model_config=line_model_config,
        img_path=rotated_img_path,
        conf=0.50,
    )

    label_path1 = YOLO_SECOND_DETECTION_LABELS
    if os.path.exists(label_path1) == True:  # Copying and removeg 2nd detection label if exists...
        to_dir = SECOND_LINE_DETECTION_LABELS
        shutil.copytree(label_path1, to_dir)


    target_image_path1 = ROTATED_LINE_BY_HOUGHLINE_AFFINE
    target_image_path = SECOND_LINE_DETECTION_FOR_ROTATED_IMAGES
    if os.path.exists(target_image_path1) == True:  # Copying and removeg 2nd detection label if exists...
        to_dir = target_image_path
        shutil.copytree(target_image_path1, to_dir)

    # Target Images labels
    target_label_path = SECOND_LINE_DETECTION_LABELS

    target_image = os.listdir(target_image_path)
    target_label = os.listdir(target_label_path)

    new_dir = FINAL_LINE_SEGMENTATION
    os.mkdir(new_dir)

    for i in target_image:
        for j in target_label:
            fn_i = i.split(".")
            fn_j = j.split(".")
            if fn_i[0] ==  fn_j[0]:
                # Line Segmentation after 1st Detection...
                # Final Line Segmentation...
                img_path = target_image_path + i
                img = i
                sorted_label = j
                sorted_label_path = target_label_path + j
                line_segmantation_2(img, img_path, sorted_label, sorted_label_path, new_dir)
                
                
    print("List of undetected images in 2nd detection ->")
    undetected_images_path = find_undetected_images(target_image_path, target_label_path)
    print()
    print("Undetected images are shown below -> \n")
    for i in list(set(undetected_images_path)):
        print("Line image path: ",i)
        
        
    final_line_segment = FINAL_LINE_SEGMENTATION
    rotate_line_Dskew = ROTATE_LINE_DSKEW
    dskew_img_list = os.listdir(rotate_line_Dskew)
    for i in dskew_img_list:
        print("Target image -> ",i)
        temp = final_line_segment + i
        print("Path -> ",temp)
        img1 = cv2.imread(temp)
        height, width, channels = img1.shape
        temp2 = rotate_line_Dskew + i
        img2 = cv2.imread(temp2)
        height2, width2, channels = img2.shape
        if height >= height2:
            # os.remove(temp)
            temp3 = trim_original_image(img1, width, height)
            cv2.imwrite(temp, temp3)
        else:
            print("No need for change as its gone through the 2nd line detection!\n")
        
    
    # final word detection
    yolo_detection(
        model_config=word_model_config,
        img_path=FINAL_LINE_SEGMENTATION,
        conf=0.40,
    )
    
    txt_loc = YOLO_THIRD_DETECTION_LABELS
    new_sort_label = SORTED_WORD_DETECTION
    sort_detection_label(txt_loc, new_sort_label, flag=1)
    
    # Word Segmentation... 
    word_labels = SORTED_WORD_DETECTION
    line_images = FINAL_LINE_SEGMENTATION
    final_word_dir = FINAL_WORD_SEGMENTATION
    os.mkdir(final_word_dir)
    word_segmentation(line_images, word_labels)
    

if __name__ == "__main__":
    # 0) reset the peak‐usage counter before you do anything GPU-related
    torch.cuda.reset_peak_memory_stats()

    # 1) measure model‐loading time
    t0 = time.time()
    line_model_config, word_model_config = load_segmentation_models(half=False)
    t1 = time.time()
    print("Time to load the models: ", t1 - t0)

    # model = line_model_config['model']
    # total_params = sum(p.numel() for p in model.parameters())
    # print(f"Total Trainable Parameters: {total_params}")

    # 2) measure segmentation time
    image_path  = 'BN_DRISHTI/page/1_1.jpg'
    image_label = '1_1.jpg'
    t2 = time.time()
    run_segmentation_model(
        image_path=image_path,
        image_label=image_label,
        line_model_config=line_model_config,
        word_model_config=word_model_config
    )
    t3 = time.time()
    print("Time to segment: ", t3 - t2)

    # 3) now query the peak GPU allocation (in bytes) since that reset
    peak_bytes = torch.cuda.max_memory_allocated()
    print(f"Peak GPU memory usage: {peak_bytes/1024**3:.2f} GB")
