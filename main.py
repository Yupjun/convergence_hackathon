import sys
import random
from skimage import io
import os
import glob
import cv2
from skimage.transform import rotate
from mtcnn.mtcnn import MTCNN
import shutil
import numpy as np
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFilter
import numpy
import collections
from imutils import face_utils
import argparse
import imutils
import dlib
from get_images import *
from act_loss_test import *
from autoencoder import *
from landmark_top3 import *
from face_landmark import *

if __name__=='__main__':
    if not "frame" in os.listdir(sys.argv[1]):
        os.mkdir(sys.argv[1] + "/frame")

    if not "result" in os.listdir(sys.argv[1]):
        os.mkdir(sys.argv[1] + "/result")

    if not "done_mtcnn" in os.listdir(sys.argv[1]):
        os.mkdir(sys.argv[1] + "/done_mtcnn")

    if not "grad_cam_raw" in os.listdir(sys.argv[1]):
        os.mkdir(sys.argv[1] + "/grad_cam_raw")

    if not "grad_cam_applied" in os.listdir(sys.argv[1]):
        os.mkdir(sys.argv[1] + "/grad_cam_applied")

    transform = transforms.Compose([transforms.Resize((240, 240)), transforms.ToTensor()])

    video_location = sys.argv[1] + "/*.mp4"
    video_name = glob.glob(video_location)
    vidcap = cv2.VideoCapture(video_name[0])

    print(video_name[0])
    # video to frame extraction

    success, image = vidcap.read()
    count = 0
    while success:
        temp_frame = sys.argv[1] + "/frame/" + str(count) + ".jpg"
        cv2.imwrite(temp_frame, image)  # save frame as JPEG file
        success, image = vidcap.read()
        count += 1

    print("temp")
    image_format = "*.*"
    rect_expand_ratio = 1.2  # ratio expanding face region
    ignore_ratio = 1.732  # ignore small faces where the w+h < the largest face's (w+h)/(this ratio)

    images = get_images(sys.argv[1] + "/frame/")

    detector = MTCNN()

    for (filename, image_path) in images:
        # image = io.imread(image_path, cv2.IMREAD_UNCHANGED)
        image = io.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        img_width = image.shape[1]
        img_height = image.shape[0]

        result = detector.detect_faces(image)

        max_wh = 0
        for n in range(len(result)):
            if result[n]['box'][2] + result[n]['box'][3] > max_wh:
                max_wh = result[n]['box'][2] + result[n]['box'][3]

        for n in range(len(result)):
            bounding_box = result[n]['box']
            landmark = result[n]['keypoints']

            x_center = bounding_box[0] + int(bounding_box[2] / 2)
            y_center = bounding_box[1] + int(bounding_box[3] / 2)

            if bounding_box[2] + bounding_box[3] < max_wh / ignore_ratio:
                continue

            width = int(bounding_box[2] * rect_expand_ratio)
            height = int(bounding_box[3] * rect_expand_ratio)

            left_eye = landmark['left_eye']  # (x, y)
            right_eye = landmark['right_eye']  # (x, y)

            dY = right_eye[1] - left_eye[1]
            dX = right_eye[0] - left_eye[0]
            angle = np.degrees(np.arctan2(dY, dX))

            M = cv2.getRotationMatrix2D((x_center, y_center), angle, 1)
            output = cv2.warpAffine(image, M, (img_width, img_height), flags=cv2.INTER_CUBIC)

            gray = output
            face_rect = (x_center - int(width / 2),  # Left
                         y_center - int(height / 2),  # Top
                         x_center + int(width / 2),  # Right
                         y_center + int(height / 2))  # Bottom

            face = Image.fromarray(gray).crop(face_rect)

            min_size = min(width, height)
            image_resize = 8
            while (image_resize + 8) < min_size:
                image_resize += 8

            face = face.resize((image_resize, image_resize), Image.NEAREST)

            f_name, _ = os.path.splitext(filename)
            saved_name = "result/" + str(n) + ".jpg"
            saved_name = os.path.join(sys.argv[1], saved_name)
            face.save(saved_name)

        shutil.move(image_path, sys.argv[1] + "/done_mtcnn/")

    ## detection

    count = 0
    device = torch.device("cuda")

    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(0)

    model = Autoencoder()
    model.load_state_dict(torch.load("ft_final_baseline.pth"))
    model.cuda()
    model = model.eval()
    batch_size = 1

    # detection and draw grad cam

    # draw each image give output on txt

    image_list = glob.glob(sys.argv[1] + "/done_mtcnn/*")
    image_file_index = 0
    for location in image_list:
        with torch.no_grad():
            # save bgr to rgb
            origin = Image.open(location)
            b, g, r = origin.split()
            origin = Image.merge("RGB", (r, g, b))
            temps = Image.merge("RGB", (r, g, b))

            # Convert RGB to BGR
            original = cv2.cvtColor(numpy.array(origin), cv2.COLOR_RGB2BGR)
            #     imgSmall = origin.resize((16,16),resample=Image.BILINEAR)
            #     result = imgSmall.resize(origin.size,Image.NEAREST)

            # We need trainging result

            image = transform(origin)
            image = torch.unsqueeze(image, dim=0)
            image = image.to(device)
            temp = torch.rand([1])
            _, act_data = model(image, temp)

            summation = 0  # Fake
            summation1 = 0  # Real
            # real
            for t_index in range(64):
                summation += abs(act_data[0][64 + t_index])
            # fake
            for t_index in range(64):
                summation1 += abs(act_data[0][t_index])
            ## do real and fake at the same time
            summation = summation.cpu()  # Real
            summation1 = summation1.cpu()  # Fake
            map1 = transforms.ToPILImage()(summation).convert()  # Real
            map2 = transforms.ToPILImage()(summation1).convert()  # Fake

            map1_t = transforms.Resize((240, 240))(map1)  # Real
            map2_t = transforms.Resize((240, 240))(map2)  # Fake

            map_arr = np.array(map1_t)  # Real
            map_arr1 = np.array(map2_t)  # Fake
            # real image save

            plt.imshow(map_arr, cmap=plt.cm.jet, interpolation='nearest')  # Real
            plt.axis('off')
            name = sys.argv[1] + "/grad_cam_raw/" + str(image_file_index) + "_.png"
            plt.pause(1e-13)

            plt.savefig(name, bbox_inches='tight', pad_inches=0)
            # fake image save part
            plt.imshow(map_arr1, cmap=plt.cm.jet, interpolation='nearest')  # Fake

            plt.axis('off')
            name1 = sys.argv[1] + "/grad_cam_raw/" + str(image_file_index) + "__.png"
            plt.pause(1e-13)
            plt.savefig(name1, bbox_inches='tight', pad_inches=0)

            # converting pil to cv cv default is bgr so we converted
            origin = cv2.cvtColor(np.array(original), cv2.COLOR_BGR2RGB)
            src2 = cv2.imread(name)  # real
            src3 = cv2.imread(name1)  # fake
            width = origin.shape[1]
            height = origin.shape[0]
            src2 = cv2.resize(src2, (width, height))
            src3 = cv2.resize(src3, (width, height))

            outputs = act_loss_test(act_data)

            if outputs.tolist()[0][0] > 0.5:
                dst = cv2.addWeighted(original, 0.7, src3, 0.3, 0)
                heat_map_name = sys.argv[1] + "/grad_cam_applied/" + str(image_file_index) + ".png"
                cv2.imwrite(heat_map_name, dst)
                text_file_name = sys.argv[1] + "/grad_cam_applied/" + str(image_file_index) + ".txt"
                data = numpy.asarray(map2_t)
                f = open(text_file_name, 'w')
                script = "Real " + str(outputs.tolist()[0][0] * 100)
                f.write(script)
                f.close()
            else:
                dst = cv2.addWeighted(original, 0.7, src3, 0.3, 0)
                heat_map_name = sys.argv[1] + "/grad_cam_applied/" + str(image_file_index) + ".png"
                cv2.imwrite(heat_map_name, dst)
                text_file_name = sys.argv[1] + "/grad_cam_applied/" + str(image_file_index) + ".txt"
                f = open(text_file_name, 'w')
                script = "Fake " + str(outputs.tolist()[0][1] * 100)
                f.write(script)
                f.close()
                data = numpy.asarray(map1_t)
            image_file_index += 1

        ## need to be checked

        temp = data.tolist()

        landmark_list = face_landmark(location)
        landmark_top3(temp, landmark_list)