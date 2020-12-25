"""Performs face alignment and calculates L2 distance between the embeddings of images."""

# MIT License
#
# Copyright (c) 2016 David Sandberg
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import imageio
from PIL import Image
from scipy import misc
import tensorflow as tf
import numpy as np
import sys
import os
import copy
import argparse
import facenet
import align.detect_face

#通过模型评估得到,10折交叉验证的平均值
threshold = 0.90704

def main(args):
    '''
    人脸识别
    读取人脸图片,通过网络计算得到特征向量
    计算与人脸数据库中的特征向量的欧式距离,判断是否是人脸数据库中的某人
    判断采用的阈值为facenet模型评估时(采用20180402-114759模型,lfw人脸数据库)10折交叉验证的平均值"0.90704"
    '''

    #读取图片文件,采用MTCNN对图片数据进行人脸检测,生成相应大小的人脸数据
    images = load_and_align_data(args.image_files, args.image_size, args.margin, args.gpu_memory_fraction)
    with tf.Graph().as_default():

        with tf.Session() as sess:

            # Load the model
            facenet.load_model(args.model)

            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            # Run forward pass to calculate embeddings
            feed_dict = { images_placeholder: images, phase_train_placeholder:False }
            emb = sess.run(embeddings, feed_dict=feed_dict)

            print('Input Images:')
            for image in args.image_files:
                print(image)
            print('')

            #加载人脸数据库
            face_database = np.load(args.data_dir, allow_pickle=True)
            label = []
            base_emb_list = []
            for key in face_database.item().keys():
                label.append(key)
            for value in face_database.item().values():
                base_emb_list.append(value)
            base_emb = np.array(base_emb_list)


            #计算所有"输入人脸图像的特征向量"与"人脸数据库中的特征向量"对之间的欧式距离
            label_num = len(label)
            image_num = len(args.image_files)
            distance_matrix = np.zeros((image_num ,label_num))
            for  i in range(image_num):
                for j in range(label_num):
                    distance = np.sqrt(np.sum(np.square(np.subtract(emb[i, :], base_emb[j, :]))))
                    distance_matrix[i][j] = distance

            #获取与"输入人脸图像的特征向量"的欧式距离最小的"人脸数据库中的特征向量"对应的label
            #如果欧氏距离小于threshold(通过模型评估得到)0.90704, 则判定当前输入图像的label命中
            for i in range(image_num):
                distance_list = distance_matrix[i, :]
                min_distance = np.min(distance_list)
                print('Face Recognise Result:')
                if min_distance <= threshold:
                    index = np.where(distance_list == min_distance)[0]
                    print(np.array(label)[index])
                else :
                    print("Unknown!!!")


def load_and_align_data(image_paths, image_size, margin, gpu_memory_fraction):

    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor

    print('Creating networks and loading parameters')
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)

    tmp_image_paths=copy.copy(image_paths)
    img_list = []
    for image in tmp_image_paths:
        img = imageio.imread(os.path.expanduser(image), pilmode='RGB')
        #caution: image channel order must be 'RGB'
        #img_open = Image.open(os.path.expanduser(image))
        #img = np.array(img_open)
        img_size = np.asarray(img.shape)[0:2]
        bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
        if len(bounding_boxes) < 1:
          image_paths.remove(image)
          print("can't detect face, remove ", image)
          continue
        det = np.squeeze(bounding_boxes[0,0:4])
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0]-margin/2, 0)
        bb[1] = np.maximum(det[1]-margin/2, 0)
        bb[2] = np.minimum(det[2]+margin/2, img_size[1])
        bb[3] = np.minimum(det[3]+margin/2, img_size[0])
        cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
        aligned = np.array(Image.fromarray(cropped).resize((image_size, image_size), resample=Image.BILINEAR))
        #aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
        prewhitened = facenet.prewhiten(aligned)
        img_list.append(prewhitened)
    images = np.stack(img_list)
    return images

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str,
        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
    parser.add_argument('--data_dir', type=str,
        help='This directory contains face database')
    parser.add_argument('--image_files', type=str, nargs='+', help='Images to compare')
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--margin', type=int,
        help='Margin for the crop around the bounding box (height, width) in pixels.', default=44)
    parser.add_argument('--gpu_memory_fraction', type=float,
        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    print('Example:')
    print('python src/face_recognise.py --model pretrained_model/20180402-114759/20180402-114759.pb --data_dir ./src/face_database/face_database.npy --image_files ./src/images/custom_faces/*.jpg')
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
