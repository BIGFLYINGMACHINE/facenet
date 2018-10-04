# coding=UTF-8
"""
An example of how to use your own dataset to train a classifier that recognizes people.
一个关于如何使用自己的数据集训练分类器的例子
"""
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
import tensorflow as tf
import numpy as np
import facenet
import os
import pickle
import cv2
import time
from mtcnn.mtcnn import MTCNN


def main():
    with tf.Graph().as_default():

        with tf.Session() as sess:

            np.random.seed(seed=666)
            # 导入训练好的模型
            print('Loading feature extraction model')
            facenet.load_model('~/models/facenet/20180402-114759/20180402-114759.pb')

            classifier_filename_exp = os.path.expanduser('~/models/facenet/lfw_classifier.pkl')
            # Classify images
            print('Testing classifier')
            with open(classifier_filename_exp, 'rb') as infile:
                (model, class_names) = pickle.load(infile)
            print(class_names)
            print(len(class_names))
            print('Loaded classifier model from file "%s"' % classifier_filename_exp)

            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]
            cam = cv2.VideoCapture("/home/zack/Downloads/zikang_air_view.mp4")
            cv2.namedWindow("detect window", 0)
            face_cascade = cv2.CascadeClassifier('lbpcascade_frontalface.xml')
            # detector = MTCNN()
            while True:
                success, img = cam.read()
                if success:
                    start_time = time.time()
                    faces, rects = facenet.get_faces_with_cascade(img, 160, face_cascade)
                    # faces, rects = facenet.get_faces_with_mtcnn(img, 160, detector)
                    process_time = time.time() - start_time
                    print("Process time on getting faces: ", process_time)
                    if faces.shape[0] > 0:
                        start_time = time.time()
                        num_images = len(faces)
                        emb_array = np.zeros((num_images, embedding_size))
                        # Run forward pass to calculate embeddings
                        print('Calculating features for images')
                        feed_dict = {images_placeholder: faces, phase_train_placeholder: False}
                        emb_array[0:num_images, :] = sess.run(embeddings, feed_dict=feed_dict)

                        predictions = model.predict_proba(emb_array)
                        best_class_indices = np.argmax(predictions, axis=1)
                        best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                        for i in range(len(best_class_indices)):
                            if best_class_probabilities[i] > 0.3:
                                print('face%4d  %s: %.3f' % (
                                    i, class_names[best_class_indices[i]], best_class_probabilities[i]))
                                x, y, w, h = rects[i]
                                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
                                cv2.putText(img, class_names[best_class_indices[i]],
                                            (x, y),
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            1,
                                            (0, 0, 255),
                                            2)
                            else:
                                x, y, w, h = rects[i]
                                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
                                cv2.putText(img, "Not target",
                                            (x, y),
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            1,
                                            (0, 0, 0),
                                            2)
                        process_time = time.time() - start_time
                        print("Process time on getting predictions is: ", process_time)
                    else:
                        print("Cannot locate face in current frame")
                    cv2.imshow("detect window", img)
                    key = cv2.waitKey(1) & 0xff
                    if key == 27:
                        break
                else:
                    print("Error: cannot open camera or video")


def test_on_images(path, size):
    path = os.path.expanduser(path)
    if os.path.isdir(path):
        names = [name for name in os.listdir(path) if os.path.isfile(os.path.join(path, name))]
        num_images = len(names)
        faces = np.zeros((num_images, size, size, 3))
        i = 0
        for name in names:
            face = cv2.imread(os.path.join(path, name))
            face = facenet.prewhiten(face)
            faces[i, :, :, :] = face
            i += 1
        # dataset = facenet.get_dataset(path)
        # paths, labels = facenet.get_image_paths_and_labels(dataset)
        # faces = facenet.load_data(paths, False, False, size)
        with tf.Graph().as_default():

            with tf.Session() as sess:

                np.random.seed(seed=666)
                # 导入训练好的模型
                print('Loading feature extraction model')
                facenet.load_model('~/models/facenet/20180402-114759/20180402-114759.pb')

                classifier_filename_exp = os.path.expanduser('~/models/facenet/lfw_classifier.pkl')
                # Classify images
                print('Testing classifier')
                with open(classifier_filename_exp, 'rb') as infile:
                    (model, class_names) = pickle.load(infile)
                print(class_names)
                print(len(class_names))
                print('Loaded classifier model from file "%s"' % classifier_filename_exp)

                # Get input and output tensors
                images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
                phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
                embedding_size = embeddings.get_shape()[1]
                if faces.shape[0] > 0:
                    start_time = time.time()
                    num_images = len(faces)
                    emb_array = np.zeros((num_images, embedding_size))

                    # Run forward pass to calculate embeddings
                    print('Calculating features for images')
                    feed_dict = {images_placeholder: faces, phase_train_placeholder: False}
                    emb_array[0:num_images, :] = sess.run(embeddings, feed_dict=feed_dict)

                    predictions = model.predict_proba(emb_array)
                    best_class_indices = np.argmax(predictions, axis=1)
                    best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                    # print(predictions)
                    # print(best_class_indices)
                    for i in range(len(best_class_indices)):
                        print('face%4d  %s: %.3f' % (
                            i, class_names[best_class_indices[i]], best_class_probabilities[i]))
                    process_time = time.time() - start_time
                    print("process time: ", process_time)
                else:
                    print("Cannot get faces")

    else:
        print("path name incorrect!")


if __name__ == '__main__':
    main()
    # test_on_images("~/datasets/Zhou_Zikang/Zhou_Zikang", 160)
