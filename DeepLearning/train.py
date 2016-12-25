import matplotlib.pyplot as plt
import tensorflow as tf
import urllib
import numpy as np
import zipfile
import os
from scipy.io import wavfile
from skimage.transform import resize as imresize
import pickle
import time
import pickle


def get_vgg_model():
    # download('https://s3.amazonaws.com/cadl/models/vgg16.tfmodel')
    with open("vgg16.tfmodel", mode='rb') as f:
        graph_def = tf.GraphDef()
        try:
            graph_def.ParseFromString(f.read())
        except:
            print('try adding PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python ' +
                  'to environment.  e.g.:\n' +
                  'PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python ipython\n' +
                  'See here for info: ' +
                  'https://github.com/tensorflow/tensorflow/issues/582')

    # download('https://s3.amazonaws.com/cadl/models/synset.txt')
    # with open('synset.txt') as f:
    #     labels = [(idx, l.strip()) for idx, l in enumerate(f.readlines())]

    return {
        'graph_def': graph_def
        # 'labels': labels
        # 'preprocess': preprocess,
        # 'deprocess': deprocess
    }

def preprocess(img, crop=True, resize=True, dsize=(224, 224)):
    if img.dtype == np.uint8:
        img = img / 255.0

    if crop:
        short_edge = min(img.shape[:2])
        yy = int((img.shape[0] - short_edge) / 2)
        xx = int((img.shape[1] - short_edge) / 2)
        crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    else:
        crop_img = img

    if resize:
        norm_img = imresize(crop_img, dsize, preserve_range=True)
    else:
        norm_img = crop_img

    return (norm_img).astype(np.float32)
def deprocess(img):
    return np.clip(img * 255, 0, 255).astype(np.uint8)
    # return ((img / np.max(np.abs(img))) * 127.5+127.5).astype(np.uint8)

net = get_vgg_model()

# labels = net['labels']

g1 = tf.Graph()


with tf.Session(graph=g1) as sess, g1.device('/cpu:0'):
    tf.import_graph_def(net['graph_def'], name='vgg')
    # names = [op.name for op in g1.get_operations()]
# print names



def get_content_feature(img_4d):
    with tf.Session(graph=g1) as sess, g1.device('/gpu:0'):


            content_layer = 'vgg/pool5:0'
            content_features= g1.get_tensor_by_name(content_layer).eval(
                    session=sess,
                    feed_dict={x1: img_4d,
                        'vgg/dropout_1/random_uniform:0': [[1.0]],
                        'vgg/dropout/random_uniform:0': [[1.0]]
                    })

            # train_new.append(content_features)
            print content_features.shape
            return content_features


# m=0
# # prepare trainig feature set
# os.mkdir("/home/ayush/Documents/xray/DeepLearning/features-nodule-only")
# os.mkdir("/home/ayush/Documents/xray/DeepLearning/test-features-nodule-only")
# for j in range(0,141,20):
# # for j in range(1):
#     img=[]
#     # labels = []
#     start_time = time.time()
#     if j==140:
#         m = 6
#     for i in range(j+0,j+20-m):
#     # for i in range(980,994):
#
#         og = plt.imread("final_train_images_nodule_only/"+str(i)+".png")
#         og = preprocess(og)
#         img.append(og)
#     print "j=",j
#
#     x1 = g1.get_tensor_by_name('vgg/images' + ':0')
#
#     img_4d = np.array(img)
#     # img_4d = img_4d.reshape((1,224,244,3))
#     # img_4d = img[np.newaxis]
#
#     print img_4d.shape , "Image Shape"
# #
#
#     # content_features = content_features.reshape((content_features.shape[0],7*7*512))
#     content_features = get_content_feature(img_4d).reshape((get_content_feature(img_4d).shape[0],7*7*512))
#     print content_features.shape , "Feature Map Shape"
#
#
#     file_Name = "/home/ayush/Documents/xray/DeepLearning/features-nodule-only/"+str(j)
#     # open the file for writing
#     fileObject = open(file_Name,'wb')
#
#     # this writes the object a to the
#     # file named 'testfile'
#     pickle.dump(content_features,fileObject)
#
#     # here we close the fileObject
#     fileObject.close()
#
#     print("--- %s seconds ---" % (time.time() - start_time))
#
#

# prepare test set

for j in range(0,26,25):
    test_img = []
    start_time = time.time()
    x1 = g1.get_tensor_by_name('vgg/images' + ':0')
    # if j==25:
    #
    #     for i in range(25,31):
    #
    #         og = plt.imread("final_test_images_nodule/"+str(i)+".png")
    #         og = preprocess(og)
    #         test_img.append(og)
    #     test_img = np.array(test_img)
    # else:
    for i in range(j,j+25):

        og = plt.imread("final_test_images_nodule_only/"+str(i)+".png")
        og = preprocess(og)
        test_img.append(og)
    test_img = np.array(test_img)



    img_4d = np.array(test_img)
    # img_4d = img_4d.reshape((1,224,244,3))
    # img_4d = img[np.newaxis]

    test_img = get_content_feature(img_4d).reshape((get_content_feature(img_4d).shape[0],7*7*512))
    print "test",test_img.shape

    print "new test",test_img.shape

    file_Name = "/home/ayush/Documents/xray/DeepLearning/test-features-nodule-only/"+str(j)

    # open the file for writing
    fileObject = open(file_Name,'wb')

    # this writes the object a to the
    # file named 'testfile'
    pickle.dump(test_img,fileObject)

    # here we close the fileObject
    fileObject.close()

    print("--- %s seconds ---" % (time.time() - start_time))
