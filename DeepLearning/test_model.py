import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
import pickle
import time
from sklearn.metrics import precision_recall_fscore_support
import sys

print sys.argv
testing_folder = sys.argv[1]
test_labels = sys.argv[3]


batch = 25
epoch = 20
testing_folder_len = len([name for name in os.listdir(os.getcwd()+"/"+testing_folder)])

filename = test_labels
fileObject = open(filename,'r')
test_labels = pickle.load(fileObject)
print "test_labels",len(test_labels)

# n_input = 200704
n_input = 25088
# The number of classes which the ConvNet has to classify into .
n_classes = 2
# The number of neurons in the each Hidden Layer .
n_hidden1 = 512
n_hidden2 = 512

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


    return {
        'graph_def': graph_def
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


epsilon = 1e-3
g2 = tf.Graph()
with g2.as_default():

    # Tensorflow Graph input .
    x = tf.placeholder("float", [None, n_input])
    y = tf.placeholder("float", [None, n_classes])

    train_label = tf.argmax(y,1)
    with tf.name_scope('layer1'):
        W_1 = tf.get_variable(
                    name="W1",
                    shape=[n_input, n_hidden1],
                    dtype=tf.float32,
                    initializer=tf.contrib.layers.xavier_initializer())

        # b_1 = tf.get_variable(
        #     name='b1',
        #     shape=[n_hidden1],
        #     dtype=tf.float32,
        #     initializer=tf.constant_initializer(0.0))

        z1_BN = tf.matmul(x, W_1)
        batch_mean1, batch_var1 = tf.nn.moments(z1_BN,[0])
        scale1 = tf.Variable(tf.ones([n_hidden1]))
        beta1 = tf.Variable(tf.zeros([n_hidden1]))
        BN1 = tf.nn.batch_normalization(z1_BN,batch_mean1,batch_var1,beta1,scale1,epsilon)

        # h_1 = tf.nn.relu(tf.nn.bias_add(tf.matmul(x, W_1),b_1))
        h_1 = tf.nn.relu(BN1)

    with tf.name_scope('layer2'):
        W_2 = tf.get_variable(
                    name="W2",
                    shape=[n_hidden1,n_hidden2],
                    dtype=tf.float32,
                    initializer=tf.contrib.layers.xavier_initializer())

        # b_2 = tf.get_variable(
        #     name='b2',
        #     shape=[n_hidden2],
        #     dtype=tf.float32,
        #     initializer=tf.constant_initializer(0.0))

        z2_BN = tf.matmul(h_1, W_2)
        batch_mean2, batch_var2 = tf.nn.moments(z2_BN,[0])
        scale2 = tf.Variable(tf.ones([n_hidden2]))
        beta2 = tf.Variable(tf.zeros([n_hidden2]))
        BN2 = tf.nn.batch_normalization(z2_BN,batch_mean2,batch_var2,beta2,scale2,epsilon)

        # h_2 = tf.nn.relu(tf.nn.bias_add(tf.matmul(h_1, W_2),b_2))
        h_2 = tf.nn.relu(BN2)

    with tf.name_scope('output'):
        W_3 = tf.get_variable(
                   name="W3",
                   shape=[n_hidden2,n_classes],
                   dtype=tf.float32,
                   initializer=tf.contrib.layers.xavier_initializer())

        b_3 = tf.get_variable(
           name='b3',
           shape=[n_classes],
           dtype=tf.float32,
           initializer=tf.constant_initializer(0.0))

        h_3 = tf.nn.bias_add(tf.matmul(h_2, W_3),b_3)

    # h_3 = tf.nn.softmax(h_3)
    # h_3 = h_3

    Cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = h_3, labels=  y))
    optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(Cost)
    # optimizer = tf.train.AdamOptimizer(0.01).minimize(Cost)

    # saver = tf.train.Saver()
    #Monitor accuracy
    soft = tf.nn.softmax(h_3)
    predicted_y = tf.argmax(tf.nn.softmax(h_3), 1)
    actual_y = tf.argmax(y, 1)


    correct_prediction = tf.equal(predicted_y, actual_y)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


    # names = [op.name for op in g2.get_operations()]
    # print names
    saver = tf.train.Saver()


class_pred = np.array([])
class_actual=np.array([])

r = (testing_folder_len - (testing_folder_len%25))+1
print r


with tf.Session(graph=g2) as sess1:
    # To initialize values with saved data
    sess1.run(tf.global_variables_initializer())
    saver.restore(sess1, os.getcwd()+"/"+sys.argv[4]+"/"+"my-model-"+str(epoch-1)+".ckpt")

    for j in range(0,r,25):
        test_img = []

        file_Name = os.getcwd()+"/"+sys.argv[2]+"/"+ str(j)
        fileObject = open(file_Name,'r')
        # load the object from the file into var b
        content_features = pickle.load(fileObject)
        print content_features.shape

        if j==r-1:
            test_label = test_labels[j:]
            print "test_label",test_label.shape
        else:
            test_label = test_labels[j:25+j]
            print "test_label",test_label.shape

        acc,pred,s,actual = sess1.run([accuracy,predicted_y,soft,actual_y], feed_dict={x: content_features,y: test_label})
        print acc
        print s
        print "predicted",pred
        print "actual",actual

        # print type(pred)
        # class_pred+=list(pred)
        class_pred = np.concatenate((class_pred, pred), axis=0)
        class_actual = np.concatenate((class_actual, actual), axis=0)
        # class_actual+=list(actual)
        print np.unique(class_actual,return_counts=True)

print class_pred
print class_actual

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

conf_matrix = confusion_matrix(class_actual,class_pred)

print conf_matrix
prfs = precision_recall_fscore_support(class_actual, class_pred)
print "precision : ",prfs[0] 
print "recall : ",prfs[1] 
print "fscore : ",prfs[2] 
print "support : ",prfs[3] 
plt.matshow(conf_matrix)
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')

plt.show()


# python test_model.py <testing images folder> <save test matrix> <testing label pickle> <save model checkpoints>
