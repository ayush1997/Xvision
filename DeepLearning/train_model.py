import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
import pickle
import time
import sys

print sys.argv
training_folder = sys.argv[1]
train_labels = sys.argv[3]
mode_folder = sys.argv[4]
batch = 20

training_folder_len = len([name for name in os.listdir(os.getcwd()+"/"+training_folder)])


filename = train_labels
fileObject = open("../"+filename,'r')
train_labels = pickle.load(fileObject)
print "train_labels",len(train_labels)


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

    Cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(h_3, y))
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
    saver = tf.train.Saver(max_to_keep=15)


r = (training_folder_len - (training_folder_len%batch))+1
print r

with tf.Session(graph=g2) as sess2, g2.device('/gpu:0'):
# sess =  tf.Session(graph=g2)
    sess2.run(tf.initialize_all_variables())
    print "shit"
    # saver.save(sess2, 'my-model')

    accuracy_list=[]
    cost=[]
    n_epochs = 20
    for epoch in range(n_epochs):

        start_time = time.time()

        for j in range(0,r,20):

            file_Name =  os.getcwd()+"/"+sys.argv[2]+"/"+ str(j)
            fileObject = open(file_Name,'r')
            # load the object from the file into var b
            content_features = pickle.load(fileObject)

            # print type(content_features)
            content_features = content_features.reshape((content_features.shape[0],7*7*512))
            # content_features = content_features.reshape((content_features.shape[0],28*28*256))
            print content_features.shape , "Feature Map Shape"

            print "j=",j

            if j==r-1:
                label = train_labels[j:]
                print label.shape
            else:
                label = train_labels[j+0:j+20]
                print label.shape

            _,l,w1,cst = sess2.run([optimizer,train_label,W_1,Cost], feed_dict={x: content_features, y:label})

            print l
            # print str(epoch) + "-------------------------------------"


            if j % 100==0:
                print "----------accuracy after: epoch="+str(epoch),"j="+str(j)
            #     accuracy_list.append(acc)
                cost.append(cst)

                print "COST",cost

        print("--- %s seconds ---" % (time.time() - start_time))
        j=0
        path_name = os.getcwd()+"/"+sys.argv[4]+"/"+"my-model-"+str(epoch)+".ckpt"
        save_path = saver.save(sess2, path_name)
        print path_name,"saved"
        #

# python train_model.py <Training images folder> <Train images codes folder> <Training image labels file> <Folder to save models>
