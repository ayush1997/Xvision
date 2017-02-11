#!/usr/bin/env python

import mxnet as mx
import settings

#define the function which returns the data iterators.
def get_iterators(batch_size, data_shape=(3, 224, 224)):
    # first create the directory for lst and rec files
    root = os.path.abspath(os.path.dirname(__file__))
    recordIODir = os.path.join(root, settings.record_IO_directory)
    train  = os.path.join(recordIODir, 'Train.rec')
    val = os.path.join(recordIODir, 'Val.rec')

    train = mx.io.ImageRecordIter(
        path_imgrec         = '../recordIO_dir/Train.rec',
        data_name           = 'data',
        label_name          = 'softmax_label',
        batch_size          = batch_size,
        data_shape          = data_shape,
        shuffle             = True,
        rand_crop           = True,
        rand_mirror         = True,
        mean_img           = "../recordIO_dir/mean.bin")
    val = mx.io.ImageRecordIter(
        path_imgrec         = '../recordIO_dir/Val.rec',
        data_name           = 'data',
        label_name          = 'softmax_label',
        batch_size          = batch_size,
        data_shape          = data_shape,
        rand_crop           = False,
        rand_mirror         = False,
        mean_img="../recordIO_dir/mean.bin")
    test = mx.io.ImageRecordIter(
        path_imgrec='../recordIO_dir/Test.rec',
        data_name='data',
        label_name='softmax_label',
        batch_size=batch_size,
        data_shape=data_shape,
        rand_crop=False,
        rand_mirror=False,
        mean_img="../recordIO_dir/mean.bin")
    return (train, val,test)

####################################################################################
# We then download a pretrained 50-layer ResNet model and load into memory.
# Note. If load_checkpoint reports error, we can remove the downloaded files and try get_model again.
import os, urllib
def download(url):
    filename = url.split("/")[-1]
    if not os.path.exists(filename):
        urllib.urlretrieve(url, filename)
def get_model(prefix, epoch):
    download(prefix+'-symbol.json')
    download(prefix+'-%04d.params' % (epoch,))

get_model('http://data.mxnet.io/models/imagenet/resnet/50-layers/resnet-50', 0)
sym, arg_params, aux_params = mx.model.load_checkpoint('resnet-50', 0)

####################################################################################
# Train
#
# We first define a function which replaces the the last fully-connected layer for a given network.

def get_fine_tune_model(symbol, arg_params, num_classes, layer_name='flatten0'):
    """
    symbol: the pre-trained network symbol
    arg_params: the argument parameters of the pre-trained model
    num_classes: the number of classes for the fine-tune datasets
    layer_name: the layer name before the last fully-connected layer
    """
    all_layers = sym.get_internals()
    net = all_layers[layer_name+'_output']
    net = mx.symbol.FullyConnected(data=net, num_hidden=num_classes, name='fc1')
    net = mx.symbol.SoftmaxOutput(data=net, name='softmax')
    new_args = dict({k:arg_params[k] for k in arg_params if 'fc1' not in k})
    return (net, new_args)

import logging
head = '%(asctime)-15s %(message)s'
logging.basicConfig(level=logging.DEBUG, format=head)

def fit(symbol, arg_params, aux_params, train, val,test, batch_size, num_gpus):
    devs = [mx.gpu(i) for i in range(num_gpus)]
    mod = mx.mod.Module(symbol=new_sym, context=devs)
    mod.bind(data_shapes=train.provide_data, label_shapes=train.provide_label)
    mod.init_params(initializer=mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2))
    mod.set_params(new_args, aux_params, allow_missing=True)
    mod.fit(train, val,
        num_epoch=1,
        batch_end_callback = mx.callback.Speedometer(batch_size, 10),
        kvstore='device',
        optimizer='sgd',
        optimizer_params={'learning_rate':0.01},
        eval_metric='acc')
    # Evaluate accuracy on val_dataiter
    metric = mx.metric.Accuracy()
    return mod.score(test,metric)
####################################################################################
# Then we can start training. We use AWS EC2 g2.8xlarge, which has 8 GPUs.
# Output may varyf
num_classes = 2
batch_per_gpu = 4
num_gpus = 1

(new_sym, new_args) = get_fine_tune_model(sym, arg_params, num_classes)

batch_size = batch_per_gpu * num_gpus
(train, val,test) = get_iterators(batch_size)
mod_score = fit(new_sym, new_args, aux_params, train, val,test, batch_size, num_gpus)
assert mod_score > 0.77, "Low training accuracy."