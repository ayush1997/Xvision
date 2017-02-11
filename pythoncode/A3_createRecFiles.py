"""
This file is used to generate the lst files for test/tran/val and then
it uses these list files to generate the .rec files

python im2rec.py -h #for help
"""

import subprocess
import os
import settings

def makeLstFile(prefix, images_loc):
    """
    :param outputfile: name of the list file
    :param imageloc: where images are located, both positive and negative samples
    :return:
    """
    #the recursive means to walk through subfolders and give labels based on which folder image is in
    subprocess.call("python ~/mxnet/tools/im2rec.py --list True --recursive True " + prefix + " " + images_loc, shell=True)
    os.remove(prefix+"_test.lst")
    os.remove(prefix+"_val.lst")

    os.rename(os.path.join(prefix + "_train.lst"), os.path.join(root, prefix + ".lst"))

def makeRecFile(prefix, imageloc):
    """
    :param prefix: name of the list file
    :param imageloc: where images are located, both positive and negative samples
    :return:
    """
    subprocess.call("python ~/mxnet/tools/im2rec.py --encoding .png " + prefix + " " + imageloc, shell=True)

#first create the directory
settings.makeDir(settings.recordIODir)

#get the absolute path
root = os.path.abspath(os.path.dirname(__file__))
recordIODir = os.path.join(root, settings.recordIODir)

#some locations
val = os.path.join(recordIODir,"Val")
test = os.path.join(recordIODir,"Test")
train = os.path.join(recordIODir,"Train")
valdir = os.path.join(root,settings.mxnet_images_val_dir)
testdir = os.path.join(root,settings.mxnet_images_test_dir)
traindir = os.path.join(root,settings.mxnet_images_train_dir)

#first lets create the list files
makeLstFile(val, valdir)
makeLstFile(test, testdir)
makeLstFile(train, traindir)

#now the rec files
makeRecFile(val, valdir)
makeRecFile(test, testdir)
makeRecFile(train, traindir)