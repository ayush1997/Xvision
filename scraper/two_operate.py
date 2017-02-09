#!/usr/bin/env python

import json
from pprint import pprint
import numpy as np
from scipy.misc import imresize
from skimage.transform import resize as imresize
import mxnet as mx

#=======================================================================================================================
# functions
#=======================================================================================================================

def preprocess(img, crop=True, resize=True, dsize=(224, 224), scale =True):
    """
    :param img:
    :param crop:
    :param resize:
    :param dsize:
    :param scale:
    :return:
    """
    # compute the mean (Where do I get this number?
    # img = img-np.array([117,117,117])

    #this scales between -1 and 1 (float 64)
    if scale == True:
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

    # convert back to ints
    return np.clip(norm_img * 255, 0, 255).astype(np.uint8)

def addfilesindirtolist(dir, listofimages):
    """
    enumerates all files in directory and populates list with their name minus the extension
    :param dir:
    :param list:
    :return:
    """
    try:
        for name in os.listdir(dir):
            listofimages.append(name)
    except OSError as exc:
        print exc.strerror


def copyfiles(source_dir, dest_dir, listofimages,  next_image_number, flip = False, crop=True, resize = True, newsize= (224,224) ):
    """
    :param source_dir:   where images_all come from  should be settings.images_directory
    :param dest_dir:   where images_all go to
    :param listofimages: list of images_all to write to directory of form [111,223,396...]
    :param next_image_number: the next 'free' number to name an image
    :param flip: whether to create a flipped version of an image
    :param crop = True,
    :param resize = True,
    :param newsize= (224,224)   default to resnet size
    :return: next_image_number - mutated in this function so return it

    """
    for img_path in listofimages:
        fn = "/" + img_path + ".png"
        try:
            src_path_img = source_dir + fn
            dest_path_img = dest_dir + fn

            #lets get the image and process it
            tmp_img = cv2.imread(src_path_img)
            tmp_img = preprocess(tmp_img, crop, resize, newsize)

            #now save it
            cv2.imwrite(dest_path_img, tmp_img)

            #and its flipped counterpart if enabled
            if flip == True:
                dest_path_img_flipped = dest_dir + "/" + str(next_image_number) + ".png"
                tmp_img = cv2.flip(tmp_img, 1)
                cv2.imwrite(dest_path_img_flipped, tmp_img)
                next_image_number += 1

        except OSError as exc:
            print exc.strerror

        #return number of files in this directory
    return next_image_number

def duplicatefiles(dir, requiredNumberImagesInDir, next_image_number):
    """
    for unbalanced datasets (lots of positive few negative) you can even it out by duplicating the negatives until
    the number equals the positives
    :param dir:
    :param requiredNumberImagesInDir:
    :return: next_image_number - mutated in this function so return it
    """
    #get list of files and the total number
    listFiles = [name for name in os.listdir(dir)]
    numbfilesinDir = len(listFiles)

    #sanity check
    if numbfilesinDir >= requiredNumberImagesInDir:
        return

    while(True):
        for file in listFiles:
            #read
            tmp_img = cv2.imread(dir + "/" + file)

            #generate newfile
            new_path_img = dir + "/" + str(next_image_number) + ".png"

            #save to newfile
            cv2.imwrite(new_path_img, tmp_img)

            next_image_number += 1
            numbfilesinDir += 1
            if (numbfilesinDir >= requiredNumberImagesInDir):
                return next_image_number



import settings
valid_values = ["normal", "opacity", "cardiomegaly", "calcinosis", "lung/hypoinflation", "calcified granuloma",
         "thoracic vertebrae/degenerative", "lung/hyperdistention", "spine/degenerative ", "catheters, indwelling",
         "granulomatous disease", "nodule", "surgical instruments", "scoliosis", "spondylosis"]

#=======================================================================================================================
# #create a list of all the diagnosis that are one of the above values, otherwise its "missing"
#  also figure out how many are nodule and normal as these are the images_all to train on
#=======================================================================================================================
all_diagnosis = {}
def get_items(val):
    global numb_nodule,numb_normal
    tmp =  "/".join(val['items']).lower()
    retval = ""
    for val in valid_values:
        if val in tmp:
            retval +=val + " "
    retval = retval.rstrip()
    return retval if retval != "" else "missing"

with open(settings.json_data_file) as data_file:
    alldata = json.load(data_file)
    all_diagnosis = {k:get_items(v) for k,v in alldata.iteritems() }

total_images = len(all_diagnosis)
print "number initial records " + str(total_images)

#=======================================================================================================================
#calculate unique combos of above
#=======================================================================================================================
from collections import Counter
unique_combos = Counter((all_diagnosis.values()))
unique_combos.items().sort(key=lambda x: x[0])
unique_combos = dict(unique_combos)
print "number unique combos " + str(len(unique_combos))

#sort by number cases, show last 4
from operator import itemgetter
sort = sorted(unique_combos.iteritems(), key=itemgetter(1))
print sort[-40:]
labels = sorted([i for i in sort[-4:] if i[0]!="missing"], key=lambda x: x[1])
print labels

#=======================================================================================================================
#find all indexes where a particular disease occurs of form { 'diagnosis':[1111,2222,3345...images_all]}
#=======================================================================================================================
from collections import defaultdict
index_list = defaultdict(list)
for key, value in all_diagnosis.iteritems():
    index_list[value].append(key)

#=======================================================================================================================
#Training and testing list of form
# [normal indices]
# [nodule indices]
# =======================================================================================================================
normal_images_list = []
nodule_images_list = []

for i in index_list:
    if "normal"in i and "nodule" in i:
        print(" WARNING-Throwing out " + str(len(index_list[i])) +
             " values, has both nodule and normal in " + i )
    elif "normal" in i:
        normal_images_list += index_list[i]
        #print( " normal in :"+ i + "  numb:"+ str(len(index_list[i])))
    # elif i=="opacity":
    #         train_images_dict["abnormal"]=index_list[i][:354]
    #         test_images_dict["abnormal"] = index_list[i][354:374]
    #     elif i=="cardiomegaly":
    #         train_images_dict["abnormal"]=index_list[i][:251]
    #         test_images_dict["abnormal"] = index_list[i][251:266]
    #     elif i=="lung/hypoinflation":
    #         train_images_dict["abnormal"]=index_list[i][:229]
    #         test_images_dict["abnormal"] = index_list[i][229:249]
    #     elif i=="calcified granuloma":
    #         train_images_dict["abnormal"]+=index_list[i][:243]
    #         test_images_dict["abnormal"] += index_list[i][243:263]
    #     elif i=="thoracic vertebrae/degenerative":
    #         train_images_dict["abnormal"]+=index_list[i][:218]
    #         test_images_dict["abnormal"] += index_list[i][218:238]
    #     elif i=="lung/hyperdistention":
    #         train_images_dict["abnormal"]+=index_list[i][:190]
    #         test_images_dict["abnormal"] += index_list[i][190:210]
    #     elif i=="surgical instruments":
    #         train_images_dict["abnormal"]+=index_list[i][:71]
    #         test_images_dict["abnormal"] += index_list[i][71:86]
    #     elif i=="catheters, indwelling":
    #         train_images_dict["abnormal"]+=index_list[i][:100]
    #         test_images_dict["abnormal"] += index_list[i][100:112]
    #     elif i=="calcinosis":
    #         train_images_dict["abnormal"]+=index_list[i][:146]
    #         test_images_dict["abnormal"] += index_list[i][146:166]
    #elif i == "nodule" or i== "calcinosis nodule":
    elif "nodule" in i:
        nodule_images_list += index_list[i]
        #print(" nodule in :" + i + "  numb:" + str(len(index_list[i])))

numb_nodule=len(nodule_images_list)
numb_normal=len(normal_images_list)
print("Number nodule: " + str(numb_nodule))
print("Number normal: " + str(numb_normal))
print "if normal and nodule are not approximately equal then dataset is unbalanced"

if settings.run_test:
    print "Running test on 500 images"
    numb_nodule = settings.vals_10
    numb_normal = settings.vals_90
    nodule_images_list = nodule_images_list[:numb_nodule]
    normal_images_list = normal_images_list[:numb_normal]
# =======================================================================================================================
#create separate directories for above classes and clear them if needed
#part of making .rec file for mxnet
#../images_rec/nodule and normal
# =======================================================================================================================
#create directories
import os, shutil
def makeDir(dir):
    try:
        os.mkdir(dir)
    except OSError as exc:
        print("error making "+ dir + "-"+exc.strerror)

def clear_folder_files(folder):
    for file in os.listdir(folder):
        file_path = os.path.join(folder, file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
                # elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e:
            print(e)

makeDir(settings.mxnet_images_train_dir)
makeDir(settings.mxnet_images_train_nodule_dir)
makeDir(settings.mxnet_images_train_normal_dir)

makeDir(settings.mxnet_images_val_dir)
makeDir(settings.mxnet_images_val_nodule_dir)
makeDir(settings.mxnet_images_val_normal_dir)

makeDir(settings.mxnet_images_test_dir)
makeDir(settings.mxnet_images_test_nodule_dir)
makeDir(settings.mxnet_images_test_normal_dir)

# clear existing image data in
if (settings.clear_subdirs == True):
    clear_folder_files(settings.mxnet_images_train_nodule_dir)
    clear_folder_files(settings.mxnet_images_train_normal_dir)
    clear_folder_files(settings.mxnet_images_val_nodule_dir)
    clear_folder_files(settings.mxnet_images_val_normal_dir)
    clear_folder_files(settings.mxnet_images_test_nodule_dir)
    clear_folder_files(settings.mxnet_images_test_normal_dir)

import cv2

images_in_noduledir =[]
next_image_number = total_images+1

#=================================================================
#figure out how many images go in each split
#=================================================================
normal_train = int(settings.trainpercent * numb_normal)
normal_test = int(settings.testpercent * numb_normal)
normal_val = int(settings.valpercent * numb_normal)

#if any leftover add back to train
leftover = numb_normal - normal_train - normal_test - normal_val
normal_train += leftover

nodule_train = int(settings.trainpercent * numb_nodule)
nodule_test  = int(settings.testpercent  * numb_nodule)
nodule_val   = int(settings.valpercent   * numb_nodule)

#if any leftover add back to train
leftover = numb_nodule - nodule_train - nodule_test - nodule_val
nodule_train += leftover

#now figure out how to balance normal and nodule
#or how many more to add to nodule
diff = numb_normal - numb_nodule
diff_train = int(settings.trainpercent * diff)
diff_test  = int(settings.testpercent  * diff)
diff_val   = int(settings.valpercent   * diff)

leftover = diff - diff_train - diff_test - diff_val
diff_train += leftover

#=================================================================
#create train_list
#=================================================================
normal_train_list = normal_images_list[0:normal_train]
normal_test_list = normal_images_list[normal_train:normal_train+ normal_test]
normal_val_list = normal_images_list[normal_train+ normal_test :]

nodule_train_list = nodule_images_list[0:nodule_train]
nodule_test_list = nodule_images_list[nodule_train:nodule_test+nodule_train]
nodule_val_list = nodule_images_list[nodule_test+nodule_train :]

# =======================================================================================================================
#copy in nodule and normal files to ../images_rec/nodule and normal
#note that this is an unbalanced dataset (10 to 1) with 211 nodules and 2706 normals
# =======================================================================================================================
next_image_number = copyfiles(settings.images_directory, settings.mxnet_images_test_nodule_dir, nodule_test_list,
          next_image_number, flip = True, crop=True, resize = True, newsize= (224,224) )
next_image_number =  copyfiles(settings.images_directory, settings.mxnet_images_val_nodule_dir, nodule_val_list,
          next_image_number, flip = True, crop=True, resize = True, newsize= (224,224) )
next_image_number =  copyfiles(settings.images_directory, settings.mxnet_images_train_nodule_dir, nodule_train_list,
          next_image_number, flip = True, crop=True, resize = True, newsize= (224,224) )

next_image_number =  copyfiles(settings.images_directory, settings.mxnet_images_test_normal_dir, normal_test_list,
          next_image_number, flip = False, crop=True, resize = True, newsize= (224,224) )
next_image_number =  copyfiles(settings.images_directory, settings.mxnet_images_val_normal_dir, normal_val_list,
          next_image_number, flip = False, crop=True, resize = True, newsize= (224,224) )
next_image_number = copyfiles(settings.images_directory, settings.mxnet_images_train_normal_dir, normal_train_list,
          next_image_number, flip = False, crop=True, resize = True, newsize= (224,224) )

#now duplicate the nodule files
next_image_number =  duplicatefiles(settings.mxnet_images_test_nodule_dir, nodule_test + diff_test, next_image_number)
next_image_number =  duplicatefiles(settings.mxnet_images_val_nodule_dir, nodule_val + diff_val, next_image_number)
next_image_number =  duplicatefiles(settings.mxnet_images_train_nodule_dir, nodule_train + diff_train, next_image_number)

# =======================================================================================================================
#convert normal and nodule to 0 and 1
#IS THIS EVEN USED?
# =======================================================================================================================
import numpy as np
label_dict = {"normal":np.array([1.0,0.0]),"nodule":np.array([0.0,1.0])}

# new_image_dict={}
# for i in all_diagnosis:
#     if all_diagnosis[i] in label_index.keys():
#         new_image_dict[i] = label_index[all_diagnosis[i]]
# new_image_dict={}
# for i in all_diagnosis:
#     if all_diagnosis[i] =="normal":
#         new_image_dict[i] = np.array([1,0])
#     else:
#         new_image_dict[i] = np.array([0,1])