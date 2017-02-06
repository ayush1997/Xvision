#!/usr/bin/env python

import json
from pprint import pprint
import numpy as np

#globals
from scipy.misc import imresize

import settings

valid_values = ["normal", "opacity", "cardiomegaly", "calcinosis", "lung/hypoinflation", "calcified granuloma",
         "thoracic vertebrae/degenerative", "lung/hyperdistention", "spine/degenerative ", "catheters, indwelling",
         "granulomatous disease", "nodule", "surgical instruments", "scoliosis", "spondylosis"]

#=======================================================================================================================
# #create a list of all the diagnosis that are one of the above values, otherwise its "missing"
#  also figure out how many are nodule and normal as these are the images to train on
#=======================================================================================================================
all_diagnosis = {}
numb_normal =0
numb_nodule =0

def get_items(val):
    global numb_nodule,numb_normal
    tmp =  "/".join(val['items']).lower()
    retval = ""
    for val in valid_values:
        if val in tmp:
            if val == "normal":
                numb_normal+=1
            elif val == "nodule":
                numb_nodule+=1
            retval +=val + " "
    retval = retval.rstrip()
    return retval if retval != "" else "missing"

with open(settings.json_data_file) as data_file:
    alldata = json.load(data_file)
    all_diagnosis = {k:get_items(v) for k,v in alldata.iteritems() }

total_images = len(all_diagnosis)
print "number initial records " + str(total_images)
print "number nodule records " + str(numb_nodule)
print "number normal records " + str(numb_normal)
print "if normal and nodule are not approximately equal then dataset is unbalanced"

#lets see if we have an unbalanced dataset (normal-nodule)/(normal + nodule)
#if 0 then balanced if .33 = 2 to 1, if > .33 more unbalanced
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
#find all indexes where a particular disease occurs of form { 'diagnosis':[1111,2222,3345...images]}
#=======================================================================================================================
from collections import defaultdict
index_list = defaultdict(list)
for key, value in all_diagnosis.iteritems():
    index_list[value].append(key)

totnod = 0;
totnorm =0
for key,val in unique_combos.iteritems():
    if "nodule" in key:
        print "nodule in " + key + " :#values " + str(val)
        totnod = totnod + val
    elif "normal" in key:
        totnorm= totnorm + val

print "total nodule = "+ str(totnod)
print "total normal = "+ str(totnorm)
# =======================================================================================================================
#Training and testing dictionaries of the form
# normal, [normal indices]
# nodule, [nodule indices]
# =======================================================================================================================
train_images_dict = defaultdict(list)
test_images_dict = defaultdict(list)

def get_train_test_sizes(i):
    length = len(index_list[i])
    if settings.run_test == True and  length >= settings.run_test_100:
        length = settings.run_test_100

    train_size = int(settings.trainpercent * length)
    test_size = int(settings.testpercent * length)

    leftover = length-(train_size + test_size)

    #pick one to add this 1 to
    train_size += leftover

    return train_size, test_size

for i in index_list:
    if i == "normal":
        train_size,test_size = get_train_test_sizes(i)
        train_images_dict["normal"] = index_list[i][0:train_size]
        test_images_dict["normal"]  = index_list[i][train_size:train_size+test_size]
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
        train_size, test_size = get_train_test_sizes(i)
        train_images_dict["nodule"] += index_list[i][0:train_size]
        test_images_dict["nodule"] += index_list[i][train_size:train_size+test_size]
print "Number nodules train " + str(len(train_images_dict["nodule"]))
print "Number normal train " + str(len(train_images_dict["normal"]))
print "Number nodules test" + str(len(test_images_dict["nodule"]))
print "Number normal test" + str(len(test_images_dict["normal"]))
# =======================================================================================================================
#create separate directories for above classes
#part of making .rec file for mxnet
#../images_rec/nodule and normal
# =======================================================================================================================
#create directories
import os
try:
    os.mkdir(settings.mxnet_rec_dir )
except OSError as exc:
    print("imagesrecdir")
    print exc.strerror

try:
    os.mkdir(settings.nodule_directory )
except OSError as exc:
    print("nodule directory")
    print exc.strerror

try:
    os.mkdir(settings.normal_directory )
except OSError as exc:
    print("normal directory exists")
    print exc.strerror

# =======================================================================================================================
# clear existing image data in ../images_rec/nodule and normal?
# =======================================================================================================================
import os, shutil
def clear_folder_files(folder):
    for file in os.listdir(folder):
        file_path = os.path.join(folder, file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
                # elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e:
            print(e)
if (settings.clear_subdirs == True):
    clear_folder_files(settings.normal_directory)
    clear_folder_files(settings.nodule_directory)
# =======================================================================================================================
# size the image?
# =======================================================================================================================
from skimage.transform import resize as imresize
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

    # convert back to ints
    return np.clip(norm_img * 255, 0, 255).astype(np.uint8)
# =======================================================================================================================
# copy in nodule and normal files to ../images_rec/nodule and normal
#note that this is an unbalanced dataset (10 to 1) with 211 nodules and 2706 normals
# =======================================================================================================================

#nodule balance and copy to ../images_rec/nodule
#first copy them all to folder
#flip all nodules horizontally (go from 211 to 422)
#then copy nodule set 6.4 times (6.4*422 = 2700)
#roughly balanced then
import cv2

#copy and flip
images_in_noduledir =[]
next_image_number = total_images+1
import shutil
for img_path in train_images_dict["nodule"]:
    fn = "/" + img_path + ".png"
    try:
        orig_image_path = settings.images_directory + fn
        copied_image_path = settings.nodule_directory + fn
        copied_and_flipped_image_path = settings.nodule_directory + "/" + str(next_image_number) + ".png"

        #lets get the image and resize it resnet expects 224,224
        tmp_img = cv2.imread(orig_image_path)
        tmp_img = preprocess(tmp_img, crop=True, resize=True, dsize=(224, 224))

        #now save it and its flipped counterpart
        cv2.imwrite(copied_image_path, tmp_img)

        tmp_img = cv2.flip(tmp_img, 1)
        cv2.imwrite(copied_and_flipped_image_path, tmp_img)

        images_in_noduledir.append(copied_image_path)
        images_in_noduledir.append(copied_and_flipped_image_path)
        next_image_number += 1

    except OSError as exc:
        print exc.strerror

#how many files in dir
_, _, files = os.walk(settings.nodule_directory).next()
file_count = len(files)

#how many times do we duplicate dataset
numb_times_to_duplicate = numb_normal/(2*numb_nodule) + 1
while(file_count != numb_normal):
    for img_path in images_in_noduledir:
        copied_image_path = settings.nodule_directory + "/" + str(next_image_number) + ".png"

        shutil.copy(img_path, copied_image_path)

        next_image_number+=1
        file_count+=1
        if (file_count == numb_normal):
            break

# copy normal files to ../images_rec/normal
for img_path in train_images_dict["normal"]:
    fn = "/" + img_path + ".png"
    try:
        orig_image_path = settings.images_directory + fn
        copied_image_path = settings.normal_directory + fn

        # lets get the image and resize it resnet expects 224,224
        tmp_img = cv2.imread(orig_image_path)
        tmp_img = preprocess(tmp_img, crop=True, resize=True, dsize=(224, 224))

        # now save it
        cv2.imwrite(copied_image_path, tmp_img)
    except OSError as exc:
        print exc.strerror
# =======================================================================================================================
#convert normal and nodule to 0 and 1

# IS THIS EVEN USED?
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