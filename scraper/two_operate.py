#!/usr/bin/env python

import json
from pprint import pprint

#globals
import settings

valid_values = ["normal", "opacity", "cardiomegaly", "calcinosis", "lung/hypoinflation", "calcified granuloma",
         "thoracic vertebrae/degenerative", "lung/hyperdistention", "spine/degenerative ", "catheters, indwelling",
         "granulomatous disease", "nodule", "surgical instruments", "scoliosis", "spondylosis"]

#=======================================================================================================================
# #create a list of all the diagnosis that are one of the above values, otherwise its "missing"
#=======================================================================================================================
all_diagnosis = {}
def get_items(val):
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

print "number initial records " + str(len(all_diagnosis))

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
#find all indexes where a particular disease occurs
#=======================================================================================================================
from collections import defaultdict
index_list = defaultdict(list)
for key, value in all_diagnosis.iteritems():
    index_list[value].append(key)

for key,val in unique_combos.iteritems():
    if "nodule" in key:
        print "nodule in " + key

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

    elif i == "nodule" or i== "calcinosis nodule":
        train_size, test_size = get_train_test_sizes(i)
        train_images_dict["nodule"] += index_list[i][0:train_size]
        test_images_dict["nodule"] += index_list[i][train_size:train_size+test_size]
print "Number nodules " + str(len(train_images_dict["nodule"]))
print "Number normal " + str(len(train_images_dict["normal"]))
# =======================================================================================================================
#create separate directories for above classes
#part of making .rec file for mxnet
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

#clear existing data?
#if (settings.clear_subdirs == True):

#copy files
import shutil
for img in train_images_dict["nodule"]:
    fn = "/" +img + ".png"
    try:
        shutil.copy(settings.images_directory + fn, settings.nodule_directory + fn)
    except OSError as exc:
        print exc.strerror

    for img in train_images_dict["normal"]:
        fn = "/" + img + ".png"
        try:
            shutil.copy(settings.images_directory + fn, settings.normal_directory + fn)
        except OSError as exc:
            print exc.strerror
# =======================================================================================================================
#convert normal and nodule to 0 and 1
# =======================================================================================================================
import numpy as np
label_dict = {"normal":np.array([1.0,0.0]),"nodule":np.array([0.0,1.0])}


new_image_dict={}
for i in all_diagnosis:
    if all_diagnosis[i] in label_index.keys():
        new_image_dict[i] = label_index[all_diagnosis[i]]
new_image_dict={}
for i in all_diagnosis:
    if all_diagnosis[i] =="normal":
        new_image_dict[i] = np.array([1,0])
    else:
        new_image_dict[i] = np.array([0,1])