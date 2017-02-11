

import json
from pprint import pprint
import logging
logging.basicConfig(filename='example.log',level=logging.DEBUG)
logging.debug('This message should go to the log file')
logging.info('So should this')
logging.warning('And this, too')

#import os
#import test1
# cmd = "python ./test1.py -o 'THIS STRING WILL PRINT'"
# out_str = os.system(cmd)
# =======================================================================================================================
# size the image?
# =======================================================================================================================
from skimage.transform import resize as imresize
import numpy as np
mean_rgb = np.array([123.68, 116.779, 103.939])
mean_rgb = mean_rgb.reshape((3, 1, 1))
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

    #convert back to ints
    return np.clip(norm_img * 255, 0, 255).astype(np.uint8)

from skimage import io, transform
def PreprocessImage(img, show_img=False):
    print("Original Image Shape: ", img.shape)
    # we crop image from center
    short_egde = min(img.shape[:2])
    yy = int((img.shape[0] - short_egde) / 2)
    xx = int((img.shape[1] - short_egde) / 2)
    crop_img = img[yy : yy + short_egde, xx : xx + short_egde]
    # resize to 224, 224
    resized_img = transform.resize(crop_img, (224, 224))
    if show_img:
        io.imshow(resized_img)
    # convert to numpy.ndarray
    sample = np.asarray(resized_img) * 256
    # swap axes to make image from (224, 224, 3) to (3, 224, 224)
    sample = np.swapaxes(sample, 0, 2)
    sample = np.swapaxes(sample, 1, 2)
    # sub mean
    normed_img = sample - mean_rgb
    print normed_img.shape, sample.shape, mean_rgb.shape
    normed_img = normed_img.reshape((1, 3, 224, 224))
    return normed_img


import numpy as np
#flip an image horizontally
import cv2
img=cv2.imread('./images_all/2684.png')

img2 = PreprocessImage(img)
img3 = preprocess(img)

cv2.imwrite("img2.png",img2 )
cv2.imwrite("img3.png",img3 )

img5=cv2.imread("img2.png")

img_flip=cv2.flip(img,1)
cv2.imwrite("2684flipped_224_by_224.jpg",img_flip )


# cv2.imshow("Original", img)
# cv2.imshow("vertical flip", rimg)
# cv2.imshow("horizontal flip", fimg)
# cv2.waitKey(0)
# cv2.destroyAllWindows()




# s = "-";
# seq = ("a", "b", "c"); # This is sequence of strings.
# print s.join( seq )
#
# my_list = ["Hello", "world", "beans"]
# print "-".join(my_list)
#
# dic = {}
# with open('./pythoncode/datas.json') as data_file:
#     data = json.load(data_file)
#     # lst = []
#     # n=0
#     print len(data)
#     for  i in data:
#         # print i
#         str = data[i]['items']
#         a= "-".join(data[i]['items'])
#         # b = (data[i]['items'])
#         dic[i] = "/".join(data[i]['items'])