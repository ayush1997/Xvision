domain = 'https://openi.nlm.nih.gov'
json_data_file = 'data_new.json'

images_directory = "../images_all"
#images_directory = "./images_all"

clear_subdirs = True   #clear the following subdirs before copying
mxnet_images_train_dir = "../images_Train"
mxnet_images_train_nodule_dir = mxnet_images_train_dir + "/nodule"
mxnet_images_train_normal_dir = mxnet_images_train_dir + "/normal"
mxnet_images_val_dir = "../images_Val"
mxnet_images_val_nodule_dir = mxnet_images_val_dir + "/nodule"
mxnet_images_val_normal_dir = mxnet_images_val_dir + "/normal"
mxnet_images_test_dir = "../images_Test"
mxnet_images_test_nodule_dir = mxnet_images_test_dir + "/nodule"
mxnet_images_test_normal_dir = mxnet_images_test_dir + "/normal"

number_pages = 75

#test with just 100 images, much-much faster
run_test = False
vals_90 =90
vals_10 =10
trainpercent = .7
valpercent = .15
testpercent = .15