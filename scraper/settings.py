domain = 'https://openi.nlm.nih.gov'
json_data_file = 'data_new.json'

images_directory = "../images"
#images_directory = "./images"
clear_subdirs = True   #clear the following subdirs before copying

mxnet_rec_dir = "../images_rec"
nodule_directory = mxnet_rec_dir + "/nodule"
normal_directory = mxnet_rec_dir + "/normal"

number_pages = 75

run_test = False
run_test_100 =100
trainpercent = .9
testpercent = .1