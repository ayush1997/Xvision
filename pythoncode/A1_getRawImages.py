#!/usr/bin/env python
from lxml import html
import requests
import re
import json
import urllib
import os

#this script parses out the 75 pages at  https://openi.nlm.nih.gov/gridquery.php?q=&it=x,xg&sub=x
#each page contains 100 images with associated data about each
#then for each of those pages it parses out various bits about each 100 images and then dumps that info to a json file
#it also will download each image and dump it to an images directory,
# this takes a very long time and uses a lot of space

#globals
import settings

#list of urls, each should have ~100 cases per page
url_list = []
for i in range(0,settings.number_pages):
    url = 'https://openi.nlm.nih.gov/gridquery.php?q=&it=x,xg&sub=x&m='+str(1+100*i)+'&n='+str(100+100*i)
    url_list.append(url)

#used to quickly ferret out data using regular expressions
regex = re.compile(r"var oi = (.*);")
final_data = {}
img_cnt = 0


def extract(url,imgDir):
    '''
    url points to a page with a single medical case, this function extracts where image is and some metadata for later
    data is stored in a JSON file, this file will be used to download the images_all
    :param url:
    :return:
    '''
    global img_cnt
    img_cnt += 1

    r = requests.get(url)
    tree = html.fromstring(r.text)

    #all info we want is in the following table
    div = tree.xpath('//table[@class="masterresultstable"]\
        //div[@class="meshtext-wrapper-left"]')

    if div == []:
        return

    div = div[0]

    typ = div.xpath('.//strong/text()')[0]
    items = div.xpath('.//li/text()')
    img = tree.xpath('//img[@id="theImage"]/@src')[0]

    #save a copy of the relevant info
    final_data[img_cnt] = {}
    final_data[img_cnt]['type'] = typ
    final_data[img_cnt]['items'] = items
    final_data[img_cnt]['img'] = settings.domain + img

    #get the image and copy it to filesystem, presumably image 1 corresponds to the first json entry.
    #ineffecient since it writes the entire list to the file every time this function is called
    #it should only be done once, at the end of main, on the other hand if something crashes you will have the latest
    #json file that corresponds to all the images downloaded
    src_img = settings.domain + img
    dst_img = os.path.join(imgDir, str(img_cnt) + ".png")
    print ("Getting:" + src_img + " Wrinting to:" + dst_img)

    urllib.urlretrieve(src_img, dst_img)
    with open(settings.json_data_file, 'w') as f:
        json.dump(final_data, f)
    print final_data[img_cnt]

def main():
    # first create the directory for lst and rec files
    root = os.path.abspath(os.path.dirname(__file__))
    imgDir = os.path.join(root, settings.images_directory)

    # make sure images directory exits
    settings.makeDir(settings.images_directory)

    for url in url_list :
        response_info = requests.get(url)
        tree = html.fromstring(response_info.text)

        #get the first occurrence of the data withen following tag (should be 100 entries)
        script = tree.xpath('//script[@language="javascript"]/text()')[0]

        #get what the first oi= points to (~100 entries)
        json_string = regex.findall(script)[0]

        #load 100 entries into json object
        json_data = json.loads(json_string)

        #get list of other pages, all anchors in the footer tag, get the href attribute
        next_page_url = tree.xpath('//footer/a/@href')

        #lets get the page of data and complete FS image for a particular case
        links = [settings.domain +"/"+ x['nodeRef'] for x in json_data]
        for link in links:
            extract(link,imgDir)

if __name__ == '__main__':
    main()


# #python A1_getRawImages.py <path to folders>
