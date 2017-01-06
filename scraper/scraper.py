from lxml import html
import requests
import re
import json
import urllib
import sys
#this script parses out the first 76 pages at  https://openi.nlm.nih.gov/gridquery.php?q=&it=x,xg&sub=x
#each page contains 100 images with associated data about each
#then for each of those pages it parses out various bits about each 100 images and then dumps that info to a json file
#this file is then used by

path = sys.argv[1]
if path[-1]!="/":
    path+="/"

#globals
domain = 'https://openi.nlm.nih.gov/'

#list of urls, each should have ~100 cases per page
url_list = []
for i in range(0,75):
    url = 'https://openi.nlm.nih.gov/gridquery.php?q=&it=x,xg&sub=x&m='+str(1+100*i)+'&n='+str(100+100*i)
    url_list.append(url)

#used to quickly ferret out data using regular expressions
regex = re.compile(r"var oi = (.*);")
final_data = {}
img_cnt = 0


def extract(url):
    '''
    url points to a page with a single medical case, this function extracts where image is and some metadata for later
    data is stored in a JSON file, this file will be used to download the images
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
    final_data[img_cnt]['img'] = domain + img

    #get the image and copy it to filesystem, presumably image 1 corresponds to the first json entry.
    #ineffecient since it writes the entire list to the file every time this function is called
    #it should only be done once, at the end of main, on the other hand if something crashes you will have the latest
    #json file that corresponds to all the images downloaded
    urllib.urlretrieve(domain + img, path + str(img_cnt) + ".png")
    with open('data_new1.json', 'w') as f:
        json.dump(final_data, f)
    print final_data[img_cnt]


def main():
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

        print 'extract'
        #lets get the page of data and complete FS image for a particular case
        links = [domain + x['nodeRef'] for x in json_data]
        for link in links:
            extract(link)

if __name__ == '__main__':

    main()


# #python scraper.py <path to folders>
