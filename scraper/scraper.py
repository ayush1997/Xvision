from lxml import html
import requests
import re
import json
import urllib
import sys

path = sys.argv[1]
if path[-1]!="/":
    path+="/"


domain = 'https://openi.nlm.nih.gov/'
url_list = []
for i in range(0,75):
    url = 'https://openi.nlm.nih.gov/gridquery.php?q=&it=x,xg&sub=x&m='+str(1+100*i)+'&n='+str(100+100*i)
    url_list.append(url)
regex = re.compile(r"var oi = (.*);")
final_data = {}
img_no = 0


def extract(url):
    global img_no

    img_no += 1
    r = requests.get(url)
    tree = html.fromstring(r.text)

    div = tree.xpath('//table[@class="masterresultstable"]\
        //div[@class="meshtext-wrapper-left"]')

    if div != []:
        div = div[0]
    else:
        return

    typ = div.xpath('.//strong/text()')[0]
    items = div.xpath('.//li/text()')
    img = tree.xpath('//img[@id="theImage"]/@src')[0]


    final_data[img_no] = {}
    final_data[img_no]['type'] = typ
    final_data[img_no]['items'] = items
    final_data[img_no]['img'] = domain + img
    urllib.urlretrieve(domain+img, path+str(img_no)+".png")
    with open('data_new.json', 'w') as f:
        json.dump(final_data, f)
    print final_data[img_no]


def main():
    for url in url_list :
        r = requests.get(url)
        tree = html.fromstring(r.text)

        script = tree.xpath('//script[@language="javascript"]/text()')[0]

        json_string = regex.findall(script)[0]
        json_data = json.loads(json_string)

        next_page_url = tree.xpath('//footer/a/@href')

        print 'extract'
        links = [domain + x['nodeRef'] for x in json_data]
        for link in links:
            extract(link)

if __name__ == '__main__':

    main()


#python scraper.py <path to folders>
