# change result in csv format to coco online evaluation format
"""
[{
"image_id" : int, "caption" : str,
}]
"""

import csv
import os
import json
from tqdm import tqdm

filepath = "/media/jhlim/HDD8TB/Home/git/python/image_captioning-master/test/results.csv"
jsonpath = "/media/jhlim/HDD8TB/Data/mscoco/annotations/captions_train2014.json"
savepath = "/media/jhlim/HDD8TB/Home/git/python/image_captioning-master/test/captions_val2014_maskcaption_results.json"

onlinecocojson = []
jsonfile = json.load(open(jsonpath))
jsonimg= jsonfile['images']
jsoncaption = {}
# convert list to dict
for im in jsonimg:
    jsoncaption[im['file_name']] = im['id']

with open(filepath, mode='r') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    line_count = 0
    for row in tqdm(csv_reader):
        caption = row['caption']
        imgfile = row['image_files']
        filename = str.split(imgfile, '/')[-1]
        imid = jsoncaption[filename]
        cocojson = {}
        cocojson['image_id'] = imid
        cocojson['caption'] = caption
        onlinecocojson.append(cocojson)

with open(savepath, "w") as write_file:
    json.dump(onlinecocojson, write_file)

