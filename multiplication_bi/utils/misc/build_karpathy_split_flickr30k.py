import json
import os
from tqdm import tqdm

k_split_path = '/media/jhlim/HDD8TB/Data/caption_datasets/dataset_flickr30k.json'
save_dir = '/media/jhlim/HDD8TB/Data/flickr30k/annotations/'

k_split = json.load(open(k_split_path, 'rb'))

k_train = {'images':[], 'annotations':[]}
k_val = {'images':[], 'annotations':[]}
k_test = {'images':[], 'annotations':[]}

k_split_images = k_split['images']

ann_id = 0
for k_split_image in tqdm(k_split_images, desc='all'):
    img_id = k_split_image['imgid']
    split = k_split_image['split']
    filename = k_split_image['filename']
    sentences = k_split_image['sentences']

    # if split == restval or train, add to train
    # split == val, add to val
    # split == test, add to test

    selected_image = {}
    selected_ann = []

    selected_image["id"] = img_id 
    selected_image["file_name"] = filename

    for sen in sentences:
        ann = {}
        ann["id"] = ann_id
        ann["image_id"] = img_id
        ann["caption"] = sen['raw']
        selected_ann.append(ann)
        ann_id += 1

    if len(selected_ann) <= 0 or selected_image is None:
        print k_split_image
        exit(0)
    else:
        if split == 'test':
            k_test['images'].append(selected_image)
            k_test['annotations'].extend(selected_ann)
        elif split == 'val':
            k_val['images'].append(selected_image)
            k_val['annotations'].extend(selected_ann)
        else:   # restval or train
            k_train['images'].append(selected_image)
            k_train['annotations'].extend(selected_ann)

print(len(k_test['images']))
print(len(k_test['annotations']))
print(len(k_val['images']))
print(len(k_train['images']))

with open(os.path.join(save_dir, 'karpathy_train.json'), 'w+') as outfile:
    json.dump(k_train, outfile)

with open(os.path.join(save_dir, 'karpathy_test.json'), 'w+') as outfile:
    json.dump(k_test, outfile)

with open(os.path.join(save_dir, 'karpathy_val.json'), 'w+') as outfile:
    json.dump(k_val, outfile)

