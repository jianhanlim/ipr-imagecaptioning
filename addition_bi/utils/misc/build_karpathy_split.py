import json
import os
from tqdm import tqdm

k_split_path = '/media/jhlim/HDD8TB/Data/caption_datasets/dataset_coco.json'
ori_train_path = '/media/jhlim/HDD8TB/Data/mscoco/annotations/captions_train2014.json'
ori_val_path = '/media/jhlim/HDD8TB/Data/mscoco/annotations/captions_val2014.json'
save_dir = '/media/jhlim/HDD8TB/Data/mscoco/annotations/'

k_split = json.load(open(k_split_path, 'rb'))
ori_train = json.load(open(ori_train_path, 'rb'))
ori_val = json.load(open(ori_val_path, 'rb'))

k_train = {'images':[], 'annotations':[]}
k_val = {'images':[], 'annotations':[]}
k_test = {'images':[], 'annotations':[]}

k_split_images = k_split['images']

for k_split_image in tqdm(k_split_images, desc='all'):
    cocoid = k_split_image['cocoid']
    split = k_split_image['split']

    if split == 'train':
        continue

    # if split == restval, add to train
    # split == val, add to val
    # split == test, add to test
    ori_val_images = ori_val['images']
    ori_val_annotations = ori_val['annotations']

    selected_image = None
    selected_ann = []

    for ori_val_image in tqdm(ori_val_images, desc='images'):
        if ori_val_image['id'] == cocoid:
            selected_image = ori_val_image
            break

    for ori_val_annotation in tqdm(ori_val_annotations, desc='anns'):
        if ori_val_annotation['image_id'] == cocoid:
            selected_ann.append(ori_val_annotation)

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

ori_train['images'].extend(k_train['images'])
ori_train['annotations'].extend(k_train['annotations'])
print(len(ori_train['images']))

with open(os.path.join(save_dir, 'karpathy_train.json'), 'w+') as outfile:
    json.dump(ori_train, outfile)

with open(os.path.join(save_dir, 'karpathy_test.json'), 'w+') as outfile:
    json.dump(k_test, outfile)

with open(os.path.join(save_dir, 'karpathy_val.json'), 'w+') as outfile:
    json.dump(k_val, outfile)

