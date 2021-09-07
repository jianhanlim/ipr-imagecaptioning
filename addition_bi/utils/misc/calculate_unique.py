from tqdm import tqdm
import os, time, json
import numpy as np

#label_path = "/media/jhlim/HDD8TB/Data/flickr30k/annotations/karpathy_train_flickr30k.json"
#result_path = "/media/jhlim/HDD8TB/Home/git/python/image_captioning2/val/results_flickr30k_resnet_mask_kpt_voc10k_32_flickr30k_finetune_29999_0.00001_44999.json"

label_path = "/media/jhlim/HDD8TB/Data/mscoco/annotations/karpathy_train.json"
result_path = "/media/jhlim/HDD8TB/Home/git/python/image_captioning2/val/results_resnet_ori_kpt_voc10k_finetune_9999'_304999.json"

label_file = json.load(open(label_path))
result_file = json.load(open(result_path))
label_file = label_file['annotations']
label_captions = []
result_captions = []
for x in tqdm(label_file):
	label_captions.append(x['caption'].lower())

for x in tqdm(result_file):
	result_captions.append(x['caption'].lower())

appear = 0
count = 0
s_length = 0
for c in tqdm(result_captions):
	count += 1
	s_length += len(c.split(' '))
	for l in label_captions:
		if c in l:
			appear += 1
			break

print("Average Length: {}".format(s_length/float(count)))
print("Unique: {}".format((count-appear)/float(count)))
