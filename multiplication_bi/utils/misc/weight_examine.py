# examine the mean of the weight

import numpy as np
a = np.load('/media/jhlim/HDD8TB/Home/git/python/image_captioning-master/models/resnet_inversemask_kpt_voc10k_32_134999_0.00001/284999.npy')
b = a.item()
total = 0
count = 0
for x in b:
  if "optimizer" not in x and "mask" in x and ("conv1" in x or "pool1" in x or "res" in x or "bn" in x):
    #print(x)
    v = b[x]
    total += np.mean(v)
    count += 1
mask = total/count

total = 0
count = 0
for x in b:
  if "optimizer" not in x and "mask" not in x and ("conv1" in x or "pool1" in x or "res" in x or "bn" in x):
    #print(x)
    v = b[x]
    total += np.mean(v)
    count += 1
nomask = total/count
print(mask)
print(nomask)
