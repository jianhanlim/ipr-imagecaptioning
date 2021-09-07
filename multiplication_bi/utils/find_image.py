# examine the mean of the weight

import numpy as np
import json
import string

a = json.load(open('../val/karpathy_test.json'))
tofind = "a man in a suit on a plaza, holding a blue umbrella in the rain."

for x in a['annotations']:
	c = x['caption']
	if string.lower(c) in string.lower(tofind):
		print(x)
		imid = x['image_id']
		for i in a['images']:
			if imid == i['id']:
				print(i)
