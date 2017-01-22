#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas
import caffe

caffe.set_mode_cpu()

# read data
df = pandas.read_csv('./test.csv', dtype=np.float64)
data = df.values

# preprocessing
net = caffe.Classifier('./net_depl.prototxt', '_iter_500.caffemodel')
data = data.reshape((28000,28,28,1))

# predict
with open('submit.csv', 'w') as f:
	f.write('ImageId, Lable\n')
	for i in range(28000):
		probs = net.predict(np.array([data[i]]), oversample=False)
		r = np.argmax(probs)
		f.write(str(i + 1) + ',' + str(r) + '\n')
