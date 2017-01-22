#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas
import h5py
import numpy as np

N = 42000
C = 1
W = 28
H = 28

df = pandas.read_csv('./train.csv')
f = h5py.File('test.hdf5', 'w')

ds_label = f.create_dataset('label', (N,), dtype='i')
ds_label[:] = df['label']

ds_data = f.create_dataset('data', (N,C,W,H), dtype='i')
for i in range(N):
    ds_data[i,0,:,:] = df.values[i,1:].reshape((W,H))
