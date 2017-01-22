#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas

with open('./predict.csv', 'r') as f:
	for line in f:
		line = line.strip()
		rows = line.split(',')
		print(str(int(rows[0]) + 1) + ',' + rows[1])
