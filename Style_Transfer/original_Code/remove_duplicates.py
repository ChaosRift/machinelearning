# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 10:16:04 2022

@author: Craig
"""

from os import listdir, remove
from os.path import isfile, join

originalDir = "data\\train_landscape"
#originalDir = "data\\train_portrait"
upscaledDir = "data\\train_upscaled"
alldict = dict()
upedfiles = []

allfiles = [f for f in listdir(originalDir) if isfile(join(originalDir, f))]
upedfiles = [f for f in listdir(upscaledDir) if isfile(join(upscaledDir, f))]

for file in allfiles:
    alldict[file] = 1
print(len(alldict))
print(len(upedfiles))

i = 0

for file in upedfiles:
    if file in alldict:
        print(join(originalDir, file))
        i = i + 1
        remove(join(originalDir, file))
print(i)