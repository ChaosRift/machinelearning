# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 10:16:04 2022

@author: Craig
"""

from os import listdir, remove
from os.path import isfile, join
from PIL import Image
import hashlib

originalDir = "data\\train_landscape"
#originalDir = "data\\new_data"
#originalDir = "train_portrait"

alldict = dict()
hashdict = dict()

allfiles = [f for f in listdir(originalDir) if isfile(join(originalDir, f))]

dups = 0
for file in allfiles:
    alldict[file] = 1
    hashedImage = hashlib.md5(Image.open(join(originalDir, file)).tobytes()).hexdigest()
    if hashedImage not in hashdict:
        print(hashedImage)
        hashdict[hashedImage] = file
    else:
        print(f"duplicate found {hashedImage}, file {file} matches {hashdict[hashedImage]}")
        remove(join(originalDir, file))
        dups += 1
print(f"{dups} duplicates found.")
