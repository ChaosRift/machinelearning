# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 10:16:04 2022

@author: Craig
"""

from os import listdir, remove, rename
from os.path import isfile, join
from PIL import Image
import hashlib

originalDir = "datasets\\dalle2"
#originalDir = "datasets\\train_landscape"

hashdict = dict()

allfiles = [f for f in listdir(originalDir) if isfile(join(originalDir, f))]

maxdimension = 2048

def resizeImg(image):
    if image.height > image.width: # It is portrait
        percentage = maxdimension / image.height
        width = int(image.width * percentage)
        image = image.resize((width, maxdimension))
    else:
        percentage = maxdimension / image.width
        height = int(image.height * percentage)
        image = image.resize((maxdimension, height))
    
    return image
    
resized = 0
dups = 0
for file in allfiles:
    try:
        image = Image.open(join(originalDir, file))
    except:
        print(f"cannot open {file} deleting")
        remove(join(originalDir, file))
        
    hashedImage = hashlib.md5(image.tobytes()).hexdigest()
    if hashedImage not in hashdict:
        extension = file.split('.')[-1].lower()
        if extension == "jpeg":
            extension = "jpg"
        hashedName = hashedImage + "." + extension
        if file == hashedName:
            hashdict[hashedImage] = hashedName
        # if hashed and copied instead of renamed.
        elif(isfile(join(originalDir, hashedName)) and file != hashedName):  
            print(f"File {hashedName} exists")
            remove(join(originalDir, file))
        else:
            if(isfile(join(originalDir, hashedName)) and file != hashedName):
               print(f"file {file} already hashed")
               remove(join(originalDir, file))
            else:
               rename(join(originalDir, file), join(originalDir, hashedName))
               hashdict[hashedImage] = hashedName
    else:
        print(f"duplicate found {hashedImage}, file {file} matches {hashdict[hashedImage]}")
        remove(join(originalDir, file))
        dups += 1
        
    if image.width > maxdimension or image.height > maxdimension:
        image = resizeImg(image)
        hashedImage = hashlib.md5(image.tobytes()).hexdigest()
        hashedName = hashedImage + "." + extension
        hashdict[hashedImage] = hashedName
        image.save(join(originalDir, hashedName))
        resized = resized + 1
        if isfile(join(originalDir, file)):
            remove(join(originalDir, file))
        print("New image size:", image.width, image.height)
        
print(f"{dups} duplicates found. {resized} images resized.")

