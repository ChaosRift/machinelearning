####################################################################
# Does what it says. Sorts images based on portrait or landscape. 
# Set the source directory. THe target directories will be created for you.
# Images will be renamed according to their HASH Id. 
# Renaming helps eliminate duplicate images. 

import os
from os import listdir, remove
from os.path import isfile, join, exists
import shutil
from PIL import Image
import hashlib

#fromDir = "data\\train2017"
fromDir = "datasets\\newImages"

portDir = f"{fromDir}_portrait"
lsDir   = f"{fromDir}_landscape"

if not exists(portDir):
    os.mkdir(portDir)
if not exists(lsDir):
    os.mkdir(lsDir)

def name_image(img):
    hashedImage = hashlib.md5(img.tobytes()).hexdigest()
    hashedName = hashedImage + "." + "jpg"
    return hashedName
    
allfiles = [f for f in listdir(fromDir) if isfile(join(fromDir, f))]

for i in range(len(allfiles)):
    file = join(fromDir, allfiles[i])
    try:
        img = Image.open(file).convert("RGB")
    except:
        os.remove(file)
        continue
    extension = allfiles[i].split('.')[-1].lower()

    newname = name_image(img)
    if img.size[0] >= img.size[1]: # landscape
        if extension != "jpg":
            img.save(join(lsDir, newname))
        else:
            shutil.copy(join(fromDir, allfiles[i]), join(lsDir, newname))
    else: # portrait or square.
        if extension != "jpg":
            img.save(join(portDir, newname))
        else:
            shutil.copy(join(fromDir, allfiles[i]), join(portDir, newname)) # Copy instead of save to limit quality loss.
    img.close()
    remove(join(fromDir, allfiles[i]))