from os import listdir
from os.path import isfile, join
import shutil
from PIL import Image


#fromDir = "data\\train2017"
fromDir = "data\\new_data"
portDir = "data\\train_portrait"
lsDir   = "data\\train_landscape"
prefix  = "new_"
allfiles = [f for f in listdir(fromDir) if isfile(join(fromDir, f))]

for i in range(len(allfiles)):
    img = Image.open(join(fromDir, allfiles[i]))
    extension = allfiles[i].split('.')[-1]
    if extension == "jpeg":
        extension = "jpg"
    if img.size[0] >= img.size[1]:
        shutil.copy(join(fromDir, allfiles[i]), join(lsDir, f"{prefix}{i}.{extension}"))
    else:
        shutil.copy(join(fromDir, allfiles[i]), join(portDir, f"{prefix}{i}.{extension}"))
