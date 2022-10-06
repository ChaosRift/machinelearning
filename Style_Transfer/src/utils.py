import scipy.misc, numpy as np, os, sys
import imageio
from PIL import Image
from os.path import exists

def save_img(out_path, img):
    img = np.clip(img, 0, 255).astype(np.uint8)
    imageio.imwrite(out_path, img)

def scale_img(style_path, style_scale):
    scale = float(style_scale)
    o0, o1, o2 = imageio.imread(style_path, pilmode='RGB').shape
    scale = float(style_scale)
    new_shape = (int(o0 * scale), int(o1 * scale), o2)
    style_target = get_img(style_path, img_size=new_shape)
    return style_target

def get_img(src, img_size=False):
    
    #print("Image source " + src)
    try:
        img = imageio.imread(src, pilmode='RGB') # misc.imresize(, (256, 256, 3))
    except:
        print(f"Can't open {src} using substitute.")
        img = imageio.imread("src\\srcfix.jpg", pilmode='RGB')
       
    if not (len(img.shape) == 3 and img.shape[2] == 3):
        img = np.dstack((img,img,img))
    if img_size != False:
        img = np.array(Image.fromarray(img).resize(img_size[:2]))
    return img

def exists(p, msg):
    assert os.path.exists(p), msg

def get_file_list(in_path):
    for r, d, f in os.walk(path):
        for file in f:
            if '.jpg' in file.lower() or '.png' in file.lower():
                files.append(os.path.join(r, file))
    return files

import tensorflow as tf
import numpy as np
import PIL.Image
import cv2
import os


def tensor_to_image(tensor):
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor)>3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)
    

def load_img(path_to_img, max_dim=None, resize=True):
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    
    if resize:
        new_shape = tf.cast([256, 256], tf.int32)
        img = tf.image.resize(img, new_shape)

    if max_dim:
        shape = tf.cast(tf.shape(img)[:-1], tf.float32)
        long_dim = max(shape)
        scale = max_dim / long_dim
        new_shape = tf.cast(shape * scale, tf.int32)
        img = tf.image.resize(img, new_shape)
        
    img = img[tf.newaxis, :]

    return img

def create_folder(diirname):
    if not exists(diirname):
        os.mkdir(diirname)
        print('Directory ', diirname, ' createrd')
    else:
        print('Directory ', diirname, ' already exists')       


def clip_0_1(image):
    """
    clips images from 0-254 to range between 0 and 1 for tensorflow. 
    """
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=255.0)