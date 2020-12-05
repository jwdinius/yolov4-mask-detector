import random
import os
import subprocess
import sys
import copy

image_dir = "./raw-data"

path, dirs, files = next(os.walk(image_dir))
data_size = len(files)

for f in os.listdir(image_dir):
    # check if filename has single quotes
    # rename jpeg to jpg
    prefix, ext = os.path.splitext(f)
    if ext == ".jpeg":
        os.rename(image_dir + "/" + f, image_dir + "/" + prefix + ".jpg")
