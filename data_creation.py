import os
import pandas as pd
from PIL import Image, ImageDraw
import sys
import numpy as np
import argparse

"""
    data_creation.py is used to generate the dataset used by TrajViVit from the raw data of SDD
"""

parser = argparse.ArgumentParser()
parser.add_argument('--scene', type=str)
parser.add_argument('--video', type=int)
args = parser.parse_args()

dataPath = "/waldo/walban/student_datasets/arfranck/SDD/scenes/"
scene = args.scene
videoID = args.video
videoPath = dataPath + scene + "/video" + str(videoID) + "/"
resize_bool = False #weither we want to resize the data

cols = ["track_id", "xmin", "ymin", "xmax", "ymax", "frame", "lost", "occluded", "generated", "label"]
data = pd.read_csv(videoPath + "annotations.txt", sep=" ", names=cols)

def get_old_size(videoPath):
    image_path = f"{videoPath}/frames/00001.jpg"
    img = Image.open(image_path)
    return img.size


# Parameters of transform
old_size = get_old_size(videoPath)
print("image shape ", old_size)
img_step = 30 #int(sys.argv[2])
box_size = 40#int(sys.argv[4])

if resize_bool:
    new_size = (int(sys.argv[3]), int(sys.argv[3]))
    box_size = int(sys.argv[4])
    old_size = get_old_size(scene)

    output_folder = f"{scene}/{new_size[0]}_{new_size[1]}_{box_size}"
    if os.path.exists(output_folder + "/annotations_" + str(img_step) + ".txt"):
        print("already computed : " + output_folder + "/annotations_" + str(img_step))
        exit(0)

    # Computation of new annotations
    data = data[data.index % img_step == 0]

    x_scale = old_size[0] / new_size[0]
    y_scale = old_size[1] / new_size[1]

    data["x"] = round(((data["xmax"] + data["xmin"]) / 2) / x_scale, 2)
    data["y"] = round(((data["ymax"] + data["ymin"]) / 2) / y_scale, 2)

    data["xmin"] = round(data["x"] - (box_size/2)).astype(int)
    data["xmax"] = round(data["xmin"] + (box_size-1)).astype(int)
    data["ymin"] = round(data["y"] - (box_size/2)).astype(int)
    data["ymax"] = round(data["ymin"] + (box_size-1)).astype(int)
else:
    output_folder = videoPath + "frames_box_" + str(box_size) + "_step_" + str(img_step)

try:
    os.mkdir(output_folder)
except FileExistsError as e:
    pass


data.to_csv(output_folder + "/annotations_" + str(img_step) + ".txt", sep=" ", index=False)

x_size = []
y_size = []
for ind, row in data.iterrows():
    print(ind, row)
    frame = f"{row.frame:05d}"
    image_path = videoPath + f"frames/{frame}.jpg"
    outname = f"{output_folder}/{row.track_id:03d}_{frame}.jpg"

    if not os.path.exists(outname):
        img = Image.open(image_path)
        if resize_bool:
            old_size = img.size
            img = img.convert("L").resize(new_size, Image.Resampling.LANCZOS)
        draw = ImageDraw.Draw(img)

        left = row.xmin
        top = row.ymin
        right = row.xmax
        bottom = row.ymax
        x_size.append(right-left)
        y_size.append((bottom-top))
        draw.rectangle((left, top, right, bottom), fill="black")
        img.save(outname)

print(len(x_size))
print(np.mean(x_size), np.mean(y_size))
