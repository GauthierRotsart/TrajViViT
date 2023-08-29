import os
import pandas as pd
from PIL import Image, ImageDraw
import numpy as np
import hydra
from openpyxl import load_workbook


def get_old_size(video_path):
    image_path = f"{video_path}/frames/00001.jpg"
    img = Image.open(image_path)
    return img.size


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg):
    saving_path = "/linux/grotsartdehe/SDD.xlsx"  # analyse dataset
    data_path = "/waldo/walban/student_datasets/arfranck/SDD/scenes/"

    scene = cfg.scene
    video_id = cfg.video
    new_size = cfg.size
    box_size = cfg.box_size
    img_step = cfg.img_step
    video_path = data_path + scene + "/video" + str(video_id) + "/"

    resize_bool = True  # weither we want to resize the data
    analyse_bool = False  # weither we want to analyse the dataset
    draw_bool = True  # weither we want to draw black boxes on the frames
    print("scene: ", scene)
    print("video: ", video_id)

    cols = ["track_id", "xmin", "ymin", "xmax", "ymax", "frame", "lost", "occluded", "generated", "label"]
    data = pd.read_csv(video_path + "annotations.txt", sep=" ", names=cols)

    # Parameters of transform
    old_size = get_old_size(video_path)
    print("image shape ", old_size)

    nbr_data = len(data)
    print("number of data: ", nbr_data)
    data = data[data.index % img_step == 0]
    if resize_bool:
        new_size = (new_size, new_size)
        if box_size != 0:
            output_folder = video_path + "frames_" + str(new_size) + "_box_" + str(box_size) + "_step_" + str(img_step)
        else:
            output_folder = video_path + "frames_" + str(new_size) + "_step_" + str(img_step)

        if os.path.exists(output_folder + "/annotations_" + str(img_step) + ".txt"):
            print("already computed : " + output_folder + "/annotations_" + str(img_step))
            exit(0)

        # Computation of new annotations

        x_scale = old_size[0] / new_size[0]
        y_scale = old_size[1] / new_size[1]

        data["x"] = round(((data["xmax"] + data["xmin"]) / 2) / x_scale, 2)
        data["y"] = round(((data["ymax"] + data["ymin"]) / 2) / y_scale, 2)

        if box_size != 0:
            data["xmin"] = round(data["x"] - (box_size/2), 0)
            data["xmax"] = round(data["xmin"] + (box_size-1), 0)
            data["ymin"] = round(data["y"] - (box_size/2), 0)
            data["ymax"] = round(data["ymin"] + (box_size-1), 0)
        else:
            data["xmin"] = round(data["xmin"] / x_scale, 0)
            data["xmax"] = round(data["xmax"] / x_scale, 0)
            data["ymin"] = round(data["ymin"] / y_scale, 0)
            data["ymax"] = round(data["ymax"] / y_scale, 0)
    else:
        if box_size != 0:
            output_folder = video_path + "frames_box_" + str(box_size) + "_step_" + str(img_step)
        else:
            output_folder = video_path + "frames_step_" + str(img_step)

    try:
        os.mkdir(output_folder)
    except FileExistsError:
        pass

    data.to_csv(output_folder + "/annotations_" + str(img_step) + ".txt", sep=" ", index=False)
    nbr_tracks = data.values[-1][0] + 1
    print("number of tracks: ", nbr_tracks)
    print(" ")

    if draw_bool:
        x_size = []
        y_size = []
        for ind, row in data.iterrows():
            print(ind, row)
            frame = f"{row.frame:05d}"
            image_path = video_path + f"frames/{frame}.jpg"
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
                print(right, left)
                print("t", right-left, bottom-top)
                x_size.append(right-left)
                y_size.append((bottom-top))
                draw.rectangle((left, top, right, bottom), fill="black")
                img.save(outname)

        box_x = np.mean(x_size)
        box_y = np.mean(y_size)
    else:
        box_x = 0
        box_y = 0
    if analyse_bool:
        df = pd.DataFrame([[scene, video_id, nbr_tracks, nbr_data, old_size[0], old_size[1], box_x, box_y]],
                          columns=["Scene", "Video", "Tracks", "Data", "Image width", "Image height", "box width",
                                   "box height"])

        if not os.path.exists(saving_path):
            with pd.ExcelWriter(saving_path, mode='w') as writer:
                df_init = pd.DataFrame([])
                df_init.to_excel(writer)
            with pd.ExcelWriter(saving_path, mode='w') as writer:
                df.to_excel(writer, index=False)
        else:
            wb = load_workbook(saving_path)

            sheet = wb.active
            sheet.append([scene, video_id, nbr_tracks, nbr_data, old_size[0], old_size[1], box_x, box_y])
            wb.save(saving_path)


if __name__ == "__main__":
    main()

