import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import dataset
from torchvision import transforms
import random
import numpy as np
from PIL import Image


class TrajDataset(dataset.Dataset):
    to_tensor = transforms.ToTensor()

    def __init__(self, data_folders, n_prev, n_next, img_step, prop, part, box_size, verbose):

        self.data_folders = data_folders
        self.n_prev = n_prev
        self.n_next = n_next
        self.img_step = img_step
        self.box_size = box_size
        self.verbose = verbose
        if self.box_size == 0:
            self.block_size = "Variable"
        else:
            self.block_size = int(data_folders[0].split("_")[-3])

        self.files = []
        self.pos = []

        for folder in self.data_folders:
            self.folder = folder
            rand = random.Random(42)
            self.raw_data = pd.read_csv(folder + "/annotations_" + str(self.img_step) + ".txt", sep=" ")
            all_ids = self.raw_data[self.raw_data['occluded'] != 1]["track_id"].unique()
            rand.shuffle(all_ids)
            split_index = (len(all_ids) * np.cumsum(prop)).astype(int)
            trajs_index = np.split(all_ids, split_index[:-1])[part]
            track_ids = self.raw_data["track_id"].unique()[trajs_index]

            for track_id in track_ids:
                if self.verbose:
                    print("opening track " + str(track_id) + " from " + folder)
                traj = self.raw_data[self.raw_data["track_id"] == track_id]  # get all positions of track
                for i in range(len(traj) - self.n_next - self.n_prev):
                    track_id = traj.iloc[i, :]["track_id"]
                    frame = traj.iloc[i, :]["frame"]
                    path = f"{folder}/{track_id:03d}_{frame:05d}.jpg"
                    self.files.append(path)
                    self.pos.append(i)

    def normalize_coords(self, tgt):
        return tgt / self.get_image_size()[0]

    def __getitem__(self, item):
        img_file = self.files[item]
        img = img_file.split('/')[-1]
        img_track = img.split('.')[0]
        track_id, frame_id = img_track.split('_')
        track_path = self.folder + f"{int(track_id):03d}_"
        x = []
        for i in range(self.n_prev):
            img_path = track_path + f"{int(frame_id) + 12*i:05d}.jpg"
            img = Image.open(img_path)
            img_tensor = self.to_tensor(img)
            x.append(img_tensor)

        pos = self.pos[item]
        traj = self.raw_data[self.raw_data["track_id"] == int(track_id)]  # get all positions of track
        c = traj.iloc[pos: pos + self.n_prev][["x", "y"]]  # coordinates of the previous images
        y = traj.iloc[pos + self.n_prev: pos + self.n_prev + self.n_next][
            ["x", "y"]]  # images that should be predicted

        self.src = torch.cat(x)
        self.coords = self.normalize_coords(Tensor(c.values))
        self.tgt = self.normalize_coords(Tensor(y.values))
        return {
            'src': self.src,
            'coords': self.coords,
            'tgt': self.tgt
        }

    def __len__(self):
        return len(self.files)

    def get_image_size(self):
        return self.src[0].size()[1:]

    @classmethod
    def conf_to_folders(cls, confname):
        folder_list = []
        for cam in confname:
            scene_letter = [*cam][0]
            video_nb = [*cam][1]

            if scene_letter == "b":
                folder_list.append("bookstore/video" + video_nb)
            elif scene_letter == "c":
                folder_list.append("coupa/video" + video_nb)
            elif scene_letter == "d":
                folder_list.append("deathCircle/video" + video_nb)
            elif scene_letter == "g":
                folder_list.append("gates/video" + video_nb)
            elif scene_letter == "h":
                folder_list.append("hyang/video" + video_nb)
            elif scene_letter == "l":
                folder_list.append("little/video" + video_nb)
            elif scene_letter == "n":
                folder_list.append("nexus/video" + video_nb)
            elif scene_letter == "q":
                folder_list.append("quad/video" + video_nb)
            else:
                raise NotImplementedError
        return folder_list
