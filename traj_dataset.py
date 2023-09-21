import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import dataset
from torchvision import transforms
import random
import numpy as np
from PIL import Image
import glob

class TrajDataset(dataset.Dataset):
    to_tensor = transforms.ToTensor()

    def __init__(self, data_folders, n_prev, n_next, img_step, prop, part=0, limit=None, path=False, box_size=0):

        self.data_folders = data_folders
        self.n_prev = n_prev
        self.n_next = n_next
        self.img_step = img_step
        self.box_size = box_size
        if self.box_size == 0:
            self.block_size = "Variable"
        else:
            self.block_size = int(data_folders[0].split("_")[-3])

        self.path = path

        src = []
        tgt = []
        coords = []
        path = [] if path else False
        self.files = []
        self.track_pos = []
        self.pos = []
        for folder in self.data_folders:
            rand = random.Random(42)
            raw_data = pd.read_csv(folder + "/annotations_" + str(self.img_step) + ".txt", sep=" ")
            all_ids = raw_data[raw_data['occluded'] != 1]["track_id"].unique()
            rand.shuffle(all_ids)
            split_index = (len(all_ids) * np.cumsum(prop)).astype(int)
            trajs_index = np.split(all_ids, split_index[:-1])[part]
            track_ids = raw_data["track_id"].unique()[trajs_index][:limit]

            self.raw_data = raw_data
            for track_id in track_ids:
                #print("opening track " + str(track_id) + " from " + folder)
                traj = raw_data[raw_data["track_id"] == track_id]  # get all positions of track
                self.track_pos.append(len(traj))
                memo = {}
                for i in range(len(traj) - self.n_next - self.n_prev):
                    if path != False:
                        path.append(folder)
                    x = self.get_n_images_after_i(folder, traj, self.n_prev, i, memo, fill=True)  # n_prev images used to predict
                    #src.append(x)
                    c = traj.iloc[i: i + self.n_prev][["x", "y"]]  # coordinates of the previous images
                    #coords.append(Tensor(c.values))
                    y = traj.iloc[i + self.n_prev: i + self.n_prev + self.n_next][
                        ["x", "y"]]  # images that should be predicted
                    #tgt.append(Tensor(y.values))
        """
        self.src = torch.stack(src, dim=0)
        self.coords = self.normalize_coords(torch.stack(coords, dim=0))
        self.tgt = self.normalize_coords(torch.stack(tgt, dim=0))
        self.path = path
        """
    def normalize_coords(self, tgt):
        return tgt / self.get_image_size()[0]

    def get_n_images_after_i(self, folder, traj, n, i, memo, fill=False):
        self.pos.append(i)
        count = 0
        X = []

        for ind, pos in traj.iloc[i: i + n, :].iterrows():
            track_id = pos["track_id"]
            frame = pos["frame"]
            path = f"{folder}/{track_id:03d}_{frame:05d}.jpg"
            if path in memo:
                img = memo[path]
            else:
                img = Image.open(f"{folder}/{track_id:03d}_{frame:05d}.jpg")
                memo[path] = img
            img_tensor = self.to_tensor(img)
            X.append(img_tensor)
            if count == 0 and fill is True:
                self.files.append(path)
            count += 1

        return torch.cat(X)

    def __getitem__(self, item):
        self.folder = '/waldo/walban/student_datasets/arfranck/SDD/scenes/nexus/video0/frames_(224, 224)_step_12/'
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
        self.src = torch.cat(x)
        #print(img_path)

        count = 0
        for i in range(len(self.track_pos)):
            count += self.track_pos[i] -20
            if count >= item:
                pos = count - item
                break
        #print(track_id)
        #print(count,item,pos)
        pos = self.pos[item]
        traj = self.raw_data[self.raw_data["track_id"] == int(track_id)]  # get all positions of track
        c = traj.iloc[pos: pos + self.n_prev][["x", "y"]]  # coordinates of the previous images

        # coords.append(Tensor(c.values))
        y = traj.iloc[pos + self.n_prev: pos + self.n_prev + self.n_next][
            ["x", "y"]]  # images that should be predicted
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

    def get_dataset_infos(self):
        return {"image_size": self.get_image_size(),
                "n_prev": self.n_prev,
                "n_next": self.n_next,
                "block_size": self.block_size
                }

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

