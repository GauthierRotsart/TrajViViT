import torch
import wandb
import numpy as np
from tqdm import tqdm


def get_run_name(multi_cam, box_size, pos_bool, img_bool, scene, video_id):
    if len(multi_cam) == 0:  # Training on a single camera
        if box_size == 0:  # Training on the box given by an object detection algorithm
            if pos_bool is True and img_bool is True:
                return scene + "_" + str(video_id) + "_Img+Pos"
            elif pos_bool is False and img_bool is True:
                return scene + "_" + str(video_id) + "_Img"
            elif pos_bool is True and img_bool is False:
                return scene + "_" + str(video_id) + "_Pos"
            else:
                print("The input is at least the positions or the images.")
                raise NotImplementedError
        else:
            if pos_bool is True and img_bool is True:
                return scene + "_" + str(video_id) + "_box_" + str(box_size) + "_Img+Pos"
            elif pos_bool is False and img_bool is True:
                return scene + "_" + str(video_id) + "_box_" + str(box_size) + "_Img"
            elif pos_bool is True and img_bool is False:
                return scene + "_" + str(video_id) + "_box_" + str(box_size) + "_Pos"
            else:
                print("The input is at least the positions or the images.")
                raise NotImplementedError
    else:  # Training on multiple cameras
        scene_name = ""
        for sc in multi_cam:
            scene_name += sc

        if box_size == 0:
            if pos_bool is True and img_bool is True:
                return scene_name + "_Img+Pos"
            elif pos_bool is False and img_bool is True:
                return scene_name + "_Img"
            elif pos_bool is True and img_bool is False:
                return scene_name + "_Pos"
            else:
                print("The input is at least the positions or the images.")
                raise NotImplementedError
        else:
            if pos_bool is True and img_bool is True:
                return scene_name + "_box_" + str(box_size) + "_Img+Pos"
            elif pos_bool is False and img_bool is True:
                return scene_name + "_box_" + str(box_size) + "_Img"
            elif pos_bool is True and img_bool is False:
                return scene_name + "_box_" + str(box_size) + "_Pos"
            else:
                print("The input is at least the positions or the images.")
                raise NotImplementedError

class Trainer:

    def __init__(self, model, train_data, test_data, val_data, criterion, optimizer, scheduler, epochs, teacher_forcing,
                 box_size, scene, video, pos_bool, img_bool, multi_cam, save_run, saving_path, verbose, device):

        self.model = model
        self.train_data = train_data
        self.test_data = test_data
        self.val_data = val_data

        self.teacher_forcing = teacher_forcing
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epochs = epochs

        self.box_size = box_size
        self.scene = scene
        self.video = video
        self.pos_bool = pos_bool
        self.img_bool = img_bool
        self.multi_cam = multi_cam
        self.save_run = save_run
        self.saving_path = saving_path
        self.verbose = verbose
        self.name = get_run_name(multi_cam=self.multi_cam, box_size=self.box_size, pos_bool=self.pos_bool,
                                 img_bool=self.img_bool, scene=self.scene, video_id=self.video) + '.pt'
        self.device = device

    def train(self):
        last_loss = 300
        best_loss = 300

        with tqdm(total=self.epochs, unit="epoch", desc=f"Training") as t_epoch:
            for epoch in range(self.epochs):
                train_loss = []
                self.model.train()
                for step, train_batch in enumerate(self.train_data):

                    x_train = train_batch["src"].to(self.device)
                    y_train = train_batch["tgt"].to(self.device)
                    src_coord = train_batch["coords"].to(self.device)

                    self.optimizer.zero_grad()

                    if epoch < self.teacher_forcing:  # Teacher forcing approach
                        pred, _ = self.model(x_train, y_train, src_coord)
                    else:  # Autoregressive approach
                        future = None
                        n_next = y_train.shape[1]
                        for k in range(n_next):
                            pred, future = self.model(video=x_train, tgt=future, src=src_coord)

                    loss = self.criterion(pred, y_train)

                    train_loss.append(loss.item())
                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()

                    t_epoch.set_postfix(loss=np.mean(loss.item()))

                current_loss = self.validation()
                if self.verbose:
                    print('Epoch [{}/{}], Training loss: {:.4f}, Validation loss: {:.4f}'.format(epoch + 1, self.epochs,
                                                                                                 np.mean(train_loss),
                                                                                                 current_loss))
                if current_loss < last_loss:
                    best_loss = current_loss
                    if self.save_run:
                        wandb.log({"Training loss": np.mean(train_loss), "Epochs": epoch + 1})
                        wandb.log({"Validation loss": current_loss, "Epochs": epoch + 1})
                        if self.verbose:
                            print(f"Saving best model for epoch: {epoch + 1}\n")

                        torch.save(self.model, self.saving_path + self.name)
                last_loss = best_loss
                t_epoch.update(1)

    def validation(self):
        with torch.no_grad():
            self.model.eval()

            val_loss = []
            for _, val_batch in enumerate(self.val_data):

                x_val = val_batch["src"].to(self.device)
                y_val = val_batch["tgt"].to(self.device)
                src_coord = val_batch["coords"].to(self.device)

                future = None
                for k in range(y_val.shape[1]):
                    pred, future = self.model(video=x_val, tgt=future, src=src_coord)

                loss = self.criterion(pred, y_val)
                val_loss.append(loss.item())
        return np.mean(val_loss)

    def test(self):
        with torch.no_grad():
            self.model.eval()
            test_loss = []
            for test_batch in self.test_data:

                x_test = test_batch["src"].to(self.device)
                y_test = test_batch["tgt"].to(self.device)
                src_coord = test_batch["coords"].to(self.device)

                future = None
                for k in range(y_test.shape[1]):
                    pred, future = self.model(x_test, future, src=src_coord)

                loss = self.criterion(pred, y_test)
                test_loss.append(loss.item())
        return np.mean(test_loss)
