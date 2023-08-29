import torch
from noam import NoamLR
import wandb
import numpy as np
from tqdm import tqdm
import os

"""
    Train Loop to train the TrajViVit model
"""
class Trainer:

    def __init__(self, model, device, train_data, test_data, val_data, criterion, optimizer, scheduler, epochs, lr,
                 teacher_forcing=False, box_size=0, scene="bookstore", video=1, pos_bool=False, img_bool=False,
                 app=False, multiCam=[]):

        self.model = model
        self.device = device
        self.train_data = train_data
        self.test_data = test_data
        self.val_data = val_data
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epochs = epochs
        self.teacher_forcing = teacher_forcing
        self.box_size = box_size
        self.scene = scene
        self.video = video
        self.pos_bool = pos_bool
        self.img_bool = img_bool
        self.app = app
        self.multiCam = multiCam

        self.loss_evolution = []
        self.test_evolution = []
        self.validation_evolution = []

        if len(self.multiCam) == 0:
            if self.box_size == 0:
                if pos_bool == True and img_bool == True:
                    if self.app:
                        self.name = scene + "_" + str(video) + "_Img+Pos-app.pt"
                    else:
                        self.name = scene + "_" + str(video) + "_Img+Pos-4.pt"
                elif pos_bool == False and img_bool == True:
                    self.name = scene + "_" + str(video) + "_Img-4.pt"
                elif pos_bool == True and img_bool == False:
                    self.name = scene + "_" + str(video) + "_Pos-4.pt"
                else:
                    raise NotImplementedError
            else:
                if pos_bool == True and img_bool == True:
                    self.name = scene + "_" + str(video) + "_box_" + str(box_size) + "_Img+Pos.pt"
                elif pos_bool == False and img_bool == True:
                    self.name = scene + "_" + str(video) + "_box_" + str(box_size) + "_Img.pt"
                elif pos_bool == True and img_bool == False:
                    self.name = scene + "_" + str(video) + "_box_" + str(box_size) + "_Pos.pt"
                else:
                    raise NotImplementedError
        else:
            scene_name = ""
            for sc in multiCam:
                scene_name += sc

            if self.box_size == 0:
                if pos_bool == True and img_bool == True:
                    if self.app:
                        self.name = scene_name + "_Img+Pos-app.pt"
                    else:
                        self.name = scene_name + "_Img+Pos.pt"
                elif pos_bool == False and img_bool == True:
                    self.name = scene_name + "_Img.pt"
                elif pos_bool == True and img_bool == False:
                    self.name = scene_name + "_Pos.pt"
                else:
                    raise NotImplementedError
            else:
                if pos_bool == True and img_bool == True:
                    self.name = scene_name + "_box_" + str(box_size) + "_Img+Pos.pt"
                elif pos_bool == False and img_bool == True:
                    self.name = scene_name + "_box_" + str(box_size) + "_Img.pt"
                elif pos_bool == True and img_bool == False:
                    self.name = scene_name + "_box_" + str(box_size) + "_Pos.pt"
                else:
                    raise NotImplementedError

    def train(self):

        optimizer = self.optimizer
        criterion = self.criterion
        model = self.model
        scheduler = self.scheduler
        update_step = -1
        self.count = 0
        last_loss = 300
        best_loss = 300
        with tqdm(range(self.epochs), total=self.epochs, unit="epoch", desc=f"Epoch {self.count}") as tepoch:
            for epoch in tepoch:
                loss_evolution = []

                model.train()
                for step, train_batch in enumerate(self.train_data):
                    update_step += 1

                    X_train = train_batch["src"].to(self.device)
                    Y_train = train_batch["tgt"].to(self.device)
                    src_coord = train_batch["coords"].to(self.device)

                    optimizer.zero_grad()

                    if epoch < self.teacher_forcing:  # Teacher Forcing approach

                        pred = model(X_train, Y_train, src_coord)

                    else:  # Autoregressive approach
                        future = None
                        n_next = Y_train.shape[1]
                        for k in range(n_next):
                            print("cc")
                            pred, future = model(video=X_train, tgt=future, src=src_coord, train=False)

                    loss = criterion(pred, Y_train)

                    loss_evolution.append(loss.item())
                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                    tepoch.set_postfix(loss=np.mean(loss.item()))

                current_loss = self.validation(epoch)
                wandb.log({"Training loss": np.mean(loss_evolution), "Epochs": epoch + 1})
                wandb.log({"Validation loss": current_loss, "Epochs": epoch + 1})
                if current_loss >= last_loss:
                    print('Epoch [{}/{}], Training loss: {:.4f}, Validation loss: {:.4f}'.format(epoch + 1, self.epochs,
                                                                                                 np.mean(
                                                                                                     loss_evolution),
                                                                                                 current_loss))
                else:
                    print('Epoch [{}/{}], Training loss: {:.4f}, Validation loss: {:.4f}'.format(epoch + 1, self.epochs,
                                                                                                 np.mean(
                                                                                                     loss_evolution),
                                                                                                 current_loss))
                    best_loss = current_loss
                    print(f"\nBest validation loss: {best_loss}")
                    print(f"Saving best model for epoch: {epoch + 1}\n")
                    saving_path = '/linux/grotsartdehe/TrajViViT-models/'
                    if not os.path.exists(saving_path):
                        os.umask(0)  # for the file permission
                        os.makedirs(saving_path)  # Create a new directory because it does not exist
                        print("New directory created to save the data: ", saving_path)
                    torch.save(model, saving_path + self.name)
                last_loss = best_loss

                tepoch.update(1)
                self.count += 1

    def validation(self, epoch):
        print("Let s go !")
        with torch.no_grad():

            model = self.model
            criterion = self.criterion
            model.eval()

            val_loss = []
            for _, val_batch in enumerate(self.val_data):

                X_val = val_batch["src"].to(self.device)
                Y_val = val_batch["tgt"].to(self.device)
                src_coord = val_batch["coords"].to(self.device)

                future = None
                for k in range(Y_val.shape[1]):
                    pred, future = model(video=X_val, tgt=future, src=src_coord, train=False)

                loss = criterion(pred, Y_val)
                val_loss.append(loss.item())

        return np.mean(val_loss)

    def test(self):

        with torch.no_grad():

            model = self.model
            criterion = self.criterion
            model.eval()

            for test_batch in self.test_data:

                X_test = test_batch["src"].to(self.device)
                Y_test = test_batch["tgt"].to(self.device)

                future = None
                for k in range(Y_test.shape[1]):
                    pred, future = model(X_test, future, train=False)

                loss = criterion(pred, Y_test)

                self.test_evolution.append(loss)
