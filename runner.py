from train import Trainer
from trajViViT import TrajViVit
from traj_dataset import TrajDataset
from torch.utils.data import DataLoader
from torch.optim import *
from torch.nn import MSELoss
from noam import NoamLR
import torch
import torch.nn as nn
import wandb
import hydra
import random
import numpy as np



def get_default_device(device, multiGPU=False):
    if torch.cuda.is_available():
        if multiGPU:
            return torch.device('cuda:' + str(list(device)[0]))
        else:
            return torch.device('cuda:' + str(device))
    else:
        return torch.device('cpu')

def to_device(data, device):
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg):
    wandb.login()
    run = wandb.init(project="TrajViViT")

    wandb.config.update(cfg.model.params)
    print(cfg.model)
    torch.set_default_dtype(torch.float32)

    seed = cfg.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    batch_size = cfg.model.params.batch_size
    lr = cfg.model.params.learning_rate
    if isinstance(cfg.device, int):
        device = get_default_device(cfg.device, multiGPU=False)
    else:
        device = get_default_device(cfg.device, multiGPU=True)
    n_prev = cfg.n_prev
    n_next = cfg.n_next
    model_dimension = cfg.model.params.embedding_size
    model_depth = cfg.model.params.layer
    n_heads = cfg.model.params.MHA
    mlp_dim = cfg.model.params.mlp_dim
    dropout = cfg.model.params.dropout
    teacher_forcing = cfg.tf
    scene = cfg.scene
    video = "/video" + str(cfg.video)
    box_size = cfg.box_size
    img_size = cfg.size
    pos_bool = cfg.pos
    img_bool = cfg.img
    multiCam = cfg.multiCam
    app = cfg.app

    optimizer_name = 'adam'
    train_prop = 0.9
    val_prop = 0.05
    test_prop = 0.05
    img_step = 30
    patch_size = 16
    n_epoch = 100
    scheduler_config = "fixed"

    if len(multiCam) == 0:
        if box_size == 0:
            if pos_bool == True and img_bool == True:
                if app:
                    wandb.run.name = scene + "_" + str(cfg.video) + "_Img+Pos-app" + str(n_next)
                else:
                    wandb.run.name = scene + "_" + str(cfg.video) + "_Img+Pos-" + str(n_next)
            elif pos_bool == False and img_bool == True:
                wandb.run.name = scene + "_" + str(cfg.video) + "_Img" + str(n_next)
            elif pos_bool == True and img_bool == False:
                wandb.run.name = scene + "_" + str(cfg.video) + "_Pos" + str(n_next)
            else:
                raise NotImplementedError
        else:
            if pos_bool == True and img_bool == True:
                wandb.run.name = scene + "_" + str(cfg.video) + "_box_" + str(box_size) + "_Img+Pos"
            elif pos_bool == False and img_bool == True:
                wandb.run.name = scene + "_" + str(cfg.video) + "_box_" + str(box_size) + "_Img"
            elif pos_bool == True and img_bool == False:
                wandb.run.name = scene + "_" + str(cfg.video) + "_box_" + str(box_size) + "_Pos"
            else:
                raise NotImplementedError
    else:
        scene_name = ""
        for sc in multiCam:
            scene_name += sc

        if box_size == 0:
            if pos_bool == True and img_bool == True:
                if app:
                    wandb.run.name = scene_name + "_Img+Pos-app"
                else:
                    wandb.run.name = scene_name + "_Img+Pos"
            elif pos_bool == False and img_bool == True:
                wandb.run.name = scene_name + "_Img"
            elif pos_bool == True and img_bool == False:
                wandb.run.name = scene_name + "_Pos"
            else:
                raise NotImplementedError
        else:
            if pos_bool == True and img_bool == True:
                wandb.run.name = scene_name + "_box_" + str(box_size) + "_Img+Pos"
            elif pos_bool == False and img_bool == True:
                wandb.run.name = scene_name + "_box_" + str(box_size) + "_Img"
            elif pos_bool == True and img_bool == False:
                wandb.run.name = scene_name + "_box_" + str(box_size) + "_Pos"
            else:
                raise NotImplementedError

    dataPath = "/waldo/walban/student_datasets/arfranck/SDD/scenes/"
    if len(multiCam) == 0:
        if box_size != 0:
            data_folders = [
                dataPath + scene + video + f"/frames_({img_size}, {img_size})_box_{box_size}_step_{img_step}/"]
        else:
            data_folders = [dataPath + scene + video + f"/frames_({img_size}, {img_size})_step_{img_step}/"]
    else:
        folders = TrajDataset.conf_to_folders(multiCam)
        if box_size != 0:
            data_folders = [dataPath + scenePath + f"/frames_({img_size}, {img_size})_box_{box_size}_step_{img_step}/"
                            for
                            scenePath in folders]
        else:
            data_folders = [dataPath + scenePath + f"/frames_({img_size}, {img_size})_step_{img_step}/"
                            for
                            scenePath in folders]


    props = [train_prop, val_prop, test_prop]
    train_data = TrajDataset(data_folders, n_prev=n_prev, n_next=n_next, img_step=img_step, prop=props, part=0,
                             box_size=box_size)
    val_data = TrajDataset(data_folders, n_prev=n_prev, n_next=n_next, img_step=img_step, prop=props, part=1,
                           box_size=box_size)
    test_data = TrajDataset(data_folders, n_prev=n_prev, n_next=n_next, img_step=img_step, prop=props, part=2,
                            box_size=box_size)
    print("TRAIN", len(train_data))
    print("VAL", len(val_data))
    print("TEST", len(test_data))

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    model = TrajViVit(dim=model_dimension, depth=model_depth, mlp_dim=mlp_dim, heads=n_heads, channels=1,
                      patch_size=patch_size, nprev=n_prev, device=device, pos_bool=pos_bool, img_bool=img_bool,
                      dropout=dropout, app=app)

    if isinstance(cfg.device, int):
        model = to_device(model, device)
    else:
        model = nn.DataParallel(model, device_ids=cfg.device)
        model = to_device(model, list(cfg.device)[0])

    # Initialize parameters with Glorot / fan_avg.
    # Les poids qui ont une dimension plus petite que 2 utilisent l initialisation par dÃ©faut de pytorch
    # Cette initialisation depend de la couche du rÃ©seau
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)  # nn.init.kaiming_uniform_(p)


    with torch.cuda.device(device):
        torch.cuda.empty_cache()

    if optimizer_name == "adam":
        optimizer = Adam(model.parameters(), lr=lr)
    elif optimizer_name == "SGD":
        optimizer = SGD(model.parameters(), lr=lr)
    elif optimizer_name == "ADAGRAD":
        optimizer = Adagrad(model.parameters(), lr=lr)
    else:
        raise NotImplementedError


    if scheduler_config == 'fixed':
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1)  # lr doesn't change over time
    elif scheduler_config == 'multistep_30_60':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [30, 60], gamma=0.1)
    elif scheduler_config == 'multistep_10_30_60':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [10, 30, 60], gamma=0.1)
    elif scheduler_config == 'step_80':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.80)
    elif scheduler_config == 'step_90':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.90)
    elif scheduler_config == 'step_95':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)
    elif scheduler_config == 'noam':
        optimizer = Adam(model.parameters(), lr=0.001, betas=(0.9, 0.98), eps=1e-9) 
        scheduler = NoamLR(optimizer, model_dimension, int(len(train_loader) * n_epoch * 0.05))
    else:
        raise Exception(f"Scheduler configuration '{scheduler_config}' not recognized")

    mse = MSELoss()
    criterion = MSELoss()

    configuration = {

        "model": model,
        "device": device,
        "train_data": train_loader,
        "test_data": test_loader,
        "val_data": val_loader,
        "criterion": criterion,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "epochs": n_epoch,
        "lr": lr,
        "teacher_forcing": teacher_forcing,
        "box_size": box_size,
        "scene": scene,
        "video": str(cfg.video),
        "pos_bool": pos_bool,
        "img_bool": img_bool,
        "app": app,
        "multiCam": multiCam
    }

    trainer = Trainer(**configuration)
    trainer.train()
    run.finish()
if __name__ == "__main__":
    main()