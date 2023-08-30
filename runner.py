from train import Trainer, get_run_name
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
import os


def get_default_device(device, multi_gpu=False):
    if torch.cuda.is_available():
        if multi_gpu:
            return torch.device('cuda:' + str(list(device)[0]))
        else:
            return torch.device('cuda:' + str(device))
    else:
        return torch.device('cpu')


def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg):
    save_run = cfg.save_run
    verbose = cfg.verbose
    if save_run:
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

    if isinstance(cfg.device, int):
        device = get_default_device(cfg.device, multi_gpu=False)
    else:
        device = get_default_device(cfg.device, multi_gpu=True)

    # PARAMETERS OF THE TRAINING
    data_path = "/waldo/walban/student_datasets/arfranck/SDD/scenes/"
    saving_path = '/linux/grotsartdehe/TrajViViT-models/'
    if save_run:
        if not os.path.exists(saving_path):
            os.umask(0)  # for the file permission
            os.makedirs(saving_path)  # Create a new directory because it does not exist

    n_prev = cfg.n_prev
    n_next = cfg.n_next
    teacher_forcing = cfg.tf
    scene = cfg.scene
    video = "/video" + str(cfg.video)
    box_size = cfg.box_size
    img_size = cfg.size
    pos_bool = cfg.pos
    img_bool = cfg.img
    multi_cam = cfg.multi_cam
    train_prop = 0.9
    val_prop = 0.05
    test_prop = 0.05
    img_step = cfg.img_step
    n_epoch = 100

    # HYPER-PARAMETERS OF THE MODEL
    model_dimension = cfg.model.params.embedding_size
    model_depth = cfg.model.params.layer
    n_heads = cfg.model.params.MHA
    mlp_dim = cfg.model.params.mlp_dim
    dropout = cfg.model.params.dropout

    # GRADIENT DESCENT PARAMETERS
    batch_size = cfg.model.params.batch_size
    lr = cfg.model.params.learning_rate
    optimizer_name = cfg.optimizer
    scheduler_config = "fixed"
    criterion = MSELoss()

    if save_run:
        wandb.run.name = get_run_name(multi_cam=multi_cam, box_size=box_size, pos_bool=pos_bool, img_bool=img_bool,
                                      scene=scene, video_id=str(cfg.video))
    if len(multi_cam) == 0:
        if box_size != 0:
            data_folders = [
                data_path + scene + video + f"/frames_({img_size}, {img_size})_box_{box_size}_step_{img_step}/"]
        else:
            data_folders = [data_path + scene + video + f"/frames_({img_size}, {img_size})_step_{img_step}/"]
    else:
        folders = TrajDataset.conf_to_folders(multi_cam)
        if box_size != 0:
            data_folders = [data_path + scenePath + f"/frames_({img_size}, {img_size})_box_{box_size}_step_{img_step}/"
                            for
                            scenePath in folders]
        else:
            data_folders = [data_path + scenePath + f"/frames_({img_size}, {img_size})_step_{img_step}/"
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
                      dropout=dropout, n_prev=n_prev, pos_bool=pos_bool, img_bool=img_bool, device=device)

    if isinstance(cfg.device, int):  # Training on single GPU
        model = to_device(model, device)
    else:  # Training on multiple GPUs
        model = nn.DataParallel(model, device_ids=cfg.device)
        model = to_device(model, list(cfg.device)[0])

    # MODEL'S WEIGHTS INITIALISATION
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)  # nn.init.kaiming_uniform_(p)

    with torch.cuda.device(device):
        torch.cuda.empty_cache()

    # OPTIMIZER CHOICE
    if optimizer_name == "adam":
        opt = Adam(params=model.parameters(), lr=lr)
    elif optimizer_name == "SGD":
        opt = SGD(params=model.parameters(), lr=lr)
    elif optimizer_name == "adagrad":
        opt = Adagrad(params=model.parameters(), lr=lr)
    else:
        print("Optimizer name not recognized.")
        raise NotImplementedError

    # SCHEDULER CHOICE
    if scheduler_config == 'fixed':
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=opt, lr_lambda=lambda epoch: 1)
    elif scheduler_config == 'multi_step_30_60':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=opt, milestones=[30, 60], gamma=0.1)
    elif scheduler_config == 'multi_step_10_30_60':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=opt, milestones=[10, 30, 60], gamma=0.1)
    elif scheduler_config == 'step_80':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer=opt, step_size=1, gamma=0.80)
    elif scheduler_config == 'step_90':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer=opt, step_size=1, gamma=0.90)
    elif scheduler_config == 'step_95':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer=opt, step_size=1, gamma=0.95)
    elif scheduler_config == 'noam':
        opt = Adam(model.parameters(), lr=0.001, betas=(0.9, 0.98), eps=1e-9)
        scheduler = NoamLR(optimizer=opt, model_dimension=model_dimension,
                           warmup_steps=int(len(train_loader) * n_epoch * 0.05))
    else:
        print("Scheduler configuration not recognized.")
        raise NotImplementedError

    configuration = {
        "model": model,
        "train_data": train_loader,
        "test_data": test_loader,
        "val_data": val_loader,
        "criterion": criterion,
        "optimizer": opt,
        "scheduler": scheduler,
        "epochs": n_epoch,
        "teacher_forcing": teacher_forcing,
        "box_size": box_size,
        "scene": scene,
        "video": str(cfg.video),
        "pos_bool": pos_bool,
        "img_bool": img_bool,
        "multi_cam": multi_cam,
        "save_run": save_run,
        "saving_path": saving_path,
        "verbose": verbose,
        "device": device
    }

    trainer = Trainer(**configuration)
    trainer.train()
    run.finish()


if __name__ == "__main__":
    main()
