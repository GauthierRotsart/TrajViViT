import torch
import numpy as np
import hydra
import pandas as pd
import os
import random

from train import Trainer
from torch.nn import MSELoss
from traj_dataset import TrajDataset
from utils import get_model_name, get_folders, get_default_device
from torch.utils.data import DataLoader
from trajViViT import TrajViVit
from openpyxl import load_workbook


def mse(error_x, error_y, time_step):
	return error_x[time_step] ** 2 + error_y[time_step] ** 2


def ade(data_mse):
	count = 0
	for mse_i in data_mse:
		count += np.sqrt(mse_i)
	return count/len(data_mse)


def fde(mse_final):
	return np.sqrt(mse_final)


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg):
	save_run = cfg.save_run
	verbose = cfg.verbose
	data_path = cfg.data_path
	saving_path = cfg.saving_path + "TrajViViT-SDD/"
	if isinstance(cfg.device, int):
		device = get_default_device(cfg.device, multi_gpu=False)
	else:
		device = get_default_device(cfg.device, multi_gpu=True)

	n_prev = cfg.n_prev
	n_next = cfg.n_next
	teacher_forcing = cfg.tf
	scene = cfg.scene
	scene_test = cfg.scene_test
	video = "/video" + str(cfg.video)
	video_test = "/video" + str(cfg.video_test)
	box_size = cfg.box_size
	img_size = cfg.size
	pos_bool = cfg.pos
	img_bool = cfg.img
	multi_cam = cfg.multi_cam
	img_step = cfg.img_step
	n_epoch = 100
	mean = cfg.mean
	var = cfg.var

	# HYPER-PARAMETERS OF THE MODEL
	model_dimension = cfg.model.params.embedding_size
	model_depth = cfg.model.params.layer
	n_heads = cfg.model.params.MHA
	mlp_dim = cfg.model.params.mlp_dim
	dropout = cfg.model.params.dropout

	# GRADIENT DESCENT PARAMETERS
	batch_size = cfg.model.params.batch_size
	criterion = MSELoss()

	seed = cfg.seed
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)

	if cfg.test_prop == 0.05:
		train_prop = 0.9
		val_prop = 0.05
		test_prop = 0.05
	else:
		train_prop = 0
		val_prop = 0
		test_prop = 1

	if scene_test is None:  # inference on the source domain
		if len(multi_cam) > 0:
			scene = ""
			for sc in multi_cam:
				scene += sc
		scene_test = scene
		if cfg.video_test is None:
			video_test_id = str(cfg.video)
			if len(multi_cam) > 0:
				video_test_id = "all"
			data_folders = get_folders(multi_cam=multi_cam, box_size=box_size, img_size=img_size, img_step=img_step,
									   data_path=data_path, scene=scene, video=video)
		else:
			video_test_id = str(cfg.video_test)
			data_folders = get_folders(multi_cam=multi_cam, box_size=box_size, img_size=img_size, img_step=img_step,
									   data_path=data_path, scene=scene, video=video_test)
	else:  # inference on the target domain
		if cfg.video_test is None:
			video_test_id = str(cfg.video)
			data_folders = get_folders(multi_cam=[], box_size=box_size, img_size=img_size, img_step=img_step,
									   data_path=data_path, scene=scene_test, video=video, source=False)
		else:
			video_test_id = str(cfg.video_test)
			data_folders = get_folders(multi_cam=[], box_size=box_size, img_size=img_size, img_step=img_step,
									   data_path=data_path, scene=scene_test, video=video_test, source=False)
		scene_name = ""
		for sc in scene_test:
			scene_name += sc
		scene_test = scene_name

	assert pos_bool is True or img_bool is True, "At least pos_bool or img_bool should be True."
	assert (train_prop == 0.9 and val_prop == 0.05 and test_prop == 0.05) or (
				train_prop == 0 and val_prop == 0 and test_prop == 1), "Not appropriate train/val/test split."
	# if the inference is on the source domain, then the test set contains the 5% remaining
	# if the inference is on the target domain, then the test set contains all the trajectories
	props = [train_prop, val_prop, test_prop]
	test_data = TrajDataset(data_folders, n_prev=n_prev, n_next=n_next, img_step=img_step, prop=props, part=2,
							box_size=box_size, verbose=verbose, mean=mean, var=var)

	test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

	video_id = str(cfg.video)

	model = TrajViVit(dim=model_dimension, depth=model_depth, mlp_dim=mlp_dim, heads=n_heads, channels=1,
					  dropout=dropout, n_prev=n_prev, pos_bool=pos_bool, img_bool=img_bool, device=device)
	model_path = scene + "_" + video_id
	model_path = get_model_name(model_path=model_path, img_bool=img_bool, pos_bool=pos_bool, tf=teacher_forcing,
								multi_cam=multi_cam)
	if len(multi_cam) > 0:
		scene = ""
		for sc in multi_cam:
			scene += sc
		video_id = "all"
	current_model_dict = model.state_dict()
	loaded_state_dict = torch.load(saving_path + model_path, map_location=torch.device('cpu'))
	new_state_dict = {k: v if v.size() == current_model_dict[k].size() else current_model_dict[k] for k, v in
					  zip(current_model_dict.keys(), loaded_state_dict.values())}
	model.load_state_dict(new_state_dict, strict=False)
	model = torch.nn.DataParallel(model).cuda()
	print("load model ", model_path)

	configuration = {
		"model": model,
		"train_data": None,
		"val_data": None,
		"test_data": test_loader,
		"criterion": criterion,
		"optimizer": None,
		"scheduler": None,
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
		"device": device,
		"mean": mean,
		"var": var
			}

	trainer = Trainer(**configuration)
	error_x, error_y = trainer.test()

	header_error_x = ["Error_X_t1", "Error_X_t2", "Error_X_t3", "Error_X_t4", "Error_X_t5", "Error_X_t6",
					  "Error_X_t7", "Error_X_t8", "Error_X_t9", "Error_X_t10", "Error_X_t11", "Error_X t_12"]
	header_error_y = ["Error_Y_t1", "Error_Y_t2", "Error_Y_t3", "Error_Y_t4", "Error_Y_t5", "Error_Y_t6",
					  "Error_Y_t7", "Error_Y_t8", "Error_Y_t9", "Error_Y_t10", "Error_Y_t11", "Error_Y_t12"]
	header_mse = ["MSE_t1", "MSE_t2", "MSE_t3", "MSE_t4", "MSE_t5", "MSE_t6", "MSE_t7", "MSE_t8", "MSE_t9",
				  "MSE_t10", "MSE_t11", "MSE_t12"]
	data_mse = [mse(error_x, error_y, 0), mse(error_x, error_y, 1), mse(error_x, error_y, 2),
				mse(error_x, error_y, 3), mse(error_x, error_y, 4), mse(error_x, error_y, 5),
				mse(error_x, error_y, 6), mse(error_x, error_y, 7), mse(error_x, error_y, 8),
				mse(error_x, error_y, 9), mse(error_x, error_y, 10), mse(error_x, error_y, 11)]
	ade_data = ade(data_mse=data_mse)
	fde_data = fde(mse_final=data_mse[-1])

	if img_bool is True:
		if pos_bool is True:
			mode = "Img-Pos"
		else:
			mode = "Img"
	else:
		if pos_bool is True:
			mode = "Pos"
		else:
			raise NotImplementedError

	df = pd.DataFrame(
		[[scene, video_id, scene_test, video_test_id, mode, str(mean), str(var)] + list(error_x) + list(error_y)
		 + data_mse + [ade_data, fde_data]],
		columns=["Source dataset", "Source domain", "Target dataset", "Target domain",
				 "Mode", "Mean", "Variance"] + header_error_x + header_error_y + header_mse + ["ADE", "FDE"])
	if cfg.scene_test is None:
		saving_path += "source/"
	else:
		saving_path += "target/"
	if not os.path.exists(saving_path + f"analysis_{scene}_{video_id}.xlsx"):
		with pd.ExcelWriter(saving_path + f"analysis_{scene}_{video_id}.xlsx", mode='w') as writer:
			df_init = pd.DataFrame([])
			df_init.to_excel(writer)
		with pd.ExcelWriter(saving_path + f"analysis_{scene}_{video_id}.xlsx", mode='w') as writer:
			df.to_excel(writer, index=False)
	else:
		wb = load_workbook(saving_path + f"analysis_{scene}_{video_id}.xlsx")

		sheet = wb.active
		sheet.append(
			[scene, video_id, scene_test, video_test_id, mode, str(mean), str(var)] + list(error_x)
			+ list(error_y) + data_mse + [ade_data, fde_data])
		wb.save(saving_path + f"analysis_{scene}_{video_id}.xlsx")

if __name__ == "__main__":
	main()