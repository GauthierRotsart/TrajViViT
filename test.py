import torch
import numpy as np
import hydra
import matplotlib.pyplot as plt
import pandas as pd
import os
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
	for mse in data_mse:
		count += np.sqrt(mse)
	return count/len(data_mse)

def fde(mse_final):
	return np.sqrt(mse_final)

def mean_data(last_mse, data_mse):
	if len(last_mse) > 0:
		mean_mse = [(a + b) // 2 for a, b in zip(last_mse, data_mse)]
		last_mse = data_mse
	else:
		mean_mse = data_mse
	return last_mse, mean_mse

@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg):
	save_run = cfg.save_run
	verbose = cfg.verbose
	data_path = cfg.data_path
	saving_path = cfg.saving_path + "TrajViViT-SDD/"
	save_fig = False
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
	train_prop = 0.9
	val_prop = 0.05
	test_prop = 0.05
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
	lr = cfg.model.params.learning_rate
	optimizer_name = cfg.optimizer
	scheduler_config = "fixed"
	criterion = MSELoss()
	props = [train_prop, val_prop, test_prop]

	if scene_test is None:
		scene_test = scene
		if cfg.video_test is None:
			video_test_id = str(cfg.video)
			data_folders = get_folders(multi_cam=multi_cam, box_size=box_size, img_size=img_size, img_step=img_step,
									   data_path=data_path, scene=scene, video=video)
		else:
			video_test_id = str(cfg.video_test)
			data_folders = get_folders(multi_cam=multi_cam, box_size=box_size, img_size=img_size, img_step=img_step,
									   data_path=data_path, scene=scene, video=video_test)
	else:
		if cfg.video_test is None:
			video_test_id = str(cfg.video)
			data_folders = get_folders(multi_cam=multi_cam, box_size=box_size, img_size=img_size, img_step=img_step,
									   data_path=data_path, scene=scene_test, video=video)
		else:
			video_test_id = str(cfg.video_test)
			data_folders = get_folders(multi_cam=multi_cam, box_size=box_size, img_size=img_size, img_step=img_step,
									   data_path=data_path, scene=scene_test, video=video_test)

	train_data = TrajDataset(data_folders, n_prev=n_prev, n_next=n_next, img_step=img_step, prop=props, part=0,
							 box_size=box_size, verbose=verbose)
	val_data = TrajDataset(data_folders, n_prev=n_prev, n_next=n_next, img_step=img_step, prop=props, part=1,
						   box_size=box_size, verbose=verbose)
	test_data = TrajDataset(data_folders, n_prev=n_prev, n_next=n_next, img_step=img_step, prop=props, part=2,
							box_size=box_size, verbose=verbose)

	if verbose:
		print("TRAIN", len(train_data))
		print("VAL", len(val_data))
		print("TEST", len(test_data))

	train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
	val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
	test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

	video_list = [cfg.video]
	last_mse_pos = []
	last_mse_img = []
	last_mse_ip = []
	for vid in video_list:
		video_id = str(vid)
		img_bool_list = [True, False, True]
		pos_bool_list = [False, True, True]

		if save_fig:
			fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1)
			fig.set_figheight(20)
			fig.set_figwidth(20)

		for img_bool, pos_bool in zip(img_bool_list, pos_bool_list):
			model = TrajViVit(dim=model_dimension, depth=model_depth, mlp_dim=mlp_dim, heads=n_heads, channels=1,
							  dropout=dropout, n_prev=n_prev, pos_bool=pos_bool, img_bool=img_bool, device=device)
			model_path = scene + "_" + video_id + "_"
			model_path = get_model_name(model_path=model_path, img_bool=img_bool, pos_bool=pos_bool)
			current_model_dict = model.state_dict()
			loaded_state_dict = torch.load(saving_path + model_path, map_location=torch.device('cpu'))
			new_state_dict = {k: v if v.size() == current_model_dict[k].size() else current_model_dict[k] for k, v in
							  zip(current_model_dict.keys(), loaded_state_dict.values())}
			model.load_state_dict(new_state_dict, strict=False)
			model = torch.nn.DataParallel(model).cuda()
			print("load model ", model_path)

			configuration = {
				"model": model,
				"train_data": train_loader,
				"val_data": val_loader,
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
			horizon = [i + 1 for i in range(len(error_x))]

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
					last_mse, mean_mse_ip = mean_data(last_mse_ip, data_mse)
				else:
					mode = "Img"
					last_mse, mean_mse_img = mean_data(last_mse_img, data_mse)
			else:
				if pos_bool is True:
					mode = "Pos"
					last_mse, mean_mse_pos = mean_data(last_mse_pos, data_mse)
				else:
					raise NotImplementedError

			df = pd.DataFrame(
				[[scene, video_id, scene_test, video_test_id, mode, str(mean), str(var)] + list(error_x) + list(error_y) + data_mse + [ade_data, fde_data]],
				columns=["Source dataset", "Source domain", "Target dataset", "Target domain",
						 "Mode", "Mean", "Variance"] + header_error_x + header_error_y + header_mse + ["ADE", "FDE"])
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
					[scene, video_id, scene_test, video_test_id, mode, str(mean), str(var)] + list(error_x) + list(error_y) + data_mse + [ade_data, fde_data])
				wb.save(saving_path + f"analysis_{scene}_{video_id}.xlsx")
			if save_fig:
				ax1.plot(horizon, error_x * cfg.size)
				ax1.set(xlabel="Hozizon [step]", ylabel="Absolute error [pixel]", title="Error along X axis")
				ax1.legend([scene + "_" + video_id + "_" + "Img", scene + "_" + video_id + "_" + "Pos"])

				ax2.plot(horizon, error_y * cfg.size)
				ax2.set(xlabel="Hozizon [step]", ylabel="Absolute error [pixel]", title='Error along Y axis')
				ax2.legend([scene + "_" + video_id + "_" + "Img", scene + "_" + video_id + "_" + "Pos"])

				ax3.plot(horizon, data_mse)
				ax3.set(xlabel="Hozizon [step]", ylabel="Absolute error [pixel]", title='MSE')
				ax3.legend([scene + "_" + video_id + "_" + "Img", scene + "_" + video_id + "_" + "Pos"])

				plt.savefig(saving_path + scene + video_id + ".png")
	if save_fig:
		fig, (ax4) = plt.subplots(1, 1)
		fig.set_figheight(20)
		fig.set_figwidth(20)

		ax4.plot(horizon, mean_mse_ip)
		ax4.plot(horizon, mean_mse_img)
		ax4.plot(horizon, mean_mse_pos)
		ax4.set(xlabel="Hozizon [step]", ylabel="Absolute error [pixel]", title='MSE')
		ax4.legend(["Img", "Pos"])
		plt.savefig(saving_path + scene + ".png")

if __name__ == "__main__":
	main()