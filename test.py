import torch
import numpy as np
import hydra
import matplotlib.pyplot as plt
import pandas as pd
import os

from traj_dataset import TrajDataset
from utils import get_model_name, get_folders, get_default_device
from torch.utils.data import DataLoader
from trajViViT import TrajViVit
from openpyxl import load_workbook


def mse(error_x, error_y, time_step):
	return error_x[time_step] ** 2 + error_y[time_step] ** 2


def test(model, test_loader, device):
	with torch.no_grad():
		model.eval()
		test_loss_x = []
		test_loss_y = []
		for test_batch in test_loader:

			x_test = test_batch["src"].to(device)
			y_test = test_batch["tgt"].to(device)
			src_coord = test_batch["coords"].to(device)

			future = None
			for k in range(y_test.shape[1]):
				pred, future = model(x_test, future, src=src_coord)

			loss = torch.abs(pred - y_test)
			test_loss_x.append(torch.mean(loss, dim=0)[:, 0].detach().cpu().numpy())
			test_loss_y.append(torch.mean(loss, dim=0)[:, 1].detach().cpu().numpy())

		error_x = np.mean(test_loss_x, axis=0)
		error_y = np.mean(test_loss_y, axis=0)
	return error_x, error_y


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg):
	data_path = "/waldo/walban/student_datasets/arfranck/SDD/scenes/"
	saving_path = "/linux/grotsartdehe/TrajViViT-models/"
	if isinstance(cfg.device, int):
		device = get_default_device(cfg.device, multi_gpu=False)
	else:
		device = get_default_device(cfg.device, multi_gpu=True)

	n_prev = cfg.n_prev
	n_next = cfg.n_next
	scene = cfg.scene
	video_id = str(cfg.video)
	video = "/video" + video_id
	box_size = cfg.box_size
	img_size = cfg.size
	multi_cam = cfg.multi_cam
	img_step = cfg.img_step
	train_prop = 0.9
	val_prop = 0.05
	test_prop = 0.05
	props = [train_prop, val_prop, test_prop]

	# HYPER-PARAMETERS OF THE MODEL
	model_dimension = cfg.model.params.embedding_size
	model_depth = cfg.model.params.layer
	n_heads = cfg.model.params.MHA
	mlp_dim = cfg.model.params.mlp_dim
	dropout = cfg.model.params.dropout

	# GRADIENT DESCENT PARAMETERS
	batch_size = cfg.model.params.batch_size

	data_folders = get_folders(multi_cam=multi_cam, box_size=box_size, img_size=img_size, img_step=img_step,
							   data_path=data_path, scene=scene, video=video)

	test_data = TrajDataset(data_folders, n_prev=n_prev, n_next=n_next, img_step=img_step, prop=props, part=2,
							box_size=box_size)

	test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

	img_bool_list = [True, True, False]
	pos_bool_list = [True, False, True]

	fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
	fig.set_figheight(15)
	fig.set_figwidth(15)
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

		error_x, error_y = test(model=model, test_loader=test_loader, device=device)
		horizon = [i + 1 for i in range(len(error_x))]

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
		df = pd.DataFrame([[scene, video_id, video_id, mode] + list(error_x) + list(error_y) + data_mse],
						  columns=["Source dataset", "Source domain", "Target domain", "Mode"] + header_error_x
						  + header_error_y + header_mse)

		if not os.path.exists(saving_path + "analysis.xlsx"):
			with pd.ExcelWriter(saving_path + "analysis.xlsx", mode='w') as writer:
				df_init = pd.DataFrame([])
				df_init.to_excel(writer)
			with pd.ExcelWriter(saving_path + "analysis.xlsx", mode='w') as writer:
				df.to_excel(writer, index=False)
		else:
			wb = load_workbook(saving_path + "analysis.xlsx")

			sheet = wb.active
			sheet.append([scene, video_id, video_id, mode] + list(error_x) + list(error_y) + data_mse)
			wb.save(saving_path + "analysis.xlsx")

		ax1.plot(horizon, error_x * cfg.size)
		ax1.set(xlabel="Hozizon [step]", ylabel="Absolute error [pixel]", title="Error along X axis")
		ax1.legend(["Img+pos", "Img", "Pos"])

		ax2.plot(horizon, error_y * cfg.size)
		ax2.set(xlabel="Hozizon [step]", ylabel="Absolute error [pixel]", title='Error along Y axis')
		ax2.legend(["Img+pos", "Img", "Pos"])

		ax3.plot(horizon, data_mse)
		ax3.set(xlabel="Hozizon [step]", ylabel="Absolute error [pixel]", title='MSE')
		ax3.legend(["Img+pos", "Img", "Pos"])

	plt.savefig(saving_path + scene + video_id + ".png")
if __name__ == "__main__":
	main()
