import torch
from traj_dataset import TrajDataset


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


# Run name on Wandb platform
def get_run_name(multi_cam, box_size, pos_bool, img_bool, scene, video_id, tf):
	assert box_size >= 0, "The box size has to be positive."
	assert tf >= 0, "The teacher forcing argument has to be positive."
	assert pos_bool is True or img_bool is True, "At least pos_bool or img_bool should be True."

	if len(multi_cam) == 0:  # Training on a single camera
		run_name = f"{scene}_{video_id}"
	else:  # Training on multiple cameras
		run_name = ""
		for sc in multi_cam:
			run_name += sc

	if tf != 100:  # Training with a mix of teacher forcing and autoregressive approaches
		run_name += f"_tf_{tf}"

	if box_size > 0:  # Training with a specific bbox size
		run_name += f"_box_{box_size}"

	if pos_bool is True and img_bool is True:
		mode = "Img+Pos"
	elif pos_bool is False and img_bool is True:
		mode = "Img"
	else:
		mode = "Pos"

	return run_name + f"_{mode}"


# Model name
def get_model_name(model_path, img_bool, pos_bool, tf):
	assert tf >= 0, "The teacher forcing argument has to be positive."
	assert pos_bool is True or img_bool is True, "At least pos_bool or img_bool should be True."

	if tf != 100:
		model_path += f"_tf_{tf}"

	if pos_bool is True and img_bool is True:
		mode = "Img+Pos"
	elif pos_bool is False and img_bool is True:
		mode = "Img"
	else:
		mode = "Pos"

	return f"{model_path}_{mode}.pt"


def get_folders(multi_cam, box_size, img_size, img_step, data_path, scene, video):
	if len(multi_cam) == 0:
		if box_size != 0:
			data_folders = [
				data_path + scene + video + f"/frames_({img_size}, {img_size})_box_{box_size}_step_{img_step}/"]
			return data_folders
		else:
			data_folders = [data_path + scene + video + f"/frames_({img_size}, {img_size})_step_{img_step}/"]
			return data_folders
	else:
		folders = TrajDataset.conf_to_folders(multi_cam)
		if box_size != 0:
			data_folders = [data_path + scenePath + f"/frames_({img_size}, {img_size})_box_{box_size}_step_{img_step}/"
							for scenePath in folders]
			return data_folders
		else:
			data_folders = [data_path + scenePath + f"/frames_({img_size}, {img_size})_step_{img_step}/" for scenePath
							in folders]
			return data_folders




