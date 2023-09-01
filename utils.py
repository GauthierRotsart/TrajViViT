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


def get_model_name(model_path, img_bool, pos_bool):
	if img_bool:
		model_path += "Img"
		if pos_bool:
			return model_path + "+Pos.pt"
		else:
			return model_path + ".pt"
	else:
		if pos_bool:
			return model_path + "Pos.pt"
		else:
			print("The input is at least the positions or the images.")
			raise NotImplementedError


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




