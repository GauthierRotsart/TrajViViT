import torch
import wandb
import numpy as np
import random
from tqdm import tqdm
from utils import get_run_name
from traj_dataset import TrajDataset


class Trainer:

	def __init__(self, model, criterion, optimizer, scheduler, epochs, teacher_forcing, box_size, scene, video,
				 pos_bool, img_bool, multi_cam, save_run, saving_path, data_folders, n_prev, n_next, img_step, prop,
				 batch_size, img_size, verbose, device, mean=0, var=0):

		self.model = model
		self.data_folders = data_folders
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
		self.data_folders = data_folders
		self.n_prev = n_prev
		self.n_next = n_next
		self.img_step = img_step
		self.prop = prop
		self.box_size = box_size
		self.batch_size = batch_size
		self.img_size = img_size
		self.mean = mean
		self.var = var
		self.list_of_position = []

	def create_batch(self, src_list, tgt_list, coords_list, size):
		batch_src = torch.zeros((size, self.n_prev, self.img_size, self.img_size))
		batch_tgt = torch.zeros((size, self.n_next, 2))  # coord. in pixels
		batch_coord = torch.zeros((size, self.n_prev, 2))  # coord. in pixels
		for i in range(size):
			src = src_list[i]
			tgt = tgt_list[i]
			coords = coords_list[i]
			batch_src[i, :, :, :] = src[self.pos_list[i], :, :, :]
			batch_tgt[i, :, :] = tgt[self.pos_list[i], :, :]
			batch_coord[i, :, :] = coords[self.pos_list[i], :, :]

		return batch_src, batch_tgt, batch_coord

	def create_track_list(self, train_data):
		track_id = []
		track_pos = []

		for track in range(self.track_ids[0]):
			src, _, _ = train_data.get_track_data(track_id=track)
			for pos in range(len(src)):
				track_id.append(track)
				track_pos.append(pos)
		temp_list = list(zip(track_id, track_pos))
		random.shuffle(temp_list)
		res1, res2 = zip(*temp_list)
		res1, res2 = list(res1), list(res2)

		return list(res1), list(res2)

	# TRAINING LOOP
	def train(self):
		last_loss = 300
		best_loss = 300

		with tqdm(total=self.epochs, unit="epoch", desc=f"Training") as t_epoch:
			for epoch in range(self.epochs):
				train_loss = []
				self.model.train()
				for folder in self.data_folders:
					train_data = TrajDataset(n_prev=self.n_prev, n_next=self.n_next, img_step=self.img_step,
											 prop=self.prop, folder=folder, part=0, box_size=self.box_size,
											 verbose=self.verbose)
					self.track_ids = train_data.get_track_ids()
					track_id_list, track_pos_list = self.create_track_list(train_data=train_data)

					for pos_id in range(0, len(track_id_list), self.batch_size):
						if len(track_id_list) - pos_id > self.batch_size:
							size_batch = self.batch_size
						else:
							size_batch = len(track_id_list) - pos_id
						src_list = []
						coord_list = []
						tgt_list = []
						self.pos_list = []

						for idx_track in range(size_batch):
							src, coords, tgt = train_data.get_track_data(track_id=track_id_list[pos_id+idx_track])
							src_list.append(src)
							coord_list.append(coords)
							tgt_list.append(tgt)
							self.pos_list.append(track_pos_list[pos_id+idx_track])
							#del track_id_list[pos_id+idx_track]
							#del track_pos_list[pos_id+idx_track]

						batch_src, batch_tgt, batch_coord = self.create_batch(src_list=src_list, tgt_list=tgt_list,
																			  coords_list=coord_list, size=size_batch)

						x_train = batch_src.to(self.device)
						y_train = batch_tgt.to(self.device)
						src_coord = batch_coord.to(self.device)
						self.optimizer.zero_grad()

						if epoch < self.teacher_forcing:  # Teacher forcing approach
							pred, _ = self.model(video=x_train, tgt=y_train, src=src_coord)
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
						if isinstance(self.model, torch.nn.DataParallel):
							torch.save(self.model.module.state_dict(), self.saving_path + self.name)
						else:
							model_parallel = torch.nn.DataParallel(self.model).to(self.device)
							torch.save(model_parallel.module.state_dict(), self.saving_path + self.name)

				if self.save_run:
					wandb.log({"Training loss": np.mean(train_loss), "Epochs": epoch + 1})
					wandb.log({"Validation loss": current_loss, "Epochs": epoch + 1})
					if self.verbose:
						print(f"Saving best model for epoch: {epoch + 1}\n")

				last_loss = best_loss
				t_epoch.update(1)

	# VALIDATION LOOP
	def validation(self):
		with torch.no_grad():
			self.model.eval()

			val_loss = []
			for folder in self.data_folders:
				val_data = TrajDataset(n_prev=self.n_prev, n_next=self.n_next, img_step=self.img_step, prop=self.prop,
									   folder=folder, part=1, box_size=self.box_size, verbose=self.verbose)
				self.track_ids = val_data.get_track_ids()
				track_id_list, track_pos_list = self.create_track_list(train_data=val_data)

				for pos_id in range(0, len(track_id_list), self.batch_size):
					if len(track_id_list) - pos_id > self.batch_size:
						size_batch = self.batch_size
					else:
						size_batch = len(track_id_list) - pos_id
					src_list = []
					coord_list = []
					tgt_list = []
					self.pos_list = []

					for idx_track in range(size_batch):
						src, coords, tgt = val_data.get_track_data(track_id=track_id_list[pos_id + idx_track])
						src_list.append(src)
						coord_list.append(coords)
						tgt_list.append(tgt)
						self.pos_list.append(track_pos_list[pos_id + idx_track])
					# del track_id_list[pos_id+idx_track]
					# del track_pos_list[pos_id+idx_track]

					batch_src, batch_tgt, batch_coord = self.create_batch(src_list=src_list, tgt_list=tgt_list,
																		  coords_list=coord_list, size=size_batch)

					x_val = batch_src.to(self.device)
					y_val = batch_tgt.to(self.device)
					src_coord = batch_coord.to(self.device)

					future = None
					for k in range(y_val.shape[1]):
						pred, future = self.model(video=x_val, tgt=future, src=src_coord)

					loss = self.criterion(pred, y_val)
					val_loss.append(loss.item())
		return np.mean(val_loss)

	# TEST LOOP
	def test(self):
		with torch.no_grad():
			self.model.eval()

			test_loss_x = []
			test_loss_y = []
			for folder in self.data_folders:
				test_data = TrajDataset(n_prev=self.n_prev, n_next=self.n_next, img_step=self.img_step,
										prop=self.prop,
										folder=folder, part=2, box_size=self.box_size, verbose=self.verbose)
				track_ids = test_data.get_track_ids()

				for track_id in track_ids:
					src, coords, tgt = test_data.get_track_data(track_id=track_id)
					self.list_of_position = [i for i in range(len(src))]
					random.shuffle(self.list_of_position)
					start = 0
					for stop in range(self.batch_size, len(src), self.batch_size):
						if len(src) - stop > self.batch_size:
							batch_src, batch_tgt, batch_coord = self.create_batch(src=src, tgt=tgt, coords=coords,
																				  start=start, stop=stop,
																				  size=self.batch_size)
						else:
							new_batch_size = len(src) - stop
							batch_src, batch_tgt, batch_coord = self.create_batch(src=src, tgt=tgt, coords=coords,
																				  start=stop, stop=len(src),
																				  size=new_batch_size)
						start = stop

						x_test = batch_src.to(self.device)
						y_test = batch_tgt.to(self.device)
						src_coord = batch_coord.to(self.device) + torch.normal(mean=self.mean, std=np.sqrt(self.var),
																			   size=batch_coord.shape).to(self.device)

					future = None
					for k in range(y_test.shape[1]):
						pred, future = self.model(video=x_test, tgt=future, src=src_coord)

					loss = torch.abs(pred - y_test)
					test_loss_x.append(torch.mean(loss, dim=0)[:, 0].detach().cpu().numpy())
					test_loss_y.append(torch.mean(loss, dim=0)[:, 1].detach().cpu().numpy())

				error_x = np.mean(test_loss_x, axis=0)
				error_y = np.mean(test_loss_y, axis=0)
		return error_x, error_y
