# TrajViViT

TrajViViT: Trajectory forecasting with Video Vision Transformers on top-view image sequences

Repository of the master's thesis of:\
Nicolas Depré (GitHub: NicolasDepre)\
Arthur Franck (GitHub: arfranck)

We trained a modified Video Vision Transformer model for trajectory prediction on top-view image sequences. We used the Stanford Drone Dataset, which we modified, to train our model. You can doload the full dataset here: https://cvgl.stanford.edu/projects/uav_data/

For technical reasons, the image sizes were reduced, the frames were transformed to grayscale images and the targets were indicated with black boxes.


## Files and Directory Structure

- `data_creation.py`: Script for creating images with the desired size and annotated with the black box from the video frames.
- `TrajViViT.py`: Defines the architecture of the trajViVit model using the transformer network.
- `noam.py`: Implementation of the Noam learning rate scheduler.
- `runner.py`: Script to open the data and call training process from CLI.
- `train.py`: Script containing the training process of the trajViVit model.
- `test.py`: Script containing the test process of the trajViVit model.
- `traj_dataset.py`: Custom dataset class for loading trajectory and image data.
- `traj_plotter.ipynb`: Jupyter notebook providing tools for visualizing and analyzing trajectory predictions.
- `README.md`: The current file, providing an overview of the project, its usage, and file descriptions.
- `requirements.txt`: Contains the required python modules to run the project.

## Environment creation

In a terminal or in Anaconda prompt (windows), you can create the TrajViViT environment as follows:
For ubuntu 20: 
```bash
pip install -r requirements_20.txt
```

For ubuntu 22: 
```bash
pip install -r requirements_22.txt
```


### Example
```bash
python data_creation.py scene=bookstore video=0
```

## Dataset creation

By default, we sample the Standford Drone Dataset at 2.5 fps. This is most often used by others researchs. You only have to chose
a scene name (coupa, bookstore, ...) and the video number (0,1,...). The default scene is bookstore and the default video is 0.

### Example
```bash
python data_creation.py scene=bookstore video=0
```

If you want to create multiple datasets, you can do it as follows:

### Example
```bash
python data_creation.py -m scene=bookstore,coupa video=0,1,2
```

If you want to sample the Standford Drone Dataset at another frequency, you can use the img_step argument. For example, the following
command will sample at 1fps.

### Example
```bash
python data_creation.py scene=bookstore video=0 img_step=30
```

## Training a model 
By default, the training is done on the GPU 0. If you do not have a GPU, then the model will be trained on CPU. You can choose to use the
position (pos=True) and/or the image (img=True) as inputs of the network. For example, to train a network on coupa 2 with only the position: 

### Example
```bash
python data_path=/PATH/TO/DATA/ saving_path=/PATH/ runner.py scene=coupa video=2 pos=True 
```

If you want to train the same network on GPU2: 

### Example
```bash
python runner.py data_path=/PATH/TO/DATA/ saving_path=/PATH/ scene=coupa video=2 pos=True device=2
```

Or in multiGPU: 

### Example
```bash
python runner.py data_path=/PATH/TO/DATA/ saving_path=/PATH/ scene=coupa video=2 pos=True device=[0,1,2]
```

A final feature is the training on multi camera. To do that, you have to use the first letter of the scene and video number. So, you have
to use : 
b --> bookstore
c --> coupa
d --> deathCircle
g --> gates
h --> hyang
l --> little
n --> nexus
q --> quad 

For example, if you want to train a network with the position and image on bookstore 0 and coupa 2: 

### Example
```bash
python runner.py data_path=/PATH/TO/DATA/ saving_path=/PATH/ pos=True img=True multi_cam=[b0,c2]
```










## DEPRICATED

## Usage

Prepare your dataset and images according to the data format specified in data_creation.py.
Use data_creation.py to preprocess and organize your dataset for training.
Configure the model architecture in model.py.
Adjust hyperparameters and training settings in train.py. You can use the noam.py module to implement the Noam learning rate scheduler.
Train the model using:
bash
Copy code
python train.py
Once the model is trained, you can utilize runner.py for trajectory predictions on new data.
The traj_plotter.ipynb notebook provides visualization tools for analyzing and plotting trajectory predictions.


To run the program, use the following command-line arguments

```bash
python3 runner.py [-h] [--batch_size BATCH_SIZE] [--lr LR] [--gpu GPU] [--optimizer_name OPTIMIZER_NAME]
                 [--n_next N_NEXT] [--n_prev N_PREV] [--train_prop TRAIN_PROP] [--val_prop VAL_PROP]
                 [--test_prop TEST_PROP] [--img_step IMG_STEP] [--model_dimension MODEL_DIMENSION]
                 [--patch_size PATCH_SIZE] [--img_size IMG_SIZE] [--block_size BLOCK_SIZE] [--patch_depth PATCH_DEPTH]
                 [--model_depth MODEL_DEPTH] [--n_heads N_HEADS] [--mlp_dim MLP_DIM] [--n_epoch N_EPOCH]
                 [--teacher_forcing TEACHER_FORCING] [--name NAME] [--dataset DATASET]
                 [--scheduler SCHEDULER]
```

### Command Line Arguments

- `-h`, `--help`: Show this help message and exit.
- `--batch_size BATCH_SIZE`: Batch size for training.
- `--lr LR`: Learning rate for the optimizer.
- `--gpu GPU`: ID of the GPU to use.
- `--optimizer_name OPTIMIZER_NAME`: Name of the optimizer.
- `--n_next N_NEXT`: Number of next frames to predict.
- `--n_prev N_PREV`: Number of previous frames to use for prediction.
- `--train_prop TRAIN_PROP`: Proportion of data to use for training.
- `--val_prop VAL_PROP`: Proportion of data to use for validation.
- `--test_prop TEST_PROP`: Proportion of data to use for testing.
- `--img_step IMG_STEP`: Frame step size.
- `--model_dimension MODEL_DIMENSION`: Dimension of the model.
- `--patch_size PATCH_SIZE`: Size of the patches.
- `--img_size IMG_SIZE`: Size of the input frames.
- `--block_size BLOCK_SIZE`: Size of the block in the frames.
- `--patch_depth PATCH_DEPTH`: Patch depth.
- `--model_depth MODEL_DEPTH`: Depth of the model.
- `--n_heads N_HEADS`: Number of heads for the multi-head attention.
- `--mlp_dim MLP_DIM`: Dimension of the MLP in the Transformer.
- `--n_epoch N_EPOCH`: Number of epochs to train the model.
- `--teacher_forcing TEACHER_FORCING`: Number of epochs where teacher forcing is used.
- `--name NAME`: Name of the run on OneDB.
- `--dataset DATASET`: Config name of the datasets to use.
- `--scheduler SCHEDULER`: Config name of the datasets to use.

### Example
```bash
python3 runner.py --batch_size 16 --lr 0.00001 --gpu 2 --optimizer_name adam --n_next 12 --n_prev 8 --train_prop 0.9 --val_prop 0.05 --test_prop 0.05 --img_step 12 --model_dimension 1024 --patch_size 8 --img_size 64 --patch_depth 4 --model_depth 6 --n_heads 8 --mlp_dim 2048 --n_epoch 100 --teacher_forcing 50 --block_size 4 --dataset dc1 --scheduler noam
```
