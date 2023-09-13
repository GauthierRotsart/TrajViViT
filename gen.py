import os
import random
import shutil

def create_train_val_sets(file_list, output_directory, seed=42, val_ratio=0.1):
    # Shuffle the list of files
    random.seed(seed)
    random.shuffle(file_list)
    
    # Calculate the number of validation samples
    n_val = int(len(file_list) * val_ratio)
    
    # Create train and validation directories
    train_dir = os.path.join(output_directory, 'train')
    val_dir = os.path.join(output_directory, 'val')
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # Create images and labels subdirectories
    train_images_dir = os.path.join(train_dir, 'images')
    train_labels_dir = os.path.join(train_dir, 'labels')
    val_images_dir = os.path.join(val_dir, 'images')
    val_labels_dir = os.path.join(val_dir, 'labels')
    
    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(train_labels_dir, exist_ok=True)
    os.makedirs(val_images_dir, exist_ok=True)
    os.makedirs(val_labels_dir, exist_ok=True)
    
    # Split the files into training and validation sets
    val_files = file_list[:n_val]
    train_files = file_list[n_val:]
    
    return train_files, val_files, train_images_dir, train_labels_dir, val_images_dir, val_labels_dir

# Define the path to the annotation file, the frames directory, and the output directory
annotation_file_path = "/home/dani/data/DroneDataset/nexus/video1/frames_(224, 224)_step_12/annotations_12.txt"
frames_directory = "/home/dani/data/DroneDataset/nexus/video1/frames_(224, 224)_step_12"
output_directory = "/home/dani/data/DroneDataset/nexus/video1/detection"

# List all jpg files in the frames directory
jpg_files = [f for f in os.listdir(frames_directory) if f.endswith('.jpg')]

# Create train and validation sets
train_files, val_files, train_images_dir, train_labels_dir, val_images_dir, val_labels_dir = create_train_val_sets(jpg_files, output_directory)

# Read the annotations_12.txt file
with open(annotation_file_path, "r") as f:
    lines = f.readlines()

# Skip the header line
lines = lines[1:]

# Loop through each line to create individual annotation files
for line in lines:
    # Split the line into its components
    components = line.strip().split(" ")
    
    # Extract the relevant annotation information
    trackid = components[0]
    xmin = float(components[1])/224.0
    ymin = float(components[2])/224.0
    xmax = float(components[3])/224.0
    ymax = float(components[4])/224.0
    frame = components[5]
    x_center=float(components[10])/224.0
    y_center= float(components[11])/224.0
    width = xmax - xmin
    height = ymax - ymin
    filename = str(trackid).zfill(3) + "_" + str(frame).zfill(5)
    
    # Determine if this file should go into train or val directory
    target_images_dir = train_images_dir if filename + '.jpg' in train_files else val_images_dir
    target_labels_dir = train_labels_dir if filename + '.jpg' in train_files else val_labels_dir
    
    # Copy the image to the target images directory
    shutil.copy(os.path.join(frames_directory, filename + '.jpg'), os.path.join(target_images_dir, filename + '.jpg'))
    
    # Create a new annotation file with the same name as the image but with a .txt extension
    annotation_filename = os.path.join(target_labels_dir, filename + '.txt')
    
    # Write the annotation information to the new file
    with open(annotation_filename, "w") as f:
        f.write(f"0 {x_center} {y_center} {width} {height}")

print("Annotation files and images have been copied to train and val directories.")
