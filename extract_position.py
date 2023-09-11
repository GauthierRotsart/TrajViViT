import pandas as pd
import numpy as np

class TrajPositions:
    def __init__(self, input_noisy_measures,input_positions_ground_truth, ground_truth_forecast):
        self.input_noisy_measures=input_noisy_measures
        self.input_positions_ground_truth = input_positions_ground_truth
        self.ground_truth_forecast = ground_truth_forecast

def normalize_positions(positions, image_size):
    return positions / image_size[0]

def add_velocity_columns(positions):
    # Create a zero array with the same number of rows and 2 columns
    zero_velocity = np.zeros((positions.shape[0], 2))
    
    # Concatenate the positions with the zero_velocity array
    return np.hstack((positions, zero_velocity))

def reshape_positions(positions):
    return positions.reshape(-1, 1)

def extract_positions_from_folder(folder_path, std_noise=1 , seed=42):
    # Construct the file path
    file_path = f"{folder_path}/bookstore_2_annotations_12.txt"
    
    # Load the data from the file
    raw_data = pd.read_csv(file_path, sep=" ", header=0)
    
    # Check if there are at least 20 positions
    if len(raw_data) < 20:
        raise ValueError("Not enough positions in the file.")
    
    # Extract the first 20 positions
    positions = raw_data[["x", "y"]].values[:20]
    
    # Assuming image is square, get the image size from the x coordinates
    #image_size = (positions[:, 0].max(), positions[:, 1].max())
    
    # Normalize the positions
    #positions = normalize_positions(positions, image_size)
        
    # Split the positions into input and ground truth
    input_positions_ground_truth = positions[:8]  # 8 positions * 4 (x, y, vx, vy)
    np.random.seed(seed)
    noise = np.random.normal(loc=0.0,scale=np.sqrt(std_noise), size=input_positions_ground_truth.shape)
    input_noisy_measures= input_positions_ground_truth+noise
    ground_truth_forecast = positions[8:]  # 12 positions * 4 (x, y, vx, vy)
    
    return TrajPositions(input_noisy_measures,
                         input_positions_ground_truth, 
                         ground_truth_forecast)

# Usage:
folder_path = "E:/data/TrajVivit"  # Replace with your actual folder path
#positions = extract_positions_from_folder(folder_path)
#print("Input Positions with Velocity:", positions.input_positions)
#print("Ground Truth with Velocity:", positions.ground_truth)


import pandas as pd
import numpy as np

# Your TrajPositions class and other functions remain the same

def generate_traj_positions_from_stream(raw_data, std_noise=1, seed=42):
    traj_positions_list = []
    
    # Group the data by track_id
    grouped = raw_data.groupby('track_id')
    
    for name, group in grouped:
        # Sort the group by frame
        group = group.sort_values('frame')
        
        # Convert positions to numpy array
        positions = group[['x', 'y']].values
        
        # Loop through the group to create TrajPositions
        for i in range(0, len(group) - 19):
            segment = positions[i:i+20]
            
            # Split the positions into input and ground truth
            input_positions_ground_truth = segment[:8]
            np.random.seed(seed)
            noise = np.random.normal(loc=0.0, scale=np.sqrt(std_noise), size=input_positions_ground_truth.shape)
            input_noisy_measures = input_positions_ground_truth + noise
            ground_truth_forecast = segment[8:]
            
            traj_positions = TrajPositions(input_noisy_measures, input_positions_ground_truth, ground_truth_forecast)
            traj_positions_list.append(traj_positions)
            
    return traj_positions_list

# Usage
file_path = "E:/data/TrajVivit/bookstore_2_annotations_12.txt"  # Replace with your actual file path
raw_data = pd.read_csv(file_path, sep=" ", header=0)

traj_positions_list = generate_traj_positions_from_stream(raw_data)

# Now, traj_positions_list contains TrajPositions objects for each 20-frame segment of each track
# for i, traj_positions in enumerate(traj_positions_list):
#     # print(f"TrajPositions object {i+1}")
#     # #print("Input Noisy Measures:", traj_positions.input_noisy_measures)
#     # print("Input Positions Ground Truth:", traj_positions.input_positions_ground_truth)
#     # print("Ground Truth Forecast:", traj_positions.ground_truth_forecast)
#     # print("------")
