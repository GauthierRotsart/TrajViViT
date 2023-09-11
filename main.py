import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
from kalmanFilter import KalmanFilter
from analysis import analyze, calculate_mean_of_results
from extract_position import extract_positions_from_folder, generate_traj_positions_from_stream
import pandas as pd
from analysis import calculate_errors
from sklearn.model_selection import KFold
import numpy as np

############
file_path = "E:/data/TrajVivit/bookstore_2_annotations_12.txt"  # Replace with your actual file path
raw_data = pd.read_csv(file_path, sep=" ", header=0)

traj_positions_list = generate_traj_positions_from_stream(raw_data)


### SIMULATION PARAMETERS ###

dt = 0.4

### STATE MODEL PARAMETERS ###

k_v = 1 # std
k_w = 10

observation_matrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])  # H
measurement_noise = np.eye(2) * k_v **2  # R
#process_noise = np.eye(4) * (k_w ** 2)  # Q

sigma_x = 0.1  # Variance of the positional noise in the x-direction
sigma_y = 0.1  # Variance of the positional noise in the y-direction
sigma_vx = 0.01  # Variance of the velocity noise in the x-direction
sigma_vy = 0.01  # Variance of the velocity noise in the y-direction
process_noise = np.array([
    [sigma_x ** 2, 0, 0, 0],
    [0, sigma_y ** 2, 0, 0],
    [0, 0, sigma_vx ** 2, 0],
    [0, 0, 0, sigma_vy ** 2]
])
initial_state = np.array([[45], [22.5], [0], [0]])#np.zeros((4,1)) #$x_0$
initial_covariance = np.eye(4) * 10 * (k_w ** 2) #$P_0$

def state_transition(dt):
    F = np.array([[1, 0, dt, 0],
                  [0, 1, 0, dt],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]
                  ])
    return F

def control_input(dt):
    G = np.array([[0], [0], [0], [0]])
    return G

noise_variance=[0.1]
col_idx = np.array([0, 1])  # Replace with your actual column indices
state_names = ['x', 'y']  # Replace with your actual state names
final_results = []
lfs=[]
# for std_noise in noise_variance:
#     #folder_path = "E:/data/TrajVivit"  # Replace with your actual folder path
#     measurement_noise = np.eye(2) * std_noise **2  # R
#     kf = KalmanFilter(initial_state=initial_state,
#                       measurement_period=dt,
#                       initial_covariance=initial_covariance,
#                       process_noise=process_noise,
#                       measurement_noise=measurement_noise,
#                       state_transition_func=state_transition,
#                       control_input_func=control_input,
#                       observation_matrix=observation_matrix,
#                       sensor_idx = str(std_noise))
    
#     for i,positions in enumerate(traj_positions_list):
#         reshaped_input_positions = [row.reshape(-1, 1) for row in positions.input_noisy_measures]
#         reshaped_input_positions = np.array(reshaped_input_positions)
#         runame="std="+str(std_noise)+"trj"+str(i)
#         lkf_res = kf.run(reshaped_input_positions,runame=runame)
#         results = calculate_errors(positions.ground_truth_forecast, [lkf_res], col_idx, state_names)
#         lfs.append(lkf_res)
#         final_results.extend(results)
#     # Assuming y_true and other required variables are defined

# mean_results = calculate_mean_of_results(final_results)
# print(mean_results)

# Define a range of possible sigma_p and sigma_v values to test
sigma_p_values = np.linspace(0.1, 1.0, 10)  # Replace with your own range
sigma_v_values = np.linspace(0.01, 0.1, 10)  # Replace with your own range

# Initialize variables to hold the best parameters and the lowest error
best_sigma_p = None
best_sigma_v = None
lowest_error = float('inf')

# Initialize 5-fold cross-validation
kf = KFold(n_splits=5)

# Loop over each fold
for train_index, val_index in kf.split(traj_positions_list):
    train_data = [traj_positions_list[i] for i in train_index]
    val_data = [traj_positions_list[i] for i in val_index]
    
    # Loop over each candidate sigma_p and sigma_v value
    for sigma_p in sigma_p_values:
        for sigma_v in sigma_v_values:
            process_noise = np.array([
                [sigma_p ** 2, 0, 0, 0],
                [0, sigma_p ** 2, 0, 0],
                [0, 0, sigma_v ** 2, 0],
                [0, 0, 0, sigma_v ** 2]
            ])
            
            # Initialize Kalman Filter with the current sigma_p and sigma_v
            kf = KalmanFilter(
                initial_state=initial_state,
                measurement_period=dt,
                initial_covariance=initial_covariance,
                process_noise=process_noise,
                measurement_noise=measurement_noise,
                state_transition_func=state_transition,
                control_input_func=control_input,
                observation_matrix=observation_matrix,
                sensor_idx=f"{sigma_p}_{sigma_v}"
            )
            
            # Initialize error for this sigma_p and sigma_v
            total_error = 0
            
            # Loop over each trajectory in the validation set
            for positions in val_data:
                reshaped_input_positions = [row.reshape(-1, 1) for row in positions.input_noisy_measures]
                reshaped_input_positions = np.array(reshaped_input_positions)
                lkf_res = kf.run(reshaped_input_positions)
                
                # Calculate errors using your function
                results = calculate_errors(positions.ground_truth_forecast, [lkf_res], col_idx, state_names)
                
                # Add the 'aggregated_error' to 'total_error'
                for result in results:
                    total_error += np.mean(result['aggregated_error'])
            
            # Calculate mean error for this sigma_p and sigma_v
            mean_error = total_error / len(val_data)
            
            # Update best parameters if this sigma_p and sigma_v have lower error
            if mean_error < lowest_error:
                best_sigma_p = sigma_p
                best_sigma_v = sigma_v
                lowest_error = mean_error

print(f"The best sigma_p value is {best_sigma_p} and the best sigma_v value is {best_sigma_v} with a mean error of {lowest_error}")

    
    

    





#print("Input Positions with Velocity:", positions.input_positions)
#print('filter',lkf_res['filtered_state'])

#print("#############    FORECASTING         ##############")

#print("Forecasted state", lkf_res['forecasted_state'])
#print("Ground Truth of forecasting", positions.ground_truth)

T_filter=len(reshaped_input_positions)*dt
time_filter = np.arange(0, T_filter, dt)
T_forecast=4.8#(len(positions.ground_truth_forecast))*dt
time_forecast = np.arange(0, T_forecast, dt)



#analyze(time_filter,positions.input_positions_ground_truth, lfs ,
 #        np.array([0,1]),['x','y'],
  #       time_forecast,
   #      positions.ground_truth_forecast)
