import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import itertools
from collections import defaultdict
from sklearn.model_selection import KFold
def define_method_style(info_dicts):
    colors = itertools.cycle(['r', 'b', 'm', 'c', 'y'])
    markers = itertools.cycle(['D', 'o', 's', '*', 'v', '^'])
    method_style = {}
    for info_dict in info_dicts:
        method = info_dict['method']
        num_sensors = info_dict['filtered_state'].shape[0]
        if method not in method_style:
            method_style[method] = {'marker': next(markers), 'colors': [next(colors) for _ in range(num_sensors)]}
    return method_style

def plot_estimates(t, y_true, info_dicts, col_idx, state_names, axs1, method_style, marksize, linwidth):
    for j, info_dict in enumerate(info_dicts):
        y_predicted = info_dict['predicted_state']
        y_filtered = info_dict['filtered_state']
        method = info_dict['method']
        sensor_idx = info_dict['sensor_idx']
        for i in range(col_idx.size):
            idx = col_idx[i]
            axs1[j, 2*i].plot(t, y_true[:, idx], 's-g', label='true', markersize = 1.5*marksize, linewidth=1.5*linwidth)
            axs1[j, 2*i+1].plot(t, y_true[:, idx], 's-g', label='true', markersize = 1.5*marksize, linewidth=1.5*linwidth)
            for k in range(y_filtered.shape[0]):  # Iterate over sensors
                style = method_style[method]
                color = style['colors'][k]
                axs1[j, 2*i].plot(t, y_predicted[k, :, idx], color=color, marker=style['marker'], label=f'x_pred s_ {k+1}', markersize=marksize, linewidth=linwidth)
                axs1[j, 2*i+1].plot(t, y_filtered[k, :, idx], color=color, marker=style['marker'], label=f'x_filt s_ {k+1}', markersize=marksize, linewidth=linwidth)
            axs1[j, 2*i].set_xlabel('Time')
            axs1[j, 2*i].set_title('Prediction of {} for s {} - {}'.format(state_names[i], sensor_idx, method))
            axs1[j, 2*i].legend()
            axs1[j, 2*i+1].set_xlabel('Time')
            axs1[j, 2*i+1].set_title('Filter of {} for s {} - {}'.format(state_names[i], sensor_idx, method))
            axs1[j, 2*i+1].legend()

def plot_errors(t, y_true, info_dicts, col_idx, state_names, axs2, method_style,marksize, linwidth):
    for j, info_dict in enumerate(info_dicts):
        y_predicted = info_dict['predicted_state']
        y_filtered = info_dict['filtered_state']
        method = info_dict['method']
        sensor_idx = info_dict['sensor_idx']
        for i in range(col_idx.size):
            idx = col_idx[i]
            for k in range(y_filtered.shape[0]):  # Iterate over sensors
                style = method_style[method]
                color = style['colors'][k]
                print(idx,y_true[:, idx],y_filtered[k, :, idx])
                errors_filtered = (y_true[:, idx] - y_filtered[k, :, idx])**2
                errors_predicted = (y_true[:, idx] - y_predicted[k, :, idx])**2
                axs2[j, 2*i].plot(t, errors_predicted, color=color, marker=style['marker'], label=f'pred s {k+1}', markersize = marksize,linewidth=linwidth)
                axs2[j, 2*i+1].plot(t, errors_filtered, color=color, marker=style['marker'], label=f'filt s {k+1}', markersize = marksize,linewidth=linwidth)
            axs2[j, 2*i].set_ylabel('MSE')
            axs2[j, 2*i].set_xlabel('Time')
            axs2[j, 2*i].set_title(' MSE {} - {} s -{}'.format(method,state_names[i], sensor_idx))
            axs2[j, 2*i].legend()
            axs2[j, 2*i+1].set_ylabel('MSE')
            axs2[j, 2*i+1].set_xlabel('Time')
            axs2[j, 2*i+1].set_title(' {} - {} s -{}'.format(method,state_names[i], sensor_idx))
            axs2[j, 2*i+1].legend()

def plot_aggregate_view_old(t, y_true, info_dicts, col_idx, state_names, axs3, method_style, marksize, linwidth):
    for i in range(col_idx.size):
        idx = col_idx[i]
        axs3[0, i].plot(t, y_true[:, idx], 's-g', label='True', markersize=2*marksize, linewidth=2*linwidth)
        axs3[0, i].set_ylabel(state_names[i])
    
    for method, style in method_style.items():
        filtered_states = [d['filtered_state'] for d in info_dicts if d['method'] == method]
        errors_states = [(y_true - d['filtered_state'])**2 for d in info_dicts if d['method'] == method]
    
        for k in range(len(filtered_states)):  # Iterate over sensors
            for sensor_idx, (filtered_state, errors_state) in enumerate(zip(filtered_states[k], errors_states[k])):
                for i in range(col_idx.size):
                    idx = col_idx[i]
                    color = style['colors'][sensor_idx]
                    axs3[0, i].plot(t, filtered_state[:, idx], color=color, marker=style['marker'], label='{} - {} sensor {}'.format(method, state_names[i], k+1), markersize = marksize, linewidth=linwidth)
                    axs3[0, i].set_xlabel('Time')
                    axs3[0, i].set_title('Filtered state - {}'.format(state_names[i]))
                    axs3[0, i].legend()

                    axs3[1, i].plot(t, errors_state[:, idx], color=color, marker=style['marker'], label='{} - {} sensor {}'.format(method, state_names[i], k+1), markersize = marksize,linewidth=linwidth)
                    axs3[1, i].set_xlabel('Time')
                    axs3[1, i].set_title('{}'.format(state_names[i]))
                    axs3[1, i].set_ylabel('MSE')
                    axs3[1, i].legend()

def plot_aggregate_view(t, y_true, info_dicts, col_idx, state_names, axs3, method_style, marksize, linwidth):
    for i in range(col_idx.size):
        idx = col_idx[i]
        axs3[0, i].plot(t, y_true[:, idx], 's-g', label='True', markersize=2*marksize, linewidth=2*linwidth)
        axs3[0, i].set_ylabel(state_names[i])
    
    for method, style in method_style.items():
        filtered_states = [d['filtered_state'] for d in info_dicts if d['method'] == method]
        errors_states = [(y_true - (d['filtered_state'][:,:, col_idx]).T)**2 for d in info_dicts if d['method'] == method]
        
        
        for i in range(col_idx.size):
            idx = col_idx[i]
            for sensor_idx, (filtered_state, errors_state) in enumerate(zip(filtered_states[0], errors_states[0])):
                color = style['colors'][sensor_idx]
                
                axs3[0, i].plot(t, filtered_state[:, idx], color=color, marker=style['marker'], label='{} - {} sensor {}'.format(method, state_names[i], sensor_idx), markersize = marksize, linewidth=linwidth)
                axs3[0, i].set_xlabel('Time')
                axs3[0, i].set_title('Filtered state - {}'.format(state_names[i]))
                axs3[0, i].legend()
                #MSE
                axs3[1, i].plot(t, errors_state[:, idx], color=color, marker=style['marker'], label='{} - {} sensor {}'.format(method, state_names[i], sensor_idx), markersize = marksize,linewidth=linwidth)
                axs3[1, i].set_xlabel('Time')
                axs3[1, i].set_title('{}'.format(state_names[i]))
                axs3[1, i].set_ylabel('MSE')
                axs3[1, i].legend()

        
               



def calculate_average_mse(y_true, info_dicts, col_idx, method_style):
    avg_mse = {}
    for method, style in method_style.items():
        method_info_dicts = [d for d in info_dicts if d['method'] == method]
        if method_info_dicts:
            n_sensors = method_info_dicts[0]['filtered_state'].shape[0]
            mse_values = np.zeros(n_sensors)
            for info_dict in method_info_dicts:
                y_filtered = info_dict['filtered_state']
                for k in range(n_sensors):  # Iterate over sensors
                    mse_values[k] += np.mean((y_true - (y_filtered[k, :, col_idx]).T)**2)
            mse_values /= len(method_info_dicts)  # Average over all instances of the same method
            for k in range(n_sensors):  # Add separate entry for each sensor
                avg_mse[f'{method}_s_{k+1}'] = mse_values[k]
    return avg_mse

def print_average_mse(avg_mse):
    print(" " * 5, end="")
    for key in avg_mse:
        print("{:<10}".format(key), end="")
    print("\nMSE ", end="")
    for key in avg_mse:
        print("{:<10.2f}".format(avg_mse[key]), end="")
    print("\n")

def calculate_disagreement(info_dicts):
    # Extract the unique methods from info_dicts
    methods = list(set(d['method'] for d in info_dicts))
    
    # Get the number of methods, time steps, and state variables
    num_methods = len(methods)
    num_timesteps = len(info_dicts[0]['filtered_state'][0])
    num_states = info_dicts[0]['filtered_state'][0].shape[1]

    # Initialize a 3D numpy array to store the disagreement for each method
    disagreements = np.zeros((num_methods, num_timesteps, num_states))

    # For each method, calculate the disagreement
    for m, method in enumerate(methods):
        # Filter the info_dicts for the current method
        filtered_info = [d for d in info_dicts if d['method'] == method]

        # If there are multiple sensors for this method, calculate the disagreement
        if len(filtered_info) > 1:
            # Extract the filtered states for all sensors
            filtered_states = np.stack([d['filtered_state'] for d in filtered_info], axis=0)

            # Calculate the variance across sensors for each time step and state variable
            disagreements[m] = np.var(filtered_states, axis=0)
    print(disagreements.shape)
    return disagreements






def analyze(t, y_true, info_dicts, col_idx, state_names, t_forecast=None, y_true_forecast = None):

    marksize = 1.5
    linwidth = 0.5

    #fig1, axs1 = plt.subplots(nrows=len(info_dicts), ncols=2*col_idx.size, figsize=(5*col_idx.size, 5*len(info_dicts)), squeeze=False)
    fig2, axs2 = plt.subplots(nrows=len(info_dicts), ncols=2*col_idx.size, figsize=(5*col_idx.size, 5*len(info_dicts)), squeeze=False)
    fig3, axs3 = plt.subplots(nrows=2, ncols=col_idx.size, figsize=(8*col_idx.size, 12), squeeze=False)

    method_style = define_method_style(info_dicts)
    methods = [d['method'] for d in info_dicts]
    #avg_disagremment = calculate_disagreement(info_dicts)

    #plot_disagreement(t, avg_disagremment, methods, col_idx, state_names)
    avg_mse = calculate_average_mse(y_true, info_dicts, col_idx, method_style)
    print_average_mse(avg_mse)
    plot_aggregate_view(t, y_true, info_dicts, col_idx, state_names, axs3, method_style, marksize, linwidth)
    #plot_estimates(t, y_true, info_dicts, col_idx, state_names, axs1, method_style, marksize, linwidth)
    plot_errors(t,y_true, info_dicts, col_idx, state_names, axs2, method_style,marksize, linwidth)

    
    #fig6, axs6 = plt.subplots(nrows=2, ncols=col_idx.size, figsize=(8*col_idx.size, 12), squeeze=False)



    if t_forecast is not None:
        fig4, axs4 = plt.subplots(nrows=len(info_dicts), ncols=col_idx.size, figsize=(5*col_idx.size, 5*len(info_dicts)), squeeze=False)
        fig5, axs5 = plt.subplots(nrows=len(info_dicts), ncols=col_idx.size, figsize=(5*col_idx.size, 5*len(info_dicts)), squeeze=False)
        plot_forecast(t_forecast, y_true_forecast, info_dicts, col_idx, state_names, axs4, method_style, marksize, linwidth)
        results=plot_errors_forecast(t_forecast,y_true_forecast, info_dicts, col_idx, state_names, axs5, method_style,marksize, linwidth)
        print(results)
        avg_mse_forecast= calculate_average_mse_forecast(y_true_forecast, info_dicts, col_idx, method_style)
        print("Forecasting MSE")
        print_average_mse(avg_mse_forecast)
        fig4.tight_layout()
        fig5.tight_layout()



    #fig1.tight_layout()
    #fig2.tight_layout()
    fig3.tight_layout()
  
    plt.show()





def plot_forecast(t, y_true, info_dicts, col_idx, state_names, axs1, method_style, marksize, linwidth):
    for j, info_dict in enumerate(info_dicts):
        y_forecasted = info_dict['forecasted_state']
        method = info_dict['method']
        sensor_idx = info_dict['sensor_idx']
        for i in range(col_idx.size):
            idx = col_idx[i]
            axs1[j, i].plot(t, y_true[:, idx], 's-g', label='true', markersize = 1.5*marksize, linewidth=1.5*linwidth)
            for k in range(y_forecasted.shape[0]):  # Iterate over sensors
                style = method_style[method]
                color = style['colors'][k]

                axs1[j, i].plot(t, y_forecasted[k, :, idx], color=color, marker=style['marker'], label=f'{state_names[i]}_forecast s_ {k+1}', markersize=marksize, linewidth=linwidth)

            
            axs1[j, i].set_xlabel('Time')
            axs1[j, i].set_title('Forecast of {} for s {} - {}'.format(state_names[i], sensor_idx, method))
            axs1[j, i].legend()


# def plot_errors_forecast(t, y_true, info_dicts, col_idx, state_names, axs2, method_style,marksize, linwidth):
#     for j, info_dict in enumerate(info_dicts):
#         y_forecasted = info_dict['forecasted_state']
#         method = info_dict['method']
#         sensor_idx = info_dict['sensor_idx']
#         for i in range(col_idx.size):
#             idx = col_idx[i]
#             for k in range(y_forecasted .shape[0]):  # Iterate over sensors
#                 style = method_style[method]
#                 color = style['colors'][k]
#                 errors_forecasted = (y_true[:, idx] - y_forecasted[k, :, idx])**2
#                                 # Compute ADE for this sensor
#                 ade = np.sqrt(np.mean(errors_forecasted))
#                 axs2[j, i].plot(t, errors_forecasted, color=color, marker=style['marker'], label=f'{state_names[i]}_s {k+1}', markersize = marksize,linewidth=linwidth)
#             axs2[j, i].set_ylabel('MSE')
#             axs2[j, i].set_xlabel('Time')
#             axs2[j, i].set_title(' MSE {} - {} s -{}'.format(method,state_names[i], sensor_idx))
#             axs2[j, i].legend()

def plot_errors_forecast(t, y_true, info_dicts, col_idx, state_names, axs2, method_style, marksize, linwidth):
    results = []
    
    for j, info_dict in enumerate(info_dicts):
        y_forecasted = info_dict['forecasted_state']
        method = info_dict['method']
        sensor_idx = info_dict['sensor_idx']
        
        sensor_results = {'sensor_idx': sensor_idx}
        
        for i in range(col_idx.size):
            idx = col_idx[i]
            
            for k in range(y_forecasted.shape[0]):  # Iterate over sensors
                style = method_style[method]
                color = style['colors'][k]
                
                # Compute squared errors
                errors_forecasted = (y_true[:, idx] - y_forecasted[k, :, idx])**2
                
                # Store the errors in the sensor_results dictionary
                sensor_results[f"error_{state_names[i]}_sensor_{k+1}"] = errors_forecasted
                
                # Sum the squared errors for each state at each time step
                if i == 0:
                    aggregated_errors_per_timestep = errors_forecasted
                else:
                    aggregated_errors_per_timestep += errors_forecasted
                # Compute and store the aggregated MSE for this sensor              
                axs2[j, i].plot(t, errors_forecasted, color=color, marker=style['marker'], label=f'{state_names[i]}_s {k+1}', markersize=marksize, linewidth=linwidth)
            
            axs2[j, i].set_ylabel('MSE')
            axs2[j, i].set_xlabel('Time')
            axs2[j, i].set_title(f'MSE {method} - {state_names[i]} s - {sensor_idx}')
            axs2[j, i].legend()
        
        # Add the aggregated errors to the sensor_results dictionary
        sensor_results['aggregated_error'] = aggregated_errors_per_timestep
        results.append(sensor_results)
    
    return results



def calculate_average_mse_forecast(y_true, info_dicts, col_idx, method_style):
    avg_mse = {}
    for method, style in method_style.items():
        method_info_dicts = [d for d in info_dicts if d['method'] == method]
        if method_info_dicts:
            n_sensors = method_info_dicts[0]['forecasted_state'].shape[0]
            mse_values = np.zeros(n_sensors)
            for info_dict in method_info_dicts:
                y_filtered = info_dict['forecasted_state']
                for k in range(n_sensors):  # Iterate over sensors
                    mse_values[k] += np.mean((y_true - (y_filtered[k, :, col_idx]).T)**2)
            mse_values /= len(method_info_dicts)  # Average over all instances of the same method
            for k in range(n_sensors):  # Add separate entry for each sensor
                avg_mse[f'{method}_s_{k+1}'] = mse_values[k]
    return avg_mse


def calculate_errors(y_true, info_dicts, col_idx, state_names):
    results = []
    
    for j, info_dict in enumerate(info_dicts):
        y_forecasted = info_dict['forecasted_state']
        method = info_dict['method']
        sensor_idx = info_dict['sensor_idx']
        
        sensor_results = {'sensor_idx': sensor_idx}
        
        for i in range(col_idx.size):
            idx = col_idx[i]
            
            for k in range(y_forecasted.shape[0]):  # Iterate over sensors
                # Compute squared errors
                errors_forecasted = (y_true[:, idx] - y_forecasted[k, :, idx])**2
                
                # Store the errors in the sensor_results dictionary
                sensor_results[f"error_{state_names[i]}_sensor_{k+1}"] = errors_forecasted
                
                # Sum the squared errors for each state at each time step
                if i == 0:
                    aggregated_errors_per_timestep = errors_forecasted
                else:
                    aggregated_errors_per_timestep += errors_forecasted
                
        # Add the aggregated errors to the sensor_results dictionary
        sensor_results['aggregated_error'] = aggregated_errors_per_timestep
        results.append(sensor_results)
    
    return results

def calculate_mean_of_results(final_results):
    # Initialize a defaultdict to hold the sum of values for each key and timestep
    sum_dict = defaultdict(lambda: defaultdict(float))
    # Initialize a defaultdict to hold the count of values for each key and timestep
    count_dict = defaultdict(lambda: defaultdict(int))
    
    for result in final_results:
        for key, value in result.items():
            if key == 'sensor_idx':
                continue  # Skip the 'sensor_idx' key as it's not a metric
            for timestep, val in enumerate(value):
                sum_dict[key][timestep] += val  # Add the value of the current key at the current timestep
                count_dict[key][timestep] += 1  # Increment the count for the current key at the current timestep
    
    # Calculate the mean for each key and timestep
    mean_dict = {key: {timestep: sum_value / count_dict[key][timestep] for timestep, sum_value in timestep_dict.items()} for key, timestep_dict in sum_dict.items()}
    
    return mean_dict



# import numpy as np
# from collections import defaultdict

# def cross_validate_kalman_filter(traj_positions_list, initial_state, dt, initial_covariance, measurement_noise, state_transition, control_input, observation_matrix, col_idx, state_names):
#     # Define a range of possible sigma_p and sigma_v values to test
#     sigma_p_values = np.linspace(0.1, 1.0, 10)
#     sigma_v_values = np.linspace(0.01, 0.1, 10)
    
#     # Initialize variables to hold the best parameters and the lowest error
#     best_sigma_p = None
#     best_sigma_v = None
#     lowest_error = float('inf')
    
#     # Initialize 5-fold cross-validation
#     kf = KFold(n_splits=5)
    
#     # Loop over each fold
#     for train_index, val_index in kf.split(traj_positions_list):
#         train_data = [traj_positions_list[i] for i in train_index]
#         val_data = [traj_positions_list[i] for i in val_index]
        
#         # Loop over each candidate sigma_p and sigma_v value
#         for sigma_p in sigma_p_values:
#             for sigma_v in sigma_v_values:
#                 process_noise = np.array([
#                     [sigma_p ** 2, 0, 0, 0],
#                     [0, sigma_p ** 2, 0, 0],
#                     [0, 0, sigma_v ** 2, 0],
#                     [0, 0, 0, sigma_v ** 2]
#                 ])
                
#                 # Initialize Kalman Filter with the current sigma_p and sigma_v
#                 kf = KalmanFilter(
#                     initial_state=initial_state,
#                     measurement_period=dt,
#                     initial_covariance=initial_covariance,
#                     process_noise=process_noise,
#                     measurement_noise=measurement_noise,
#                     state_transition_func=state_transition,
#                     control_input_func=control_input,
#                     observation_matrix=observation_matrix,
#                     sensor_idx=f"{sigma_p}_{sigma_v}"
#                 )
                
#                 # Initialize error for this sigma_p and sigma_v
#                 total_error = 0
                
#                 # Loop over each trajectory in the validation set
#                 for positions in val_data:
#                     reshaped_input_positions = [row.reshape(-1, 1) for row in positions.input_noisy_measures]
#                     reshaped_input_positions = np.array(reshaped_input_positions)
#                     lkf_res = kf.run(reshaped_input_positions)
                    
#                     # Calculate errors using your function
#                     results = calculate_errors(positions.ground_truth_forecast, [lkf_res], col_idx, state_names)
                    
#                     # Add the 'aggregated_error' to 'total_error'
#                     for result in results:
#                         total_error += np.mean(result['aggregated_error'])
                
#                 # Calculate mean error for this sigma_p and sigma_v
#                 mean_error = total_error / len(val_data)
                
#                 # Update best parameters if this sigma_p and sigma_v have lower error
#                 if mean_error < lowest_error:
#                     best_sigma_p = sigma_p
#                     best_sigma_v = sigma_v
#                     lowest_error = mean_error
    
#     return best_sigma_p, best_sigma_v, lowest_error

# # Usage
# best_sigma_p, best_sigma_v, lowest_error = cross_validate_kalman_filter(traj_positions_list, initial_state, dt, initial_covariance, measurement_noise, state_transition, control_input, observation_matrix, col_idx, state_names)
# print(f"The best sigma_p value is {best_sigma_p} and the best sigma_v value is {best_sigma_v} with a mean error of {lowest_error}")
