import numpy as np


class KalmanFilter:
    def __init__(self,
                 initial_state,
                 measurement_period,
                 initial_covariance,
                 process_noise,
                 measurement_noise,
                 state_transition_func,
                 control_input_func,
                 observation_matrix,
                 sensor_idx=None):
        """
        Initializes the Kalman filter.

        Args:
            initial_state: The initial state estimate.
            measurement_period: The time between measurements.
            initial_covariance: The initial covariance estimate.
            process_noise: The process noise matrix.
            measurement_noise: The measurement noise matrix.
            state_transition_func: The state transition function.
            control_input_func: The control input function.
            observation_matrix: The observation matrix.
            sensor_idx: The sensor index.
        """
        self.initial_state=initial_state
        self._state = initial_state
        self._covariance = initial_covariance
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.transition_matrix = state_transition_func
        self.observation_matrix = observation_matrix
        self.control_matrix = control_input_func
        self.measurement_period = measurement_period
        self.sensor_idx = sensor_idx

    def predict(self, dt, u_dt=None):
        """
        Predicts the next state of the system.

        Args:
            dt: The time interval since the last prediction.
            u_dt: The control input at the current time step.
        """
        F = self.transition_matrix(dt)
        if u_dt is not None:
            G = self.control_matrix(dt)
            self._state = F @ self._state + G @ u_dt
        else:
            self._state = F @ self._state
        self._covariance = F @ self._covariance @ F.T + self.process_noise
        
    def update(self, measurement):
        """
        Updates the state estimate based on the current measurement.

        Args:
            measurement: The measurement at the current time step.
        """
        H = self.observation_matrix
        S = H @ self._covariance @ H.T + self.measurement_noise
        kalman_gain = self._covariance @ H.T @ np.linalg.inv(S)
        y_tilde = measurement - H @ self._state
        self._state = self._state + kalman_gain @ y_tilde

        self._covariance = (np.eye(self._state.size) - kalman_gain @ H) @ self._covariance

    def forecast(self, T:int):
        s = (1, T, self._state.size)
        forecasted_state = np.zeros(s)
        for t in range(T):
            self.predict(self.measurement_period)
            forecasted_state[0][t]=self._state.T
        return forecasted_state 
    def run(self, measurements, u_n=None,runame=None):
        """
        Runs the Kalman filter on the given measurements.

        Args:
            measurements: The measurements to filter.
            u_n: The control input at each time step.

        Returns:
            A dictionary containing the predicted and filtered states, sensor index, method, and trace of the a posteriori covariance matrix.
        """
        self.state=self.initial_state
        s = (1, len(measurements), self._state.size)
        predicted_state = np.zeros(s)
        filtered_state = np.zeros(s)
        a_posteriori_cov_trace = np.zeros(s)
        for i in range(0, len(measurements)):
            if u_n is not None:
                self.predict(self.measurement_period, np.array([[u_n[i]]]))
            else:
                self.predict(self.measurement_period)

            predicted_state[0][i] = self._state.T
            self.update(measurements[i])
            filtered_state[0][i] = self._state.T
            a_posteriori_cov_trace[0][i] = np.trace(self.observation_matrix @ self._covariance @self.observation_matrix.T)

        forecasted_state=self.forecast(T=12)
        return {'predicted_state': predicted_state, 
                'filtered_state': filtered_state, 
                'sensor_idx': self.sensor_idx, 
                'method': f'LKF_{runame}',
                'forecasted_state':forecasted_state, 
                'a_posteriori_cov_trace': a_posteriori_cov_trace}            
    def state(self):
        """
        Returns the current state estimate.

        Returns:
            The current state estimate.
        """
        return self._state

    @property
    def covariance(self):
        """
        Returns the current covariance estimate.

        Returns:
            The current covariance estimate.
        """
        return self._covariance


            