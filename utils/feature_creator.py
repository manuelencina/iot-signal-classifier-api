import numpy as np


class FeatureCreator:
    def get_features(self, data: dict, sae_params: dict) -> list:
        return self._create_features(data, sae_params)
    
    def _create_features(self, data: dict, sae_params: dict) -> list:
        nclasses = sae_params["nclasses"]
        X, Y     = np.array([]), np.array([])
        for idx, s in enumerate(data.values()):
            amplitudes      = self._calculate_amplitudes(s, sae_params)
            binary_params   = {"current_column": idx, "rows": amplitudes.shape[0], "columns": nclasses}
            Y_binary        = self._create_binary_labels(binary_params)
            Y               = self._stack_arrays(Y, Y_binary)
            X               = self._stack_arrays(X, amplitudes)
        X = self._normalize_data(X)
        return X, Y
    
    def _calculate_amplitudes(self, s: np.ndarray, sae_params: dict) -> np.ndarray:
        nframe      = sae_params["nframe"]
        frame_size  = sae_params["frame_size"]
        signals     = s.T[:, :nframe * frame_size].reshape(s.T.shape[0], nframe, frame_size)
        ft          = np.fft.fft(signals, axis=2)
        return np.abs(ft[:, :, :ft.shape[2]//2]).reshape(-1, ft.shape[2]//2)
    
    def _create_binary_labels(self, binary_params: dict) -> np.ndarray:
        current_column                   = binary_params["current_column"]
        rows                             = binary_params["rows"]
        columns                          = binary_params["columns"]
        binary_array                     = np.zeros((rows, columns))
        binary_array[:, current_column]  = 1
        return binary_array
    
    def _stack_arrays(self, arr: np.ndarray, new_arr: np.ndarray) -> np.ndarray:
        return np.concatenate((arr, new_arr)) if arr.shape[0] != 0 else new_arr
    
    def _normalize_data(self, X: np.ndarray, a = 0.01, b = 0.99):
        return ((X - X.min())/(X.max() - X.min())) * (b - a) + a