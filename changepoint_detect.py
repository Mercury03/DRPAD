import matplotlib.pyplot as plt
import numpy as np
from data.data_loader import *
import ruptures as rpt
from typing import List, Tuple, Optional, Union, Dict, Any
import os


class ChangePointDetector:
    """Change Point Detector class, integrating various change point detection methods"""
    
    @staticmethod
    def detect_with_window(data: np.ndarray, min_size: int, model: str = 'l2', 
                         width: int = 100, jump: int = 2, pen: Optional[float] = None) -> Tuple[List[int], Any]:
        """Detect change points using sliding window method

        Args:
            data: Input data
            min_size: Minimum segment size
            model: Loss model ('l1', 'l2', 'rbf', etc.)
            width: Window width
            jump: Window sliding step
            pen: Penalty parameter, if None, use preset number of breakpoints

        Returns:
            List of change points and scoring function
        """
        algo = rpt.Window(width=width, jump=jump, model=model, min_size=min_size).fit(data)
        
        if pen is None:
            my_bkps = algo.predict(n_bkps=2)
        else:
            my_bkps = algo.predict(pen=pen)
            
        if len(my_bkps) <= 1:
            my_bkps = algo.predict(n_bkps=1)
            
        return my_bkps, algo.score

    @staticmethod
    def detect_with_pelt(data: np.ndarray, model: str = 'l1', min_size: int = 400, 
                        jump: int = 2, pen: float = None) -> List[int]:
        """Detect change points using PELT algorithm

        Args:
            data: Input data
            model: Loss model
            min_size: Minimum segment size
            jump: Step size
            pen: Penalty parameter

        Returns:
            List of change points
        """
        algo = rpt.Pelt(model=model, min_size=min_size, jump=jump).fit(data)
        return algo.predict(pen=pen)

    @staticmethod
    def detect_two_stage(data: np.ndarray, min_size: int, model: str = 'l2', 
                       width: int = 100, pen: float = 8) -> List[List[int]]:
        """Two-stage change point detection: first coarse localization with large step size, 
           then precise localization with small step size

        Args:
            data: Input data
            min_size: Minimum segment size
            model: Loss model
            width: Window width
            pen: Base penalty coefficient

        Returns:
            List of change point lists
        """
        # Stage 1: Coarse localization with large step size
        algo = rpt.Pelt(model='l2', min_size=min_size, jump=500).fit(data)
        coarse_bkps = algo.predict(pen=(np.log(len(data)) * np.std(data)**2) * pen)
        print(f"Coarse change points: {coarse_bkps}")
        
        # Stage 2: Precise localization with small step size
        window_size = 1000  # Window radius for precise localization
        cp_list = []
        
        for bk in coarse_bkps:
            # Ensure valid index range
            start_idx = max(0, bk - window_size)
            end_idx = min(len(data), bk + window_size)
            
            X = data[start_idx:end_idx]
            
            # Check data validity
            if np.isnan(X).any() or np.isinf(X).any():
                print(f"Warning: Data around change point {bk} contains NaN or Inf values, skipped")
                continue
                
            std_X = np.std(X)
            if std_X == 0:
                print(f"Warning: Data around change point {bk} has zero standard deviation, skipped")
                continue
                
            # Calculate penalty parameter
            pen_2 = (np.log(len(X)) * std_X**2) * pen * 3/4
            
            # Precise localization of change points
            my_bkps_2, _ = ChangePointDetector.detect_with_window(
                X, min_size=300, model='l1', width=200, jump=2, pen=pen_2
            )
            
            # Convert back to original coordinates
            my_bkps_2 = [start_idx + i for i in my_bkps_2]
            
            # Remove the last element in the list (usually the end of data)
            if len(my_bkps_2) >= 1:
                my_bkps_2.pop()
                
            print(f"Precise change points: {my_bkps_2}")
            cp_list.append(my_bkps_2)
            
        return cp_list

    @staticmethod
    def remove_close_points(cp_list: List[int], threshold: int = 200) -> List[int]:
        """Remove change points that are too close to each other

        Args:
            cp_list: List of change points
            threshold: Minimum distance threshold

        Returns:
            Filtered list of change points
        """
        if not cp_list:
            return []
            
        # Sort first to ensure order
        cp_list = sorted(cp_list)
        
        # Filter nearby points
        filtered_points = [cp_list[0]]
        for point in cp_list[1:]:
            if point - filtered_points[-1] >= threshold:
                filtered_points.append(point)
                
        return filtered_points

    @staticmethod
    def detect_multivariate(data: np.ndarray, min_size: int, pen: float = 8) -> List[List[int]]:
        """Detect change points for each feature in multivariate data

        Args:
            data: Multivariate input data, shape [n_samples, n_features]
            min_size: Minimum segment size
            pen: Penalty parameter

        Returns:
            List of change points for each feature
        """
        cp_list_all = []
        print(f'Penalty parameter: {pen}')
        
        for i in range(data.shape[1]):
            X = data[:, i].reshape(-1, 1)
            cp_list = ChangePointDetector.detect_two_stage(X, min_size, pen=pen)
            
            # Flatten nested list
            cp_list = [item for sublist in cp_list for item in sublist]
            
            # Remove points that are too close
            cp_list = ChangePointDetector.remove_close_points(cp_list)
            
            print(f"Change points for feature {i}: {cp_list}")
            cp_list_all.append(cp_list)
            print(f'Feature {i} detection completed')
            
        return cp_list_all

    @staticmethod
    def detect_single_feature(data: np.ndarray, feature_idx: int, min_size: int, pen: float = 8, 
                             show_plot: bool = False) -> List[int]:
        """Detect change points for a single feature and visualize

        Args:
            data: Multivariate input data
            feature_idx: Index of the feature to detect
            min_size: Minimum segment size
            pen: Penalty parameter
            show_plot: Whether to display the plot

        Returns:
            List of change points
        """
        X = data[:, feature_idx]
        cp_list = ChangePointDetector.detect_two_stage(X, min_size, pen=pen)
        cp_list = [item for sublist in cp_list for item in sublist]
        cp_list = ChangePointDetector.remove_close_points(cp_list)
        
        print(f"Change points for feature {feature_idx}: {cp_list}")
        
        ChangePointDetector.visualize_change_points(
            X, cp_list=cp_list, show_plot=show_plot, feature_idx=feature_idx
        )
        
        return cp_list

    @staticmethod
    def visualize_change_points(data: np.ndarray, score=None, cp_list=None, 
                               anomaly_label=None, anomaly_pre=None, anomaly_pre_true=None, 
                               show_plot: bool = False, feature_idx: int = 0) -> None:
        """Visualize time series and its change points

        Args:
            data: Input time series
            score: Scoring function
            cp_list: List of change points
            anomaly_label, anomaly_pre, anomaly_pre_true: Anomaly point annotations
            show_plot: Whether to display the plot
            feature_idx: Feature index (for saving file)
        """
        plt.figure(figsize=(10, 5))
        
        if score is not None:
            plt.subplot(2, 1, 1)
        else:
            plt.subplot(1, 1, 1)

        plt.plot(data)
        
        # Annotate anomaly points
        markers = {'Anomaly_label': 'red', 'Anomaly_pre': 'blue', 'Anomaly_pre_true': 'green'}
        annotations = {
            'Anomaly_label': anomaly_label,
            'Anomaly_pre': anomaly_pre,
            'Anomaly_pre_true': anomaly_pre_true
        }
        
        for label, points in annotations.items():
            if points is not None:
                plt.scatter(points, data[points], color=markers[label], label=label)
                
        # Annotate change points
        if cp_list:
            for bk in cp_list:
                plt.axvline(bk, color='g')
                
        plt.legend()
        
        # Display scores
        if score is not None:
            plt.subplot(2, 1, 2)
            plt.plot(score)
            plt.title("Change Point Score")
            
        plt.tight_layout()
            
        if show_plot:
            plt.show()
            
        # Save high-resolution image
        plt.savefig(f'change_points_plot_{feature_idx}.png', dpi=300)
        plt.close()

    @staticmethod
    def inspect_local_segment(data: np.ndarray, mid_point: int, feature_dim: int, 
                            window_size: int = 700) -> None:
        """Inspect changes around a specific time point

        Args:
            data: Input data
            mid_point: Center time point
            feature_dim: Feature dimension to inspect
            window_size: Window radius
        """
        # Ensure valid index range
        start_idx = max(0, mid_point - window_size)
        end_idx = min(len(data), mid_point + window_size)
        
        X = data[start_idx:end_idx, feature_dim]
        
        my_bkps, score = ChangePointDetector.detect_with_window(
            X, model='l1', width=200, jump=2, min_size=300, 
            pen=(np.log(len(X)) * np.std(X)**2) * 8
        )
        
        ChangePointDetector.visualize_change_points(X, score=score, cp_list=my_bkps, show_plot=True)


class DatasetHandler:
    """Dataset handling class for loading various datasets and processing change point detection"""
    
    @staticmethod
    def get_dataset_config(dataset_name: str) -> Dict[str, Any]:
        """Get configuration parameters for a dataset
        
        Args:
            dataset_name: Dataset name
            
        Returns:
            Dictionary of configuration parameters
        """
        configs = {
            'SMD':  {'min_size': 1500, 'pen': 8},
            'SWaT': {'min_size': 30000, 'pen': 20},
            'WADI': {'min_size': 1500, 'pen': 28},
            'MSL':  {'min_size': 1500, 'pen': 28},  # No segmentation needed
            'SMAP': {'min_size': 1500, 'pen': 28},
            'NAB':  {'min_size': 1500, 'pen': 8},   # No segmentation needed
            'MBA':  {'min_size': 2000, 'pen': 8},
            'MSDS': {'min_size': 5000, 'pen': 8},
            'PSM':  {'min_size': 1000, 'pen': 8}
        }
        
        if dataset_name not in configs:
            raise ValueError(f"Unknown dataset name: {dataset_name}")
            
        return configs[dataset_name]
        
    @staticmethod
    def load_dataset(dataset_name: str) -> Tuple[Dataset, Dict[str, Any]]:
        """Load the specified dataset
        
        Args:
            dataset_name: Dataset name
            
        Returns:
            Dataset object and configuration parameters
        """
        config = DatasetHandler.get_dataset_config(dataset_name)
        
        dataset_classes = {
            'SMD': Dataset_SMD,
            'SWaT': Dataset_SWaT,
            'WADI': Dataset_WADI,
            'MSL': Dataset_MSL,
            'SMAP': Dataset_SMAP,
            'NAB': Dataset_NAB,
            'MBA': Dataset_MBA,
            'MSDS': Dataset_MSDS,
            'PSM': Dataset_PSM
        }
        
        if dataset_name not in dataset_classes:
            raise ValueError(f"Unknown dataset name: {dataset_name}")
            
        dataset = dataset_classes[dataset_name](
            input_len=50, N='normal', flag='test', data_path=f'data/{dataset_name}'
        )
        
        return dataset, config
        
    @staticmethod
    def process_and_save(dataset_name: str) -> None:
        """Process dataset and save change points
        
        Args:
            dataset_name: Dataset name
        """
        # If it's a dataset that doesn't need segmentation, save an empty array directly
        if dataset_name in ['MSL', 'NAB']:
            print(f"{dataset_name} dataset does not need segmentation processing")
            save_dir = f'data/{dataset_name}'
            os.makedirs(save_dir, exist_ok=True)
            np.save(f'{save_dir}/cp_list_all_{dataset_name}_test.npy', np.array([]))
            np.save(f'{save_dir}/cp_list_all_{dataset_name}_train.npy', np.array([]))
            return
        
        # Load dataset and configuration
        dataset, config = DatasetHandler.load_dataset(dataset_name)
        min_size, pen = config['min_size'], config['pen']
        
        # Load test and training data
        data_test = dataset.get_test()
        data_train = dataset.get_train()
        
        print(f'Test data shape: {data_test.shape}')
        print(f'Training data shape: {data_train.shape}')
        print('Data loaded, starting change point detection...')
        
        # Detect change points
        change_points_test = ChangePointDetector.detect_multivariate(data_test, min_size, pen)
        change_points_train = ChangePointDetector.detect_multivariate(data_train, min_size, pen)
        
        print('Change point detection completed, saving data...')
        
        # Save test data change points
        DatasetHandler._save_change_points(change_points_test, dataset_name, 'test')
        
        # Save training data change points
        DatasetHandler._save_change_points(change_points_train, dataset_name, 'train')
        
        print('Data saving completed!')
    
    @staticmethod
    def _save_change_points(change_points: List[List[int]], dataset_name: str, split: str) -> None:
        """Save change points to NPY file
        
        Args:
            change_points: List of change points
            dataset_name: Dataset name
            split: 'train' or 'test'
        """
        if not change_points:
            # If there are no change points, save an empty array
            save_dir = f'data/{dataset_name}'
            os.makedirs(save_dir, exist_ok=True)
            np.save(f'{save_dir}/cp_list_all_{dataset_name}_{split}.npy', np.array([]))
            return
        
        # Find the length of the longest subsequence
        max_length = max(len(sublist) for sublist in change_points)
        
        if max_length == 0:
            # If all sublists are empty
            padded_data = np.zeros((len(change_points), 1), dtype=int)
        else:
            # Pad all subsequences to the same length
            padded_data = np.array([
                np.pad(sublist, (0, max_length - len(sublist)), 'constant') 
                for sublist in change_points
            ])
        

        padded_data = padded_data.astype(int)
        print(f"{split} data change points:\n{padded_data}")
        

        save_dir = f'data/{dataset_name}'
        os.makedirs(save_dir, exist_ok=True)
        np.save(f'{save_dir}/cp_list_all_{dataset_name}_{split}.npy', padded_data)


if __name__ == "__main__":

    dataset_name = 'SMD'
    DatasetHandler.process_and_save(dataset_name)