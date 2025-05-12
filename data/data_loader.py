import os
import warnings
import numpy as np
import pandas as pd
import json
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

def process_SN(data, cp_file):
    # Read the change point file
    change_points = np.load(cp_file)
    
    # Initialize the processed data list
    processed_data = []
    data_copy = data.copy()
    
    # Process data for each dimension
    for dim in range(data_copy.shape[1]):
        dim_data = data_copy[:, dim]
        segments = []
        
        # Split based on change points
        prev_cp = 0
        for cp in change_points[dim]:
            cp = int(cp)  # Ensure change point is an integer
            if cp == 0 or cp > len(dim_data):
                break
            segments.append(dim_data[prev_cp:cp])
            prev_cp = cp
            
        if prev_cp < len(dim_data):
            segments.append(dim_data[prev_cp:])
        
        # Standardize each segment
        scaler = StandardScaler()
        standardized_segments = []
        for segment in segments:
            if len(segment) > 0:  # Check if segment is empty
                standardized_segment = scaler.fit_transform(segment.reshape(-1, 1)).flatten()
                standardized_segments.append(standardized_segment)
            else:
                standardized_segments.append(segment)  # If segment is empty, keep as is
        
        # Concatenate standardized segments
        processed_dim_data = np.concatenate(standardized_segments)
        # If the concatenated length is greater than original, remove the excess
        if len(processed_dim_data) > len(dim_data):
            processed_dim_data = processed_dim_data[:len(dim_data)]

        processed_data.append(processed_dim_data)
    
    # Convert processed data to numpy array
    return np.array(processed_data).T


class BaseDataset(Dataset):
    """Base class for all datasets"""
    def __init__(self, flag='train', input_len=0, data_path=None, data_process=True, N='LIN',
                 partial_train=False, ratio=0.2):
        # Basic information
        self.input_len = input_len
        self.N = N
        
        if self.N == 'normal':
            self.scaler = StandardScaler()
            
        # Initialization
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.data_path = data_path
        self.data_process = data_process
        self.partial_train = partial_train
        self.ratio = ratio if self.partial_train else 1
        
        # Subclass implements specific data loading
        self.train = None
        self.val = None
        self.test = None
        self.test_labels = None
        self.feature_map = None
        self.fc_struc = None
        self.fc_edge_index = None
    
    def __getitem__(self, index):
        r_begin = index
        r_end = r_begin + self.input_len
        if self.set_type == 0:
            seq_x = self.train[r_begin:r_end]
        elif self.set_type == 1:
            seq_x = self.val[r_begin:r_end]
        else:
            seq_x = self.test[r_begin:r_end]
        
        return np.nan_to_num(seq_x)

    def __len__(self):
        if self.set_type == 0:
            return self.train.shape[0] - self.input_len + 1
        elif self.set_type == 1:
            return self.val.shape[0] - self.input_len + 1
        else:
            return self.test.shape[0] - self.input_len + 1

    def get_label(self):
        return self.test_labels

    def get_test(self):
        return self.test
    
    def get_train(self):
        return self.train
    
    def get_fc_graph_struc(self):
        struc_map = {}
        for ft in self.feature_map:
            if ft not in struc_map:
                struc_map[ft] = []
            for other_ft in self.feature_map:
                if other_ft is not ft:
                    struc_map[ft].append(other_ft)
        return struc_map

    def build_loc_net(self):
        index_feature_map = self.feature_map
        edge_indexes = [[], []]
        
        for node_name, node_list in self.fc_struc.items():
            if node_name not in index_feature_map:
                index_feature_map.append(node_name)

            p_index = index_feature_map.index(node_name)
            for child in node_list:
                if child not in index_feature_map:
                    print(f'error: {child} not in index_feature_map')
                    continue

                c_index = index_feature_map.index(child)
                edge_indexes[0].append(c_index)
                edge_indexes[1].append(p_index)
        return edge_indexes
    
    def process_data(self, data, is_train=True, cp_file=None):
        """Common data processing method"""
        data = np.nan_to_num(data)
        
        if self.N == 'normal':
            if is_train:
                self.scaler.fit(data)
            data = self.scaler.transform(data)
            
        elif self.N == 'SN' and cp_file is not None:
            data = process_SN(data, cp_file=cp_file)
            
        return np.nan_to_num(data)


class Dataset_SMD(BaseDataset):
    def __init__(self, flag='train', input_len=0, data_path='./data/SMD', data_process=True, N='LIN',
                 partial_train=False, ratio=0.2):
        super().__init__(flag, input_len, data_path, data_process, N, partial_train, ratio)
        self.__read_data__()

    def __read_data__(self):
        data_train = np.load(self.data_path + "/SMD_train.npy")
        data_test = np.load(self.data_path + "/SMD_test.npy")
        data_test_label = np.load(self.data_path + "/SMD_test_label.npy")
        
        if self.data_process:
            data_train = np.delete(data_train, 7, 1)
            data_test = np.delete(data_test, 7, 1)
        
        print('data_train:', data_train.shape)
        print('data_test:', data_test.shape)
        
        # Process training data
        train_cp_file = './data/SMD/cp_list_all_SMD_train_improved.npy' if self.N == 'SN' else None
        data_train = self.process_data(data_train, is_train=True, cp_file=train_cp_file)
        print('data_train_processed:', data_train.shape)
        
        # Split training and validation sets
        data_len = data_train.shape[0]
        self.train = data_train[:int(data_len * 0.8 * self.ratio), :]
        self.val = data_train[int(data_len * 0.8):, :]
        
        # Feature map and edge indices
        self.feature_map = [i for i in range(data_train.shape[-1])]
        self.fc_struc = self.get_fc_graph_struc()
        self.fc_edge_index = self.build_loc_net()
        self.fc_edge_index = torch.tensor(self.fc_edge_index, dtype=torch.float)

        # Process test data
        test_cp_file = './data/SMD/cp_list_all_SMD_test_improved.npy' if self.N == 'SN' else None
        self.test = self.process_data(data_test, is_train=False, cp_file=test_cp_file)
        self.test_labels = data_test_label

class Dataset_MSL(BaseDataset):
    def __init__(self, flag='train', input_len=0, data_path='./data/MSL', data_process=True, N='LIN',
                 partial_train=False, ratio=0.2):
        super().__init__(flag, input_len, data_path, data_process, N, partial_train, ratio)
        self.__read_data__()

    def __read_data__(self):
        data_train = np.load(self.data_path + "/MSL_train.npy")
        data_test = np.load(self.data_path + "/MSL_test.npy")
        data_test_label = np.load(self.data_path + "/MSL_test_label.npy")
        
        if self.data_process:
            data_train = np.concatenate([data_train[:, :1], data_train[:, [2, 3, 5, 6, 7, 9, 11, 12, 13, 14, 15, 16,
                                                                      17, 19, 20, 23, 27, 28, 29, 31, 33, 35, 39,
                                                                      41, 42, 43, 45, 46, 47, 48, 49, 53, 54]]],
                                    axis=1)
            data_test = np.concatenate([data_test[:, :1], data_test[:, [2, 3, 5, 6, 7, 9, 11, 12, 13, 14, 15, 16,
                                                                   17, 19, 20, 23, 27, 28, 29, 31, 33, 35, 39,
                                                                   41, 42, 43, 45, 46, 47, 48, 49, 53, 54]]],
                                   axis=1)
        
        # Process training data
        train_cp_file = './data/MSL/cp_list_all_MSL_train.npy' if self.N == 'SN' else None
        data_train = self.process_data(data_train, is_train=True, cp_file=train_cp_file)
        
        # Split training and validation sets
        data_len = data_train.shape[0]
        self.train = data_train[:int(data_len * 0.8 * self.ratio), :]
        self.val = data_train[int(data_len * 0.8):, :]
        
        # Feature map and edge indices
        self.feature_map = [i for i in range(data_train.shape[-1])]
        self.fc_struc = self.get_fc_graph_struc()
        self.fc_edge_index = self.build_loc_net()
        self.fc_edge_index = torch.tensor(self.fc_edge_index, dtype=torch.float)

        # Process test data
        test_cp_file = './data/MSL/cp_list_all_MSL_test.npy' if self.N == 'SN' else None
        self.test = self.process_data(data_test, is_train=False, cp_file=test_cp_file)
        self.test_labels = data_test_label


class Dataset_SMAP(BaseDataset):
    def __init__(self, flag='train', input_len=0, data_path='./data/SMAP', data_process=True, N='LIN',
                 partial_train=False, ratio=0.2):
        super().__init__(flag, input_len, data_path, data_process, N, partial_train, ratio)
        self.__read_data__()

    def __read_data__(self):
        # Load data
        data_train = np.load(self.data_path + "/SMAP_train.npy")
        data_test = np.load(self.data_path + "/SMAP_test.npy")
        data_test_label = np.load(self.data_path + "/SMAP_test_label.npy")

        # Data preprocessing
        if self.data_process:
            data_train = np.delete(data_train, 16, 1)
            data_test = np.delete(data_test, 16, 1)
        
        # Process training data
        train_cp_file = f'./data/SMAP/cp_list_all_SMAP_train.npy' if self.N == 'SN' else None
        data_train = self.process_data(data_train, is_train=True, cp_file=train_cp_file)
        
        # Split training and validation sets
        data_len = data_train.shape[0]
        self.train = data_train[:int(data_len * 0.8 * self.ratio), :]
        self.val = data_train[int(data_len * 0.8):, :]
        
        # Feature map and edge indices
        self.feature_map = [i for i in range(data_train.shape[-1])]
        self.fc_struc = self.get_fc_graph_struc()
        self.fc_edge_index = self.build_loc_net()
        self.fc_edge_index = torch.tensor(self.fc_edge_index, dtype=torch.float)

        # Process test data
        test_cp_file = f'./data/SMAP/cp_list_all_SMAP_test.npy' if self.N == 'SN' else None
        self.test = self.process_data(data_test, is_train=False, cp_file=test_cp_file)
        self.test_labels = data_test_label


class Dataset_PSM(BaseDataset):
    def __init__(self, flag='train', input_len=0, data_path='./data/PSM', data_process=True, N='LIN',
                 partial_train=False, ratio=0.2):
        super().__init__(flag, input_len, data_path, data_process, N, partial_train, ratio)
        self.__read_data__()

    def __read_data__(self):
        # Load data
        data_train = pd.read_csv(self.data_path + '/train.csv').values[:, 1:]
        data_test = pd.read_csv(self.data_path + '/test.csv').values[:, 1:]
        data_test_label = pd.read_csv(self.data_path + '/test_label.csv').values[:, 1:]

        # Process training data
        train_cp_file = f'./data/PSM/cp_list_all_PSM_train.npy' if self.N == 'SN' else None
        data_train = self.process_data(data_train, is_train=True, cp_file=train_cp_file)
        
        # Split training and validation sets
        data_len = data_train.shape[0]
        self.train = data_train[:int(data_len * 0.8 * self.ratio), :]
        self.val = data_train[int(data_len * 0.8):, :]
        
        # Feature map and edge indices
        self.feature_map = [i for i in range(data_train.shape[-1])]
        self.fc_struc = self.get_fc_graph_struc()
        self.fc_edge_index = self.build_loc_net()
        self.fc_edge_index = torch.tensor(self.fc_edge_index, dtype=torch.float)

        # Process test data
        test_cp_file = f'./data/PSM/cp_list_all_PSM_test.npy' if self.N == 'SN' else None
        self.test = self.process_data(data_test, is_train=False, cp_file=test_cp_file)
        self.test_labels = data_test_label


class Dataset_SWaT(BaseDataset):
    def __init__(self, flag='train', input_len=0, data_path='./data/SWaT', data_process=True, N='LIN',
                 partial_train=False, ratio=0.2):
        super().__init__(flag, input_len, data_path, data_process, N, partial_train, ratio)
        self.__read_data__()

    def __read_data__(self):
        # Load data
        data_train = pd.read_csv(self.data_path + '/Normal.csv')
        data_test = pd.read_csv(self.data_path + '/Attack.csv')
        data_test_label = pd.read_csv(self.data_path + '/Attack.csv')

        # Data preprocessing
        if self.data_process:
            # Process column indices
            int_index = []
            for i in range(data_train.columns.shape[0]):
                int_index.append(str(int(float(data_train.columns[i]))))
            data_train.columns = int_index
            
            int_index.append(str(int(float(data_test.columns[-1]))))
            data_test.columns = int_index
            
            # Select specific columns
            cols_dig = ['3', '4', '10', '13', '15', '20', '21', '22', '23', '24', '25', '31', '34', '43', '50']
            cols_normal = ['1', '2', '6', '7', '8', '9', '17', '18', '19', '26', '27', '28', '29', '35', 
                          '36', '37', '38', '39', '40', '41', '42', '45', '46', '47', '48']
            
            data_train = data_train[cols_normal + cols_dig].values
            data_test = data_test[cols_normal + cols_dig].values
        else:
            data_train = data_train.values
            data_test = data_test.values[:, :-1]
        
        # Get test labels
        data_test_label = data_test_label.values[:, -1:]
        
        # Process training data
        train_cp_file = './data/SWaT/cp_list_all_SWaT_train.npy' if self.N == 'SN' else None
        data_train = self.process_data(data_train, is_train=True, cp_file=train_cp_file)
        
        # Split training and validation sets
        data_len = data_train.shape[0]
        self.train = data_train[:int(data_len * 0.8 * self.ratio), :]
        self.val = data_train[int(data_len * 0.8):, :]
        
        # Feature map and edge indices
        self.feature_map = [i for i in range(data_train.shape[-1])]
        self.fc_struc = self.get_fc_graph_struc()
        self.fc_edge_index = self.build_loc_net()
        self.fc_edge_index = torch.tensor(self.fc_edge_index, dtype=torch.float)
        
        # Process test data
        test_cp_file = './data/SWaT/cp_list_all_SWaT_test.npy' if self.N == 'SN' else None
        self.test = self.process_data(data_test, is_train=False, cp_file=test_cp_file)
        self.test_labels = data_test_label


class Dataset_WADI(BaseDataset):
    def __init__(self, flag='train', input_len=0, data_path='./data/WADI', data_process=True, N='LIN',
                 partial_train=False, ratio=0.2):
        super().__init__(flag, input_len, data_path, data_process, N, partial_train, ratio)
        self.__read_data__()

    def __read_data__(self):
        # Load data
        data_train = pd.read_csv(self.data_path + '/Normal.csv')
        data_test = pd.read_csv(self.data_path + '/Attack.csv')
        data_test_label = pd.read_csv(self.data_path + '/Attack.csv')

        # Data preprocessing
        if self.data_process:
            # Process column indices
            int_index = []
            for i in range(data_train.columns.shape[0]):
                int_index.append(str(int(float(data_train.columns[i]))))
            data_train.columns = int_index
            
            int_index.append(str(int(float(data_test.columns[-1]))))
            data_test.columns = int_index
            
            # Select specific columns
            cols_dig = ['10', '13', '14', '16', '18', '48', '49', '50', '51', '52', '53', '54', '55', '56', 
                        '57', '58', '59', '71', '74', '76', '77', '78', '79', '80', '81', '83']
            cols_normal = ['1', '2', '3', '4', '5', '6', '9', '20', '21', '22', '23', '24', '25', '26', '27', 
                          '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41', 
                          '42', '43', '44', '45', '46', '47', '60', '61', '63', '64', '65', '66', '67', '68', 
                          '82', '84', '86', '87', '89', '90', '91', '98', '99', '100', '101', '102', '103', 
                          '104', '105', '107', '108', '109', '110', '111', '113', '121', '123']
            
            data_train = data_train[cols_normal + cols_dig].values
            data_test = data_test[cols_normal + cols_dig].values
        else:
            data_train = data_train.values
            data_test = data_test.values[:, :-1]
        
        # Get test labels
        data_test_label = data_test_label.values[:, -1:]
        
        # Process training data
        train_cp_file = './data/WADI/cp_list_all_WADI_train.npy' if self.N == 'SN' else None
        data_train = self.process_data(data_train, is_train=True, cp_file=train_cp_file)
        
        # Split training and validation sets
        data_len = data_train.shape[0]
        self.train = data_train[:int(data_len * 0.8 * self.ratio), :]
        self.val = data_train[int(data_len * 0.8):, :]
        
        # Feature map and edge indices
        self.feature_map = [i for i in range(data_train.shape[-1])]
        self.fc_struc = self.get_fc_graph_struc()
        self.fc_edge_index = self.build_loc_net()
        self.fc_edge_index = torch.tensor(self.fc_edge_index, dtype=torch.float)
        
        # Process test data
        test_cp_file = './data/WADI/cp_list_all_WADI_test.npy' if self.N == 'SN' else None
        self.test = self.process_data(data_test, is_train=False, cp_file=test_cp_file)
        self.test_labels = data_test_label


class Dataset_MBA(BaseDataset):
    def __init__(self, flag='train', input_len=0, data_path='./data/MBA', data_process=True, N='LIN',
                 partial_train=False, ratio=0.2):
        super().__init__(flag, input_len, data_path, data_process, N, partial_train, ratio)
        self.__read_data__()

    def __read_data__(self):
        # Load data
        data_train = pd.read_excel(self.data_path + '/train.xlsx')
        data_test = pd.read_excel(self.data_path + '/test.xlsx')
        data_test_label = pd.read_excel(self.data_path + '/labels.xlsx')

        # Preprocess data
        data_train = data_train.values[1:, 1:].astype(float)
        data_test = data_test.values[1:, 1:].astype(float)
        
        # Process training data
        train_cp_file = './data/MBA/cp_list_all_MBA_train.npy' if self.N == 'SN' else None
        data_train = self.process_data(data_train, is_train=True, cp_file=train_cp_file)
        
        # Split training and validation sets
        data_len = data_train.shape[0]
        self.train = data_train[:int(data_len * 0.8 * self.ratio), :]
        self.val = data_train[int(data_len * 0.8):, :]

        # Feature map and edge indices
        self.feature_map = [i for i in range(data_train.shape[-1])]
        self.fc_struc = self.get_fc_graph_struc()
        self.fc_edge_index = self.build_loc_net()
        self.fc_edge_index = torch.tensor(self.fc_edge_index, dtype=torch.float)
        
        # Process test data
        data_test_label = data_test_label.values[:, 1].astype(int)
        test_cp_file = './data/MBA/cp_list_all_MBA_test.npy' if self.N == 'SN' else None
        self.test = self.process_data(data_test, is_train=False, cp_file=test_cp_file)
        
        # Special handling for test labels: MBA dataset has a window around labels
        labels = np.zeros_like(self.test)
        for i in range(-20, 20):
            # Handle index out of bounds cases
            valid_indices = np.clip(data_test_label + i, 0, labels.shape[0] - 1)
            labels[valid_indices, :] = 1
        self.test_labels = labels[:, :1]


class Dataset_NAB(BaseDataset):
    def __init__(self, flag='train', input_len=0, data_path='./data/NAB', data_process=True, N='LIN',
                 partial_train=False, ratio=0.2):
        super().__init__(flag, input_len, data_path, data_process, N, partial_train, ratio)
        self.__read_data__()

    def __read_data__(self):
        # Load data
        data = pd.read_csv(self.data_path + '/ec2_request_latency_system_failure.csv')
        data_train = data[['value']].values
        data_test = data[['value']].values
        
        # Load labels from JSON file
        data_test_label = np.zeros_like(data_test)
        with open(self.data_path + '/labels.json') as f:
            labeldict = json.load(f)
            
        # Mark anomaly points with surrounding window
        for timestamp in labeldict['realKnownCause/ec2_request_latency_system_failure.csv']:
            tstamp = timestamp.replace('.000000', '')
            index = np.where(((data['timestamp'] == tstamp).values + 0) == 1)[0][0]
            # Add context window
            start_idx = max(0, index - 4)
            end_idx = min(data_test_label.shape[0], index + 4)
            data_test_label[start_idx:end_idx] = 1

        # Process training data
        train_cp_file = './data/NAB/cp_list_all_NAB_train.npy' if self.N == 'SN' else None
        data_train = self.process_data(data_train, is_train=True, cp_file=train_cp_file)
        
        # Split training and validation sets
        data_len = data_train.shape[0]
        self.train = data_train[:int(data_len * 0.8 * self.ratio), :]
        self.val = data_train[int(data_len * 0.8):, :]
        
        # Feature map and edge indices
        self.feature_map = [i for i in range(data_train.shape[-1])]
        self.fc_struc = self.get_fc_graph_struc()
        self.fc_edge_index = self.build_loc_net()
        self.fc_edge_index = torch.tensor(self.fc_edge_index, dtype=torch.float)

        # Process test data
        test_cp_file = './data/NAB/cp_list_all_NAB_test.npy' if self.N == 'SN' else None
        self.test = self.process_data(data_test, is_train=False, cp_file=test_cp_file)
        self.test_labels = data_test_label


class Dataset_MSDS(BaseDataset):
    def __init__(self, flag='train', input_len=0, data_path='./data/MSDS', data_process=True, N='LIN',
                 partial_train=False, ratio=0.2):
        super().__init__(flag, input_len, data_path, data_process, N, partial_train, ratio)
        self.__read_data__()

    def __read_data__(self):
        # Load data
        data_train = pd.read_csv(self.data_path + '/train.csv', header=None).values
        data_test = pd.read_csv(self.data_path + '/test.csv', header=None).values
        data_test_label = pd.read_csv(self.data_path + '/labels.csv', header=None).values

        # Process training data
        train_cp_file = './data/MSDS/cp_list_all_MSDS_train.npy' if self.N == 'SN' else None
        data_train = self.process_data(data_train, is_train=True, cp_file=train_cp_file)
        
        # Split training and validation sets
        data_len = data_train.shape[0]
        self.train = data_train[:int(data_len * 0.8 * self.ratio), :]
        self.val = data_train[int(data_len * 0.8):, :]
        
        # Feature map and edge indices
        self.feature_map = [i for i in range(data_train.shape[-1])]
        self.fc_struc = self.get_fc_graph_struc()
        self.fc_edge_index = self.build_loc_net()
        self.fc_edge_index = torch.tensor(self.fc_edge_index, dtype=torch.float)

        # Process test data
        test_cp_file = './data/MSDS/cp_list_all_MSDS_test.npy' if self.N == 'SN' else None
        self.test = self.process_data(data_test, is_train=False, cp_file=test_cp_file)
        self.test_labels = data_test_label