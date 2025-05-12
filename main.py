import math
import time
from sklearn.metrics import precision_recall_fscore_support
from models.DLinear.DLinear import DLinear
from models.Autoformer.Autoformer import Autoformer
from models.DeepAR.model import DeepAR
from models.FEDformer.FEDformer import Model as FEDformer
from models.GTA.gta import GTA
from models.Informer.model import Informer
from models.LSTNet.model import LSTNet
from models.MA.MA import MA
from models.RT.model import RF as RTNet
from torch.optim.lr_scheduler import OneCycleLR
import os
import torch
import numpy as np
import argparse
from data.data_loader import *
from utils.tools import adjust_predictions_k
import torch.multiprocessing as mp

def init():
    parser = argparse.ArgumentParser(description='[DRPAD]')
    # Model selection
    parser.add_argument('--model', type=str, required=True,
                        help='Experimental model')

    # Data related
    parser.add_argument('--data', type=str, required=True, help='Data')
    parser.add_argument('--data_path', type=str, default='./data/SMD', help='Root path of data files')

    # General settings 
    parser.add_argument('--input_len', type=int, default=None, help='Input sequence length for the model')
    parser.add_argument('--label_len', type=int, default=None, help='Input sequence length for the model')
    parser.add_argument('--num_max_anomaly', type=int, default=None, help='Maximum number of consecutive anomalies allowed')
    parser.add_argument('--variate', type=int, default=0, help='Number of input variables')
    parser.add_argument('--out_variate', type=int, default=0, help='Number of output variables')
    parser.add_argument('--dropout', type=float, default=None, help='Dropout')
    parser.add_argument('--anomaly_ratio', type=float, default=None, help='Anomaly ratio')
    parser.add_argument('--itr', type=int, default=5, help='Number of experiments')
    parser.add_argument('--train_epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training input data')
    parser.add_argument('--patience', type=int, default=3, help='Early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Optimizer learning rate')
    parser.add_argument('--loss', type=str, default='mse', help='Loss function')
    parser.add_argument('--sigmod', type=int, default=6, help='Sigmoid coefficient')


    # AFMF
    parser.add_argument('--data_process', action='store_true',
                        help='Whether to preprocess data'
                        , default=True)
    parser.add_argument('--drop', type=int, default=0, help='Loop variable k')
    parser.add_argument('--thresh', type=float, default=3, help='Decrease ratio')

    # RTNet
    parser.add_argument('--d_model', type=int, default=128, help='Model dimension')
    parser.add_argument('--pyramid', type=int, default=1)
    parser.add_argument('--kernel', type=int, default=3)
    parser.add_argument('--block_nums', type=int, default=3)

    # Autoformer (Shared by GTA and Informer)
    parser.add_argument('--moving_avg', default=[24], help='Window size for moving average')
    parser.add_argument('--e_layers', type=int, default=2, help='Number of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='Number of decoder layers')
    parser.add_argument('--n_heads', type=int, default=8, help='')
    parser.add_argument('--d_ff', type=int, default=2048, help='Model dimension')
    parser.add_argument('--activation', type=str, default='gelu', help='Activation function')
    parser.add_argument('--factor', type=int, default=1, help='Attention factor')

    # DeepAR
    parser.add_argument('--num_layers', type=int, default=3)

    # GTA
    parser.add_argument('--num_levels', type=int, default=3, help='Number of expansion levels for graph embedding')

    # Informer
    parser.add_argument('--attn', type=str, default='prob', help='Activation function')
    parser.add_argument('--distil', action='store_true',
                        help='Whether to use distillation operation',
                        default=True)
    parser.add_argument('--mix', action='store_true',
                        help='Whether to mix after attention',
                        default=True)

    # LSTNet
    parser.add_argument('--RNN_hid_size', default=512, type=int, help='Hidden channels for RNN module')
    parser.add_argument('--CNN_hid_size', type=int, default=100, help='Number of CNN hidden units')
    parser.add_argument('--CNN_kernel', type=int, default=6, help='Kernel size for CNN layers')
    parser.add_argument('--highway_window', type=int, default=0, help='Window size for highway component')
    parser.add_argument('--skip', type=int, default=24)
    parser.add_argument('--hidSkip', type=int, default=5)

    # Experiment settings
    parser.add_argument('--gpu', type=int, default=0, help='GPU')
    parser.add_argument('--save', action='store_true',
                        help='Whether to save prediction results',
                        default=False)
    parser.add_argument('--reproducible', action='store_true',
                        help='Whether to make results reproducible'
                        , default=False)
    parser.add_argument('--seed_list', type=str, default=None, 
                    help='Comma-separated list of random seeds, e.g.: "42,123,456,789,1024"')

    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() else False


    return args


def get_model(model_name, args, device, dataset_name=None, mode=None):

    LIN = False

    print('input_len:', args.input_len)


    model_constructors = {
        'Autoformer': lambda: Autoformer(
            args.variate,
            args.out_variate,
            args.input_len,
            args.label_len,
            args.moving_avg,
            512,
            args.dropout,
            1,  #  factor=1
            args.n_heads,
            args.activation,
            2,  #  e_layers=2
            1,  #  d_layers=1
            LIN
        ),
        
        'RTNet': lambda: RTNet(
            args.variate,
            args.out_variate,
            args.input_len,
            args.kernel,
            args.block_nums,
            args.d_model,
            args.pyramid,
            LIN,
            args.dropout
        ),
        
        'DeepAR': lambda: DeepAR(
            args.variate,
            args.out_variate,
            args.input_len,
            args.d_model,
            args.num_layers,
            LIN
        ),
        
        'Informer': lambda: Informer(
            args.variate,
            args.variate,
            args.out_variate,
            args.input_len,
            args.label_len,
            args.factor,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.dropout,
            args.attn,
            args.activation,
            args.distil,
            args.mix,
            LIN
        ),
        
        'LSTNet': lambda: LSTNet(
            args.input_len,
            args.variate,
            args.RNN_hid_size,
            args.CNN_hid_size,
            args.hidSkip,
            args.skip,
            args.CNN_kernel,
            args.highway_window,
            args.dropout,
            args.out_variate,
            LIN
        ),
        
        'GTA': lambda: GTA(
            args.variate,
            args.out_variate,
            args.input_len,
            args.label_len,
            args.num_levels,
            args.factor,
            512,  #  d_model=512
            args.n_heads,
            2,  #  e_layers=2
            1,  #  d_layers=1
            0.05,  #  dropout=0.05
            args.activation,
            LIN,
            device
        ),
        
        'MA': lambda: MA(
            args.out_variate,
            args.input_len,
            LIN,
        ),
        
        'DLinear': lambda: DLinear(
            args.variate,
            args.out_variate,
            args.input_len,
            args.kernel,
            LIN
        ),
        
        'FEDformer': lambda: create_fedformer(args, LIN)
    }
    
    if model_name not in model_constructors:
        raise ValueError(f"Unknown model: {model_name}")

    model = model_constructors[model_name]()
    return model.float().to(device)


def create_fedformer(args, LIN):
    class Configs(object):
        def __init__(self):
            self.output_v = args.out_variate
            self.modes = 32
            self.mode_select = 'low'
            self.version = 'Wavelets'
            self.moving_avg = [24]
            self.L = 1
            self.features = 'M'
            self.base = 'legendre'
            self.cross_activation = 'softmax'
            self.seq_len = args.input_len
            self.label_len = args.label_len
            self.pred_len = 1
            self.output_attention = False
            self.d_model = 32
            self.dropout = args.dropout
            self.factor = 1
            self.n_heads = 8
            self.d_ff = 512
            self.embed = 'timeF'
            self.freq = 'h'
            self.e_layers = 2
            self.d_layers = 1
            self.activation = 'gelu'
            self.LIN = LIN

    configs = Configs()
    return FEDformer(configs)

def init_and_train_model(model_name, dataset_name, mode, device, current_seed):

    print('model:', model_name)
    
    if mode=='AFMF':
        bs = args.batch_size
    else:
            bs = 128
    print('bs_training:', bs)
    # Generate filename based on model name
    best_model_filename = f'archive/{model_name}_{dataset_name}_{mode}_best_{current_seed}.ckpt'

    if dataset_name == 'SWaT':
        args.variate = 40
        args.out_variate = 25
    if dataset_name == 'WADI':
        args.variate = 92
        args.out_variate = 66
    if dataset_name == 'SMD':
        args.variate = 37
        args.out_variate = 37
    if dataset_name == 'MSL':
        args.variate = 34
        args.out_variate = 1
    if dataset_name == 'SMAP':
        args.variate = 24
        args.out_variate = 1
    if dataset_name == 'NAB':
        args.variate = 1
        args.out_variate = 1
    if dataset_name == 'MBA':
        args.variate = 2
        args.out_variate = 2
    if dataset_name == 'MSDS':
        args.variate = 2
        args.out_variate = 2
    if dataset_name == 'PSM':
        args.variate = 25
        args.out_variate = 25

    if os.path.exists(best_model_filename):
            model = get_model(model_name, args, device,dataset_name=dataset_name, mode=mode)
            model.load_state_dict(torch.load(best_model_filename, map_location=device), strict=False)
            model.eval()
    else:
            model = get_model(model_name, args, device,dataset_name=dataset_name, mode=mode)
            
            # Get training set and validation set
            train_dataset = get_dataset(dataset_name, mode, 'train')
            val_dataset = get_dataset(dataset_name, mode, 'val')
            
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=bs, shuffle=True)
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=bs, shuffle=False)
    
            print("Data loaded, starting training")
            
            # Define optimizer
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
            # Define learning rate scheduler
            total_steps = len(train_loader) * args.train_epochs
            scheduler = OneCycleLR(
                optimizer,
                max_lr=args.learning_rate * 2,  # Peak learning rate
                total_steps=total_steps,
                pct_start=0.3,                    # First 30% steps for warm-up
                div_factor=10,                    # Initial learning rate = max_lr/div_factor
                final_div_factor=100,             # Final learning rate = max_lr/final_div_factor
                anneal_strategy='cos'             # Cosine annealing strategy
            )
            # Define loss function
            criterion = torch.nn.MSELoss()
            
            # Early stopping parameters
            patience = args.patience  # Get patience value from command line arguments
            best_val_loss = float('inf')
            counter = 0
            early_stopped = False
            
            # Train model
            epoches = args.train_epochs
            for epoch in range(epoches):
                # Training phase
                model.train()
                epoch_loss = 0.0
                batch_count = 0
                for i, (batch_x) in enumerate(train_loader):
                    batch_x = batch_x.to(device).float()
                    # Forward propagation
                    output, gt = model(batch_x)
                    # Calculate loss
                    loss = criterion(output, gt)
                    epoch_loss += loss.item()
                    batch_count += 1

                    # Add NaN detection and handling in training loop
                    if torch.isnan(loss):
                        print(f"Warning: NaN detected in epoch {epoch+1}, step {i+1}")
                        # Reduce learning rate for current batch
                        for param_group in optimizer.param_groups:
                            param_group['lr'] *= 0.5
                        # Skip current batch
                        continue
                    
                    # Backward propagation
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                    optimizer.step()
                    scheduler.step()
                    
                    # Print loss
                    if (i + 1) % 1000 == 0:
                        print(f"Epoch [{epoch + 1}/{epoches}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.6f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
                    torch.cuda.empty_cache()
                
                # Calculate average training loss
                avg_train_loss = epoch_loss / batch_count
                print(f"Epoch [{epoch + 1}/{epoches}] completed, Avg Train Loss: {avg_train_loss:.6f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
                
                # Validation phase
                model.eval()
                val_loss = 0.0
                val_count = 0
                with torch.no_grad():
                    for batch_x in val_loader:
                        batch_x = batch_x.to(device).float()
                        output, gt = model(batch_x)
                        loss = criterion(output, gt)
                        val_loss += loss.item()
                        val_count += 1
                
                # Calculate average validation loss
                avg_val_loss = val_loss / val_count if val_count > 0 else float('inf')
                print(f"Epoch [{epoch + 1}/{epoches}], Validation Loss: {avg_val_loss:.6f}")
                
                # Early stopping check
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    counter = 0
                    # Save best model
                    os.makedirs(os.path.dirname(best_model_filename), exist_ok=True)
                    torch.save(model.state_dict(), best_model_filename)
                    print(f"Best model updated, validation loss: {best_val_loss:.6f}")
                else:
                    counter += 1
                    print(f"Validation loss did not improve, counter: {counter}/{patience}")
                    if counter >= patience:
                        print(f"Early stopping triggered! Training stopped at epoch {epoch+1}")
                        early_stopped = True
                        break
            
            # After training, load best model if early stopping was triggered
            if early_stopped and os.path.exists(best_model_filename):
                print("Loading best model...")
                model.load_state_dict(torch.load(best_model_filename))
            model.eval()
    return model


def get_dataset_config(dataset_name):

    configs = {
        'NAB': {
            'input_len': 360, 
            'label_len': 180, 
            'dropout': 0.05, 
            'anomaly_ratio': 0.5, 
            'num_max_anomaly': 20
        },
        'MSDS': {
            'input_len': 720, 
            'label_len': 360, 
            'dropout': 0.05, 
            'anomaly_ratio': 2.5, 
            'num_max_anomaly': 50
        },
        'MSL': {
            'input_len': 48, 
            'label_len': 24, 
            'dropout': 0.1, 
            'anomaly_ratio': 1.5, 
            'num_max_anomaly': 30
        },
        'PSM': {
            'input_len': 720, 
            'label_len': 360, 
            'dropout': 0.1, 
            'anomaly_ratio': 1.5, 
            'num_max_anomaly': 50
        },
        'MBA': {
            'input_len': 100, 
            'label_len': 50, 
            'dropout': 0.1, 
            'anomaly_ratio': 2.0, 
            'num_max_anomaly': 5
        },
        'SMD': {
            'input_len': 720, 
            'label_len': 360, 
            'dropout': 0.1, 
            'anomaly_ratio': 0.5, 
            'num_max_anomaly': 100
        },
        'SMAP': {
            'input_len': 24, 
            'label_len': 12, 
            'dropout': 0.1, 
            'anomaly_ratio': 1.5, 
            'num_max_anomaly': 20
        },
        'SWaT': {
            'input_len': 720, 
            'label_len': 360, 
            'dropout': 0.1, 
            'anomaly_ratio': 1.0, 
            'num_max_anomaly': 100
        },
        'WADI': {
            'input_len': 100, 
            'label_len': 50, 
            'dropout': 0.1, 
            'anomaly_ratio': 0.5, 
            'num_max_anomaly': 100
        }
    }
    
    if dataset_name not in configs:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return configs[dataset_name]


def get_dataset(dataset_name, mode, flag, graph=False):
    if mode == 'AFMF':
        mode = 'LIN'
    if dataset_name == 'SMD':
        dataset = Dataset_SMD(input_len=args.input_len, N=mode, flag=flag, data_path='data/SMD')
    elif dataset_name == 'SWaT':
        dataset = Dataset_SWaT(input_len=args.input_len, N=mode, flag=flag, data_path='data/SWaT')
    elif dataset_name == 'WADI':
        dataset = Dataset_WADI(input_len=args.input_len, N=mode, flag=flag, data_path='data/WADI')
    elif dataset_name == 'MSL':
        dataset = Dataset_MSL(input_len=args.input_len, N=mode, flag=flag, data_path='data/MSL')
    elif dataset_name == 'SMAP':
        dataset = Dataset_SMAP(input_len=args.input_len, N=mode, flag=flag, data_path='data/SMAP')
    elif dataset_name == 'NAB':
        dataset = Dataset_NAB(input_len=args.input_len, N=mode, flag=flag, data_path='data/NAB')
    elif dataset_name == 'MBA':
        dataset = Dataset_MBA(input_len=args.input_len, N=mode, flag=flag, data_path='data/MBA')
    elif dataset_name == 'MSDS':
        dataset = Dataset_MSDS(input_len=args.input_len, N=mode, flag=flag, data_path='data/MSDS')
    elif dataset_name == 'PSM':
        dataset = Dataset_PSM(input_len=args.input_len, N=mode, flag=flag, data_path='data/PSM')    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    if graph:
        edge_index_sets = dataset.build_loc_net()
        return edge_index_sets
    else:
        return dataset



def predict_normal(model, dataset_name, mode, device, model_name, mse_thresh=None, save_mses=False, 
                  proporation_mean=0.5, proporation_pre_dim=0.5, deal='mean', current_seed=0):
    """
    Use the model to predict normal data and calculate anomaly detection thresholds
    
    Args:
        model: The trained model
        dataset_name: Dataset name
        mode: Running mode
        device: Running device
        model_name: Model name
        mse_thresh: MSE threshold, if None it will be automatically calculated
        save_mses: Whether to save MSE results
        proporation_mean: Proportion used to calculate average threshold
        proporation_pre_dim: Proportion used to calculate per-dimension threshold
        deal: Anomaly judgment method, can be 'mean' or 'add'
        current_seed: Current random seed
    
    Returns:
        Different thresholds according to the deal parameter
    """
    model.eval()
    bs = 1024 
    if model_name in ['GTA', 'Autoformer']:
        bs = 128
    elif model_name == 'FEDformer':
        bs = 300 if dataset_name in ['SMD', 'SWaT'] else 1024

    # Load test data
    dataset = get_dataset(dataset_name, mode, 'test')
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=bs, shuffle=False, 
        num_workers=1, pin_memory=True, 
        prefetch_factor=2, persistent_workers=False
    )
    test_label = dataset.get_label()
    test_data = dataset.get_test()

    # Prediction data file path
    file_path = f"archive/{model_name}_{dataset_name}_{mode}_normal_predictions_{current_seed}.pt"
    
    # Check if prediction results are already saved
    if os.path.exists(file_path):
        print("Loading existing prediction data")
        saved_data = torch.load(file_path)
        mse_list = saved_data['mse_list']
        pre = saved_data['pre']
        gt_list = saved_data['gt_list']
    else:
        print("Starting prediction")
        pre = []
        gt_list = []
        mse_list = []
        mse_list_gpu = []
        pre_gpu = []
        gt_list_gpu = []
        
        # Batch processing predictions
        with torch.no_grad():
            for i, (batch_x) in enumerate(data_loader):
                batch_x = batch_x.to(device).float()
                output, gt = model(batch_x)
                mse = (output - gt).pow(2)
                
                # Collect results
                mse_list_gpu.append(mse)
                pre_gpu.append(output)
                gt_list_gpu.append(gt)
                
                # Clear memory
                torch.cuda.empty_cache()
                del batch_x, output, gt
                torch.cuda.empty_cache()
                
                # Periodically transfer GPU data to CPU to reduce GPU memory usage
                if len(mse_list_gpu) >= 64:
                    mse_list.append(torch.cat(mse_list_gpu, dim=0).cpu())
                    pre.append(torch.cat(pre_gpu).cpu())
                    gt_list.append(torch.cat(gt_list_gpu).cpu())
                    # Empty GPU lists
                    mse_list_gpu = []
                    pre_gpu = []
                    gt_list_gpu = []
                    torch.cuda.empty_cache()
        
            # Process remaining batches
            if mse_list_gpu:
                mse_list.append(torch.cat(mse_list_gpu, dim=0).cpu())
                pre.append(torch.cat(pre_gpu).cpu())
                gt_list.append(torch.cat(gt_list_gpu).cpu())
            
            # Merge all batch data
            mse_list = torch.cat(mse_list, dim=0).numpy()
            pre = torch.cat(pre, dim=0).numpy()
            gt_list = torch.cat(gt_list, dim=0).numpy()
            
            # Save prediction results
            if args.save and mode == 'normal':
                torch.save({'mse_list': mse_list, 'pre': pre, 'gt_list': gt_list}, file_path)
                print("Prediction data has been saved")

    # Ensure correct data shape
    mse_list = mse_list.reshape(-1, args.out_variate)

    # Calculate statistics for each dimension
    mse_mean_per_dim = np.mean(mse_list, axis=0)
    mse_std_per_dim = np.std(mse_list, axis=0)
    
    # Calculate thresholds based on sigma rules
    mse_thresh_per_dim_3_sigma = mse_mean_per_dim + 3 * mse_std_per_dim
    mse_thresh_per_dim_5_sigma = mse_mean_per_dim + args.sigmod * mse_std_per_dim
    
    # Calculate anomalies based on different processing modes
    if deal == 'mean':
        # Calculate average MSE for each sample
        mse_mean = np.mean(mse_list, axis=1)
        
        # Output model and dataset information
        print(f'**model_name**: {model_name}')
        print(f'**dataset_name**: {dataset_name}')
        print(f'**mode**: {mode}')
        print(f'**proportion**: {proporation_mean}')
        
        # Calculate threshold
        if mse_thresh is None:
            mse_thresh = np.percentile(mse_mean, 100 - proporation_mean)
        
        # Mark anomalies
        anomaly = (mse_mean > mse_thresh).astype(int)
        print('Mode: mean')
    
    elif deal == 'add':
        # Calculate average MSE for samples
        mse_mean = np.mean(mse_list, axis=1)
        
        # Calculate threshold
        if mse_thresh is None:
            mse_thresh = np.percentile(mse_mean, 100 - proporation_mean)
        
        # Detect anomalies in each dimension
        anomaly_per_dim_pro = (mse_list > mse_thresh_per_dim_5_sigma).astype(int)
        anomaly_pro_count = np.sum(anomaly_per_dim_pro, axis=1)
        
        # Combine global and local anomaly judgments
        anomaly = np.where((mse_mean > mse_thresh) | (anomaly_pro_count >= 1), 1, 0)
        print('Mode: add')
    

    # Calculate true labels
    former_len = args.input_len - 1
    label_anomaly = test_label[former_len:former_len + anomaly.shape[0]].astype(int)
    
    # Save labels for later use
    np.save(f'label_anomaly_{dataset_name}.npy', label_anomaly)
    
    # Return thresholds
    if deal == 'mean':
        return mse_thresh, mse_thresh_per_dim_3_sigma, mse_thresh_per_dim_5_sigma



def predict_replacement_add(model, dataset_name, mode, device, model_name,mse_thresh,proporation_mean,mse_thresh_per_dim_3_sigma ,mse_thresh_per_dim_5_sigma,num_max_anomaly=100,current_seed=0):

    model.eval()

    dataset = get_dataset(dataset_name, mode, 'test')
    test_label = dataset.get_label()#(Total_length, 1)
    test_data = dataset.get_test()#(Total_length, variate)
    Total_length = test_data.shape[0]  # Total_length: About 700,000

    
    # Generate file name based on model and dataset
    file_path = f"archive/{model_name}_{dataset_name}_{mode}_{proporation_mean}_replacement_add_predictions_{current_seed}.pt"
    
    if os.path.exists(file_path):
        print("Loading existing prediction data")
        saved_data = torch.load(file_path)
        pred_list = saved_data['pred_list']# (Total_length, variate)
        anomaly_list = saved_data['anomaly_list']# (Total_length, variate)
        mse_mean_list = saved_data['mse_mean_list']# (Total_length, variate)
        mse_list = saved_data['mse_list']
        test_data_replaced_by_pred = saved_data['test_data_replaced_by_pred']# (Total_length, variate)
    else:
        test_data_replaced_by_pred = test_data.copy()  # (Total_length, variate)
        group_size = int(max(1, math.sqrt(Total_length / (1.2 * args.input_len)))) #number of data segments
        numbers_group = Total_length // group_size  # numbers_group: 1416
        len_rest = Total_length % group_size  # len_rest: 420
        pred_list = np.zeros((Total_length, args.out_variate))  # (Total_length, out_variate)
        anomaly_list = np.zeros(Total_length)
        n_continuous_anomaly = np.zeros(group_size)
        mse_mean_list = np.zeros(Total_length)
        mse_list = np.zeros((Total_length, args.out_variate))  # (Total_length, out_variate)
    
        print("Starting prediction")
        with torch.no_grad():
            for j in range(numbers_group - (args.input_len - 1)):
                batch_x = []
    
                for i in range(group_size):
                    batch_x.append(test_data_replaced_by_pred[numbers_group * i + j: numbers_group * i + j + args.input_len,:])
                    if not np.array_equal(test_data_replaced_by_pred[numbers_group * i + j + args.input_len - 1], test_data[numbers_group * i + j + args.input_len - 1]):
                        print('error')
    
                # Convert list to numpy array
                batch_x = np.array(batch_x)  # batch_x.shape=(500,720,variate)
    
                # Convert numpy array to torch.Tensor
                batch_x = torch.tensor(batch_x, dtype=torch.float32).to(device)  # batch_x.shape=(500,720,37)
    
                output, gt = model(batch_x)  # output.shape=(500,1,out_variate), gt.shape=(500,1,out_variate)
                mse = (output - gt).pow(2)  # mse.shape=(500,out_variate)
                mse_mean = mse.mean(-1)  # mse.shape=(500,1)
    
                # Replace values in test_data_replaced_by_pred with prediction values where mse is greater than mse_thresh
                for loc, pred in enumerate(output.cpu().numpy()):  # pred.shape=(1,out_variate), loc=0,1,2,...,499
                    pred_list[numbers_group * loc + j + args.input_len - 1] = pred[0]
                    mse_mean_cpu = mse_mean.cpu().numpy()[loc][0]  # mse_mean_cpu.shape=(1,)
                    mse_cpu = mse.cpu().numpy()[loc]  # mse.shape=(out_variate,)
                    mse_mean_list[numbers_group * loc + j + args.input_len - 1] = mse_mean_cpu
                    mse_list[numbers_group * loc + j + args.input_len - 1] = mse_cpu

                    anomaly_per_dim_pro = (mse_cpu > mse_thresh_per_dim_5_sigma).astype(int)
                    anomaly_pro_count = np.sum(anomaly_per_dim_pro)

                    if (mse_mean[loc].item() > mse_thresh)  or (anomaly_pro_count >= 1):
                        n_continuous_anomaly[loc] += 1
                        if n_continuous_anomaly[loc] >= num_max_anomaly:
                            None
                        else:
                            test_data_replaced_by_pred[numbers_group * loc + j + args.input_len - 1,:args.out_variate] = pred[0]
                            anomaly_list[numbers_group * loc + j + args.input_len - 1] = 1
                    else:
                        n_continuous_anomaly[loc] = 0
    
            print('Start predicting middle section')
            # Middle section
            for i in range(1, group_size):  # i=1,2,...,499
                interval = i * numbers_group  # interval=1416,2832,...,706584
                continuous_none_anomaly = 0  # Number of consecutive non-anomalous points
                n_anomaly = 0  # Number of anomalies within length 720
                n_continuous_anomaly = 0  # Number of consecutive anomalies
                for j in range(numbers_group):
                    data = test_data_replaced_by_pred[j + interval - (args.input_len - 1):j + interval + 1]
                    # Replace the last value with actual value
                    data[-1] = test_data[j + interval]  # data.shape=(720,variate),test_data.shape=(Total_length,variate)
    
                    # Convert numpy array to torch.Tensor
                    data_tensor = torch.tensor(data, dtype=torch.float32).reshape(1, args.input_len, -1).to(device)
    
                    output, gt = model(data_tensor)  # output.shape=(1,1,out_variate), gt.shape=(1,1,37)
                    mse = (output - gt).pow(2)  # mse.shape=(1,out_variate)
                    mse_mean = mse.mean(-1)  # mse.shape=(1,1)
    
                    mse_mean_cpu = mse_mean.cpu().numpy()[0][0]  # mse_mean_cpu.shape=(1,)
                    mse_cpu = mse.cpu().numpy()[0]  # mse.shape=(out_variate,)
                    mse_mean_list[j + interval] = mse_mean_cpu  # (Total_length,1)
                    mse_list[j + interval] = mse_cpu  # (Total_length,out_variate)
    
                    output_cpu = output.cpu().numpy()[0][0]
    
                    pred_list[j + interval] = output_cpu

                    anomaly_per_dim_pro = (mse_cpu > mse_thresh_per_dim_5_sigma).astype(int)
                    anomaly_pro_count = np.sum(anomaly_per_dim_pro)
    
                    if (mse_mean[0][0].item() > mse_thresh)  or (anomaly_pro_count >= 1):
                        n_anomaly += 1
                        n_continuous_anomaly += 1
                        continuous_none_anomaly = 0
                        if n_continuous_anomaly >= num_max_anomaly:
                            # To be modified, whether to mask previous points as 0 for re-prediction is to be discussed
                            None
                        else:
                            test_data_replaced_by_pred[j + interval,:args.out_variate] = output_cpu
                            anomaly_list[j + interval] = 1
                    else:
                        anomaly_list[j + interval] = 0
                        continuous_none_anomaly += 1
                        n_continuous_anomaly = 0
    
                    if j >= args.input_len - 1:
    
                        if continuous_none_anomaly >= args.input_len / 6:
                            print(f"Step [{j + 1}/{numbers_group}]")
                            break
                        if continuous_none_anomaly >= args.input_len / 8:
                            if n_anomaly <= args.input_len / 20:
                                print(f"Step [{j + 1}/{numbers_group}]")
                                break
                        n_anomaly -= 1
    
            print('Start predicting tail section')
            # Tail section processing
            for i in range(len_rest):
                n_continuous_anomaly = 0  # Number of consecutive anomalies
                interval = group_size * numbers_group
                data = test_data_replaced_by_pred[i + interval - (args.input_len - 1):i + interval + 1]
                # Replace the last value with actual value
                data[-1] = test_data[i + interval] 
    
                # Convert numpy array to torch.Tensor
                data_tensor = torch.tensor(data, dtype=torch.float32).reshape(1, args.input_len, -1).to(device)
    
                output, gt = model(data_tensor) 
                mse = (output - gt).pow(2)  
                mse_mean = mse.mean(-1)  
    
                mse_mean_cpu = mse_mean.cpu().numpy()[0][0]  # mse_mean_cpu.shape=(1,)
                mse_cpu = mse.cpu().numpy()[0]  # mse.shape=(out_variate,)
                mse_mean_list[i + interval] = mse_mean_cpu  # (Total_length,1)
                mse_list[i + interval] = mse_cpu  # (Total_length,out_variate)
    
                output_cpu = output.cpu().numpy()[0][0]
    
                pred_list[i + interval] = output_cpu

                anomaly_per_dim_pro = (mse_cpu > mse_thresh_per_dim_5_sigma).astype(int)
                anomaly_pro_count = np.sum(anomaly_per_dim_pro)
    
                if (mse_mean[0][0].item() > mse_thresh)  or (anomaly_pro_count >= 1):
                    n_continuous_anomaly += 1
                    if n_continuous_anomaly >= num_max_anomaly:
                        None
                    else:
                        test_data_replaced_by_pred[i + interval,:args.out_variate] = output_cpu
                        anomaly_list[i + interval] = 1

    
        
        if args.save==True and mode=='SN':
        # Save prediction data
            torch.save({
                'pred_list': pred_list,# (Total_length, out_variate)
                'anomaly_list': anomaly_list,# (Total_length,)
                'mse_mean_list': mse_mean_list,# (Total_length,)
                'mse_list': mse_list,# (Total_length, out_variate)
                'test_data_replaced_by_pred': test_data_replaced_by_pred# (Total_length, variate)
            }, file_path)
            print("Prediction data has been saved")
    

    anomaly = anomaly_list[args.input_len - 1:]  # (Total_length-(args.input_len - 1),)
    
    former_len = args.input_len - 1
    label_anomaly = test_label[former_len:former_len + anomaly.shape[0]].astype(int)
    anomaly_origin = anomaly.copy()
    mse_list = mse_list[args.input_len - 1:]  # (Total_length-(args.input_len - 1), out_variate)

    precision, recall, f_score, support = \
        precision_recall_fscore_support(label_anomaly, anomaly_origin, average='binary')
    print('Without detection adjustment:',
        "Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f}".format(
            precision, recall, f_score))
    score.append([precision, recall, f_score])
    
    anomaly = adjust_predictions_k(anomaly_origin, label_anomaly, 20)  # Percentage anomaly detection adjustment
    
    precision, recall, f_score, support = \
        precision_recall_fscore_support(label_anomaly, anomaly, average='binary')
    print('20% ratio:',
        "Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f}".format(
            precision, recall, f_score))
    score[-1].extend([precision, recall, f_score])
    
    anomaly = adjust_predictions_k(anomaly_origin, label_anomaly, 40)  # Percentage anomaly detection adjustment
    
    precision, recall, f_score, support = \
        precision_recall_fscore_support(label_anomaly, anomaly, average='binary')
    print('40% ratio:',
        "Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f}".format(
            precision, recall, f_score))
    score[-1].extend([precision, recall, f_score])
    
    anomaly = adjust_predictions_k(anomaly_origin, label_anomaly, 60)  # Percentage anomaly detection adjustment
    
    precision, recall, f_score, support = \
        precision_recall_fscore_support(label_anomaly, anomaly, average='binary')
    print('60% ratio:',
        "Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f}".format(
            precision, recall, f_score))
    score[-1].extend([precision, recall, f_score])



def train_and_test(model_name, dataset_name, device, proporation_mean):
    global score  
    num_max_anomaly = args.num_max_anomaly
    # Experiment runs 5 times to take average
    seed_list = None
    if args.seed_list:
        if isinstance(args.seed_list, list):
            seed_list = args.seed_list
        else:
            seed_list = [int(s) for s in args.seed_list.split(',')]

    for i in range(args.itr):    
        # Set random seed for current iteration
        current_seed = seed_list[i]
        print(f"Using random seed: {current_seed}")
        np.random.seed(current_seed)
        torch.manual_seed(current_seed)
        torch.cuda.manual_seed_all(current_seed)
        
        mode = 'SN'
        
        # Set parameters
        args.input_len = input_len
        args.label_len = label_len
        args.batch_size = batch_size

        # Initialize and train model
        model = init_and_train_model(model_name, dataset_name, mode, device, current_seed)
        print('Model loaded or training completed')
        # Clear cache
        torch.cuda.empty_cache()
        
        # Execute normal+mean detection
        print(f'Starting: {mode}+normal+mean')
        mse_thresh_mean, mse_thresh_per_dim_3_sigma, mse_thresh_per_dim_5_sigma = predict_normal(
            model, dataset_name, mode, device, model_name, 
            mse_thresh=None, save_mses=False, 
            proporation_mean=proporation_mean, proporation_pre_dim=0.5, deal='mean',
            current_seed=current_seed
        )
        torch.cuda.empty_cache()
        
        # Execute replacement+add detection
        print(f'Starting: {mode}+replacement+add')
        predict_replacement_add(
            model, dataset_name, mode, device, model_name,
            mse_thresh_mean, proporation_mean,
            mse_thresh_per_dim_3_sigma, mse_thresh_per_dim_5_sigma,
            num_max_anomaly, current_seed=current_seed
        )
        torch.cuda.empty_cache()

    # Take average of scores 
    score = np.array(score).reshape(args.itr, -1, 12)  
    score_avg = np.mean(score, axis=0)  
    # Extract scores for different detection modes
    modes = ["No adjustment", "20% adjustment", "40% adjustment", "60% adjustment"]
    metrics = ["Precision(P)", "Recall(R)", "F1-score"]

    # Print formatted result table
    print("\n================ Anomaly Detection Results ================")
    print(f"{'Mode':<10} {'Precision(P)':<10} {'Recall(R)':<10} {'F1-score':<10}")
    print("="*43)

    for i, mode in enumerate(modes):
        p = score_avg[0][i*3]
        r = score_avg[0][i*3+1]
        f1 = score_avg[0][i*3+2]
        print(f"{mode:<10} {p:.4f}       {r:.4f}       {f1:.4f}")

    print("="*43)
    print(f"Note: Results are average of {args.itr} experiments, random seeds: {seed_list}")



if __name__ == '__main__':
    score=[]
    args=init()
    # Load preset parameters based on dataset name
    dataset_configs = get_dataset_config(args.data)
    
    # If not explicitly specified in command line, use preset parameters
    if not args.input_len:
        args.input_len = dataset_configs['input_len']
    if not args.label_len:
        args.label_len = dataset_configs['label_len']
    if not args.dropout:
        args.dropout = dataset_configs['dropout']
    if not args.anomaly_ratio:
        args.anomaly_ratio = dataset_configs['anomaly_ratio']
    if not args.num_max_anomaly:
        args.num_max_anomaly = dataset_configs['num_max_anomaly']

    input_len = args.input_len
    label_len = args.label_len
    anomaly_ratio = args.anomaly_ratio
    batch_size = args.batch_size
    device = torch.device(f"cuda:{args.gpu}" if args.use_gpu else "cpu")

    mp.set_start_method('spawn')  # Use 'spawn' start method
    if args.reproducible:
        if args.seed_list == None:
            args.seed_list=[70836, 67836, 68836, 69836, 71836]
    else:
        args.seed_list = [int(time.time() * 1000 + i * 1000) % 100000 for i in range(5)]

    # Initialization
    model_name = args.model  # Model name
    dataset_name=args.data   # Dataset name
    score = []
    seed_list=train_and_test(model_name, dataset_name, device, proporation_mean=args.anomaly_ratio)