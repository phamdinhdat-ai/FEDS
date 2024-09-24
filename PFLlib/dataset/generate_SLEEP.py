import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import os 
from utils.HAR_utils import *


data_path = "Sleep/"
def generate_sleep(dir_path, seq_len = 100, overlap = 0.4):
    config_path = dir_path + "config.json"
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"

    if not os.path.exists(train_path):
        os.makedirs(train_path)
    if not os.path.exists(test_path):
        os.makedirs(test_path)
    
    X_train, X_val, y_train, y_val = load_data_sleep(data_path + "all/", sequence_length=seq_len, overlap=overlap)
    statistic = []
    num_clients = len(X_train)
    train_data = []
    test_data = []
    num_classes = np.unique(y_train[0])
    for i in range(num_clients): 
        classes, counts = np.unique(y_train[i], return_counts = True)
        statistic.append({"classes": classes.tolist(), "n_samples": counts.tolist()})
        train_data.append({"x":X_train[i], "y": y_train[i]})
        test_data.append({"x":X_val[i], "y": y_val[i]})
        print(f"Client {i}\t Size of data: {len(X_train[i])}\t Labels: ", np.unique(y_train[i]))
        print(f"\t\t Samples of labels: ", [counts])
        print(f"\t\t Shape of data: ", X_train[i].shape)
        print(f"\t\t Shape of label: ", y_train[i].shape)
        print("-" * 50)
        
    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes.tolist(), statistic)
    




def load_data_sleep(folder_path, sequence_length=20, overlap= 0.4, valid_ratio = 0.2):
    list_file = os.listdir(folder_path)
    train_X = np.array([])
    val_X = np.array([])
    train_y = np.array([])
    val_y = np.array([])
    if valid_ratio is not None: 
        for file in list_file:
            file_name = file.split("_")[-1]
            ps_name = file_name[:-4]
            print("<=====> Data is collected by {}<=====>".format(ps_name))
            
            X_train, X_val, y_train, y_val = load_and_process_data(f"{folder_path}/{file}", sequence_length=sequence_length, overlap=overlap, valid_ratio=valid_ratio)
            X_train = X_train.reshape(-1, 3, 1, sequence_length)
            X_val   = X_val.reshape(-1, 3, 1, sequence_length)
            X_train = np.expand_dims(X_train,  axis = 0)
            X_val = np.expand_dims(X_val,  axis = 0)
            y_train = np.expand_dims(y_train,  axis = 0)
            y_val = np.expand_dims(y_val,  axis = 0)

            if train_X.shape[0] == 0:
                train_X = X_train
                val_X = X_val
                train_y = y_train
                val_y = y_val
            else: 
                train_X = np.concatenate([train_X, X_train], axis=0)
                val_X = np.concatenate([val_X, X_val], axis=0)
                train_y = np.concatenate([train_y, y_train], axis=0)
                val_y = np.concatenate([val_y, y_val], axis=0)
        return train_X, val_X, train_y, val_y
    else: 
        for file in list_file:
            file_name = file.split("_")[-1]
            ps_name = file_name[:-4]
            print("<=====> Data is collected by {}<=====>".format(ps_name))
            
            X_train, y_train = load_and_process_data(f"{folder_path}/{file}", sequence_length=sequence_length, overlap=overlap, valid_ratio=valid_ratio)
            X_train = X_train.reshape(-1, 3, 1, sequence_length)
            X_val = X_val.reshape(-1, 3, 1, sequence_length)
            X_train = np.expand_dims(X_train,  axis = 0)
            y_train = np.expand_dims(y_train,  axis = 0)
            if train_X.shape[0] == 0:
                train_X = X_train
                train_y = y_train
                
            else: 
                train_X = np.concatenate([train_X, X_train], axis=0)
                train_y = np.concatenate([train_y, y_train], axis=0)
        return train_X, train_y
                

    


def load_and_process_data(file_path, sequence_length= 20, overlap = 0.3,  valid_ratio = None):
    dataset  = np.load(file=file_path)
    data = dataset[:,1:4]
    labels = dataset[:,4]
    steps = int(sequence_length - overlap * sequence_length)
    # normalize data across min max scale
    data_scaled = min_max_scale(data)
    X, sequence_labels = generate_data(data_scaled, labels, sequence_lenght= sequence_length, step=steps)
    y = LabelEncoder().fit_transform(sequence_labels)

    if valid_ratio is not None:
        X_train, X_val, y_train, y_val = train_test_split(X,y, test_size=valid_ratio, shuffle=True, random_state=42)
        return X_train, X_val, y_train, y_val 
    return X, y



def generate_data(X, y, sequence_lenght = 10, step = 1):
    X_local = []
    y_local = []
    for start in range(0, X.shape[0] - sequence_lenght, step):
        end = start + sequence_lenght
        X_local.append(X[start:end])
        y_local.append(y[end-1])
    return np.array(X_local), np.array(y_local)


def min_max_scale(dataset):
    scaler = MinMaxScaler()
    scaler.fit(dataset)
    scaled_data = scaler.transform(dataset)
    return scaled_data


if __name__ == "__main__":
    dir_path = "SLEEP/"
    generate_sleep(dir_path)