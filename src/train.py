import time
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from river import forest,tree, neural_net, ensemble, evaluate, metrics, preprocessing, stream

# XGboost functions

def train_xgb_model(model, X_train_normalized, y_train_normalized):
    """
    XGBoost script for holdout training (pre-training)
    """
    start_time = time.time()
    model = model.fit(X_train_normalized, y_train_normalized)
    end_time = time.time()
    training_time = end_time - start_time
    return model, training_time

def test_xgb_model(model, X_test_normalized):
    """
    XGBoost script for holdout test
    """
    start_time = time.time()
    y_pred_normalized = model.predict(X_test_normalized)
    end_time = time.time()
    inference_time = end_time - start_time
    return y_pred_normalized, inference_time

def test_xgb_model_update(model, X_train_normalized,  X_test_normalized, y_train_normalized, y_test_normalized):
    """
    XGBoost script for online testing
    """
    start_time = time.time()

    predictions = []
    
    total = len(X_test_normalized)
    
    with tqdm(total=total, desc="Progress", position=0, leave=True, bar_format="{l_bar}{bar}{r_bar}") as pbar:
                for idx in range(total):
                    y_pred = model.predict(np.array(X_test_normalized.loc[idx, :]).reshape(1, -1))
                    predictions.append(y_pred)
                    model.fit(np.vstack((X_train_normalized, X_test_normalized.loc[idx, :])),
                            np.concatenate((y_train_normalized, [y_test_normalized[idx]])))
                    pbar.update(1)
                    
    end_time = time.time()
    
    inference_time = end_time - start_time
    
    y_pred_normalized = np.array(predictions).squeeze(1)
    
    return y_pred_normalized, inference_time

# River Functions

def train_river_model(model, X_train_normalized, y_train_normalized):
    # pre-training
    start_time = time.time()
    for x, y in stream.iter_pandas(X_train_normalized, y_train_normalized):
        model.learn_one(x, y)
    end_time = time.time()
    training_time = end_time - start_time
    return model, training_time

def test_river_model(model,  X_test_normalized, y_test_normalized):
    start_time = time.time()
    predictions = []
    for x, y in stream.iter_pandas(X_test_normalized, y_test_normalized):
                y_pred = model.predict_one(x)
                predictions.append(y_pred)
                model.learn_one(x, y)
                y_pred_normalized = np.array(predictions)#[X,]
    end_time = time.time()
    inference_time = end_time - start_time
    return np.array(predictions), inference_time

# Scikitlearn partial_fit functions
def train_sklearn_partial_fit(model, X_train_normalized, y_train_normalized):
    start_time = time.time()
    X_train_normalized_np = X_train_normalized.values
    y_train_normalized_np = y_train_normalized.values
    trained_model = model.partial_fit(X_train_normalized_np, y_train_normalized_np)
    end_time = time.time()
    training_time = end_time - start_time
    return trained_model, training_time

def test_sklearn_partial_fit(model, X_test_normalized,y_test_normalized):
    start_time = time.time()
    predictions = []
    X_test_normalized = X_test_normalized.values
    y_test_normalized = y_test_normalized.values

    for idx in range(len(X_test_normalized)):

        x = X_test_normalized[idx]
        x = np.reshape(x, (x.shape[0],1)).T # feat,1 
        y = y_test_normalized[idx]
        y_pred = model.predict(x)[0]
        predictions.append(y_pred)
        model.partial_fit(x, y.reshape(-1))#x -->(1, feat), y--> y.reshape(-1).shape yields shape(1,)
        y_pred_normalized = np.array(predictions)#[X,]
        
    end_time = time.time()
    inference_time = end_time - start_time
    return np.array(predictions), inference_time

# Scikitlearn fit functions
def train_sklearn_model(model, X_train_normalized, y_train_normalized):
    start_time = time.time()
    trained_model = model.fit(X_train_normalized, y_train_normalized)
    end_time = time.time()
    training_time = end_time - start_time
    return trained_model, training_time

def test_sklearn_model(model, X_test_normalized):
    start_time = time.time()
    y_pred_normalized = model.predict(X_test_normalized)
    end_time = time.time()
    inference_time = end_time - start_time
    return y_pred_normalized, inference_time


# Pytorch functions

def train_pytorch_model(model, X_train_normalized, y_train_normalized, attention, device):
    start_time = time.time()
    # Convert to numpy array
    X_train_normalized, y_train_normalized = np.array(X_train_normalized), np.array(y_train_normalized)
    
    # Reshape data 
    X_train_normalized = np.expand_dims(np.array(X_train_normalized), 2)
    y_train_normalized = np.expand_dims(np.array(y_train_normalized), 1)
    
    # Convert data to tensors
    X_train_tensor = torch.from_numpy(X_train_normalized).float()
    y_train_tensor = torch.from_numpy(y_train_normalized).float()
    
    # Send train data to cuda
    X_train_tensor = X_train_tensor.to(device)
    y_train_tensor = y_train_tensor.to(device)
    
    # Reshape due to attention input 
    if attention == True:
        X_train_tensor = X_train_tensor.view(X_train_tensor.shape[2], X_train_tensor.shape[0], X_train_tensor.shape[1])
    print(f"X_train_tensor shape: {X_train_tensor.shape}")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(100):
        optimizer.zero_grad()
        # outputs = model(X_train_tensor.to(device))
        outputs = model(X_train_tensor)
        outputs = outputs.squeeze(0)
        # loss = criterion(outputs, y_train_tensor.to(device))
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

    end_time = time.time()
    training_time = end_time - start_time
    return model, training_time

def test_pytorch_model(model, X_test_normalized, attention, device):
    start_time = time.time()
    
    # Convert to numpy array
    X_test_normalized = np.array(X_test_normalized)
    X_test_normalized = np.expand_dims(np.array(X_test_normalized),2)
    
    # Convert the data to PyTorch tensors
    X_test_tensor = torch.from_numpy(X_test_normalized).float()
    
    # Send test data to cuda
    X_test_tensor = X_test_tensor.to(device)
    
    # Reshape due to attention input 
    if attention == True:
        X_test_tensor = X_test_tensor.view(X_test_tensor.shape[2], X_test_tensor.shape[0], X_test_tensor.shape[1])
    
    # Predict the CPU values for the testing data
    with torch.no_grad():
        outputs = model(X_test_tensor)
        y_pred_normalized = outputs.detach().cpu().numpy()
    end_time = time.time()
    inference_time = end_time - start_time

    return y_pred_normalized, inference_time

def hingeLoss(outputVal,dataOutput,model):
    loss1=torch.sum(torch.clamp(1 - torch.matmul(outputVal.t(),dataOutput),min=0))
    loss2=torch.sum(model.linear.weight ** 2)  # l2 penalty
    totalLoss=loss1+loss2
    return(totalLoss)
    
def train_pytorch_model_LR(model, X_train_normalized, y_train_normalized, loss):
    start_time = time.time()
    
    X_train_tensor = torch.from_numpy(np.array(X_train_normalized)).float()
    y_train_tensor = torch.from_numpy(np.array(y_train_normalized)).float()
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(10):
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        if loss == True:
            # In case of SVR calculate Hinge Loss
            loss = hingeLoss(y_train_tensor,outputs,model)
        else:
            loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

    end_time = time.time()
    training_time = end_time - start_time
    return model, training_time

def test_pytorch_model_LR(model, X_test_normalized, attention):
    start_time = time.time()
    X_test_tensor = torch.from_numpy(np.array(X_test_normalized)).float()
    with torch.no_grad():
        outputs = model(X_test_tensor)
        y_pred_normalized = outputs.detach().numpy()
    end_time = time.time()
    inference_time = end_time - start_time

    return y_pred_normalized, inference_time

def test_pytorch_model_KNR(model, X_train_normalized, y_train_normalized, X_test_normalized):
    start_time = time.time()
    X_train_tensor = torch.from_numpy(np.array(X_train_normalized)).float()
    y_train_tensor = torch.from_numpy(np.array(y_train_normalized)).float()
    X_test_tensor = torch.from_numpy(np.array(X_test_normalized)).float()
    predictions = model(X_train_tensor, y_train_tensor, X_test_tensor)
    y_pred_normalized = np.array(predictions)
    end_time = time.time()
    inference_time = end_time - start_time

    return y_pred_normalized, inference_time