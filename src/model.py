import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# Define the LSTM model architecture
class LSTMModel(nn.Module):
    def __init__(self, input_size, num_layers, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size,
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)#outputing the # features

    def forward(self, x):
        output, _ = self.lstm(x)
        output = self.fc(output[:, -1, :])
        return output

# Define the GRU model architecture
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        output, _ = self.gru(x)
        output = self.dropout(output[:, -1, :])
        output = self.fc(output)
        return output

# Define the BiLSTM model architecture
class BiLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(BiLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, 1)  # Multiply hidden_size by 2 for bidirectional LSTM

    def forward(self, x):
        output, _ = self.lstm(x)
        output = self.fc(output[:, -1, :])
        return output

# Define the Attention module
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attn_weights = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, lstm_output, lstm_hidden):
        attn_energies = self.attn_weights(lstm_output)
        attn_energies = torch.bmm(attn_energies, lstm_hidden.permute(0, 2, 1))
        attn_energies = F.softmax(attn_energies, dim=1)
        attn_applied = torch.bmm(attn_energies, lstm_output)
        return attn_applied

# Define the LSTM model with attention
class LSTMModelWithAttention(nn.Module):
    def __init__(self, input_size, num_layers, hidden_size, output_size):
        super(LSTMModelWithAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.attention = Attention(hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_output, _ = self.lstm(x)
        attn_applied = self.attention(lstm_output, lstm_output)  # Use lstm_output for attention
        output = self.fc(attn_applied.squeeze(1))
        return output

# Define LR model
class LinearRegression(nn.Module): 
    def __init__(self, input_dim): 
        super(LinearRegression, self).__init__() 
        self.linear = nn.Linear(input_dim, 1)  # Linear layer
        
    def forward(self, X): 
        out = self.linear(X)
        return out.squeeze(1)  # Squeeze to get (N,) shape instead of (N,1)

# Define KNN regressor
class KNNRegressor(nn.Module):
    def __init__(self, k=5):
        super(KNNRegressor, self).__init__()
        self.k = k

    def forward(self, X_train, y_train, X_test):
        distances = torch.cdist(X_test, X_train)  # Compute distances between test and training points
        _, indices = torch.topk(distances, self.k, largest=False)  # Find indices of k nearest neighbors
        knn_outputs = y_train[indices]  # Get outputs of k nearest neighbors
        predictions = torch.mean(knn_outputs, dim=1)  # Take the average of outputs as predictions
        return predictions