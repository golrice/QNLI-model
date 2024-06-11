##################################################################################################################################
import os

words_path = os.path.join(os.path.dirname(__file__), '..', "resources", "words", "save_words_50d.npy")
vector_path = os.path.join(os.path.dirname(__file__), '..', "resources", "vector", "save_vector_50d.npy")
train_data_path = os.path.join(os.path.dirname(__file__), '..', "resources", "train_40.tsv")
dev_data_path = os.path.join(os.path.dirname(__file__), '..', "resources", "dev_40.tsv")

hidden_dim = 128
output_dim = 2
n_layers = 2
bidirectional = True
dropout = 0.5
lr = 0.001

##################################################################################################################################
import torch
import torch.nn as nn

class QNLIModel(nn.Module):
    def __init__(self, embedding_matrix, hidden_dim, output_dim, n_layers, bidirectional, dropout):
        super(QNLIModel, self).__init__()
        # 假设embedding_matrix是一个vocab_size x embedding_dim的矩阵
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix))
        self.lstm = nn.LSTM(self.embedding.embedding_dim, 
                             hidden_dim, 
                             n_layers, 
                             bidirectional=bidirectional, 
                             dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        
    def forward(self, x, lengths):
        embedded = self.embedding(x)
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, lengths, batch_first=True)
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        return self.fc(hidden)

##################################################################################################################################
import numpy as np
import pandas as pd

words_list = np.load(words_path).tolist()
embedding_matrix = np.load(vector_path)
dev_data = pd.read_csv(dev_data_path, sep='\t', header=0, on_bad_lines='skip')
train_data = pd.read_csv(train_data_path, sep='\t', header=0, on_bad_lines='skip')

model = QNLIModel(embedding_matrix, hidden_dim, output_dim, n_layers, bidirectional, dropout)
criterion = nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters(), lr=lr)
