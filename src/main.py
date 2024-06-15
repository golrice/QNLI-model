##################################################################################################################################
import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
##################################################################################################################################
import os

words_path = os.path.join(os.path.dirname(__file__), '..', "resources", "words", "save_words_300d.npy")
vector_path = os.path.join(os.path.dirname(__file__), '..', "resources", "vector", "save_vector_300d.npy")

accuracy_path = os.path.join(os.path.dirname(__file__), '..', "accuracy300d.png")
loss_path = os.path.join(os.path.dirname(__file__), '..', "loss300d.png")
time_train_path = os.path.join(os.path.dirname(__file__), '..', "time_train300d.png")
time_validate_path = os.path.join(os.path.dirname(__file__), '..', "time_validate300d.png")

train_data_path = os.path.join(os.path.dirname(__file__), '..', "resources", "train_40.tsv")
dev_data_path = os.path.join(os.path.dirname(__file__), '..', "resources", "dev_40.tsv")

hidden_dim = 128
output_dim = 2
n_layers = 2
bidirectional = True
dropout = 0.5
lr = 0.001

batch_size = 36
num_epochs = 10

# 记录正确率
validate_accuracy = []
# loss记录
train_loss = []
# 训练时间记录
time_train = []
# 验证时间记录
time_validate = []

logging.info(f"critical constants: batch_size={batch_size}, num_epochs={num_epochs}")

logging.info(f"constants init finished")

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
        
    def forward(self, question, answer, lengths_of_question, lengths_of_answer):
        # 将问题和答案通过嵌入层
        embedded_question = self.embedding(question)
        embedded_answer = self.embedding(answer)

        # 打包pack_padded_sequence，如果问题和答案长度不一
        packed_question = nn.utils.rnn.pack_padded_sequence(embedded_question, lengths_of_question, batch_first=True, enforce_sorted=False)
        packed_answer = nn.utils.rnn.pack_padded_sequence(embedded_answer, lengths_of_answer, batch_first=True, enforce_sorted=False)

        question_output, (question_hidden, question_cell) = self.lstm(packed_question)
        answer_output, (answer_hidden, answer_cell) = self.lstm(packed_answer)

        # 将问题和答案的最后一个时间步的隐藏状态合并
        combined = torch.cat((question_hidden[-2, :, :], answer_hidden[-2, :, :]), dim=1)

        return self.fc(self.dropout(combined))

##################################################################################################################################
import numpy as np
import pandas as pd

words_list = np.load(words_path).tolist()
embedding_matrix = np.load(vector_path)
# 莫名读取错误 应该是文件问题
# train_data = pd.read_csv(train_data_path, sep='\t', header=0, on_bad_lines='skip', encoding='utf-8')
# dev_data = pd.read_csv(dev_data_path, sep='\t', header=0, on_bad_lines='skip', encoding='utf-8')

with open(train_data_path, 'r', encoding='utf-8') as file:
    train_data = list(map(lambda x: x.strip().split('\t'), file.readlines()))
    # 将label转换成0/1
    train_data = [(id, question, answer, np.where(label == 'entailment', 1, 0)) for id, question, answer, label in train_data]

with open(dev_data_path, 'r', encoding='utf-8') as file:
    dev_data = list(map(lambda x: x.strip().split('\t'), file.readlines()))
    dev_data = [(id, question, answer, np.where(label == 'entailment', 1, 0)) for id, question, answer, label in dev_data]

max_seq_len = 10

for id, question, answer, label in train_data:
    max_seq_len = max(max_seq_len, len(question.split()), len(answer.split()))

logging.info(f"data loaded, with max_seq_len={max_seq_len}")

##################################################################################################################################
from torch.utils.data import Dataset, DataLoader
import random

def convert_to_index(sentence, words_list, words_set):
    result = []
    for word in sentence.lower().split():
        if word in words_set:
            result.append(words_list.index(word))
        else:
            result.append(random.randint(0, len(words_list)-1))
    return result

class SimpleDataset(Dataset):
    def __init__(self, words_list, data):
        self.words_list = words_list
        self.data = data
        self.words_set = set(words_list)

    def __getitem__(self, index):
        index, question, answer, label = self.data[index]
        # 转换为index 还需要注意某些词没有对应的index
        question_index = torch.tensor(convert_to_index(question, self.words_list, self.words_set))
        answer_index = torch.tensor(convert_to_index(answer, self.words_list, self.words_set))

        question_lengths = len(question_index)
        answer_lengths = len(answer_index)

        # 填充
        padded_question = nn.functional.pad(question_index, (0, max_seq_len - question_lengths), 'constant', 0)
        padded_answer = nn.functional.pad(answer_index, (0, max_seq_len - answer_lengths), 'constant', 0)

        return index, padded_question, padded_answer, label, question_lengths, answer_lengths

    def __len__(self):
        return len(self.data)

train_dataset = SimpleDataset(words_list, train_data)
dev_dataset = SimpleDataset(words_list, dev_data)

train_data_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
dev_data_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

logging.info(f"data loader init finished")

##################################################################################################################################

model = QNLIModel(embedding_matrix, hidden_dim, output_dim, n_layers, bidirectional, dropout)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

logging.info(f"model init finished")

##################################################################################################################################
import time

def train_model(model, data_loader, criterion, optimizer):
    total_loss = 0
    start = time.time()
    for batch in data_loader:
        _, question, answer, label, question_lengths, answer_lengths = batch
        optimizer.zero_grad()  # 清除之前的梯度
        outputs = model(question, answer, question_lengths, answer_lengths)
        loss = criterion(outputs, label)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数
        total_loss += loss.item()
    end = time.time()
    time_train.append(end - start)
    return total_loss / len(data_loader)

# 定义评估函数
def evaluate_model(model, data_loader):
    correct = 0
    total = 0
    start = time.time()
    with torch.no_grad():
        for batch in data_loader:
            index, question, answer, label, question_lengths, answer_lengths = batch
            outputs = model(question, answer, question_lengths, answer_lengths)
            _, predicted = torch.max(outputs.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
    end = time.time()
    time_validate.append(end - start)
    return 100 * correct / total

logging.info(f"start training")

# 训练模型
for epoch in range(num_epochs):
    model.train()  # 设置模型为训练模式
    loss = train_model(model, train_data_loader, criterion, optimizer)  # 训练模型
    print(f'Epoch {epoch+1}, Loss: {loss}')
    train_loss.append(loss)
    
    # 在验证集上评估模型
    model.eval()  # 设置模型为评估模式
    validation_accuracy = evaluate_model(model, dev_data_loader)
    validate_accuracy.append(validation_accuracy)
    print(f'Validation Accuracy: {validation_accuracy}')

logging.info(f"training and validation finished")

##################################################################################################################################
logging.getLogger().setLevel(logging.CRITICAL)

# 绘制正确率图
import matplotlib.pyplot as plt

plt.plot(range(num_epochs), validate_accuracy)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Validation Accuracy')
plt.savefig(accuracy_path)

# 绘制loss图
plt.clf()
plt.plot(range(num_epochs), train_loss)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.savefig(loss_path)

# 绘制训练时间图 标注单位
plt.clf()
plt.plot(range(num_epochs), time_train)
plt.xlabel('Epochs')
plt.ylabel('Time(s)')
plt.title('Training Time')
plt.savefig(time_train_path)

# 绘制验证时间图
plt.clf()
plt.plot(range(num_epochs), time_validate)
plt.xlabel('Epochs')
plt.ylabel('Time(s)')
plt.title('Validation Time')
plt.savefig(time_validate_path)

##################################################################################################################################
# 保存模型
torch.save(model.state_dict(), os.path.join(os.path.dirname(__file__), '..', "resources", "model", "save_model.pth"))

# 说明指标
# 最高的准确率
max_accuracy = max(validate_accuracy)
# 最低的loss
min_loss = min(train_loss)
# 最长的训练时间
max_train_time = max(time_train)
# 最短的训练时间
min_train_time = min(time_train)
# 最长的验证时间
max_validate_time = max(time_validate)
# 最短的验证时间
min_validate_time = min(time_validate)

print(f"max_accuracy: {max_accuracy}")
print(f"min_loss: {min_loss}")
print(f"max_train_time: {max_train_time}, min_train_time: {min_train_time}")
print(f"max_validate_time: {max_validate_time}, min_validate_time: {min_validate_time}")
