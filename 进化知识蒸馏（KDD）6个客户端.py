import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE, RandomOverSampler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import recall_score, accuracy_score, precision_score
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import csv
import os
from collections import defaultdict
from sklearn.metrics import roc_curve, accuracy_score, precision_score, recall_score, f1_score
import pickle
import warnings
import torch.nn.functional as F
import torch.fft as fft

warnings.filterwarnings("ignore")
from torchvision import transforms
from sklearn.preprocessing import StandardScaler

# 创建结果保存目录
save_dir = './result/NSL-KDD(进化知识蒸馏)'
os.makedirs(save_dir, exist_ok=True)


def get_model_size(model):
    """计算模型参数的字节数（包含参数和缓冲区）"""
    param_size = 0
    buffer_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    return param_size + buffer_size


class ResultLogger:
    def __init__(self, client_name):
        self.client_name = client_name
        self.metrics = defaultdict(list)
        self.best_metrics = {
            'epoch': 0,
            'test_acc': 0,
            'test_f1': 0,
            'test_precision': 0,
            'test_recall': 0,
            'train_acc': 0,
            'train_f1': 0,
            'train_precision': 0,
            'train_recall': 0
        }

        # 初始化CSV文件
        self.csv_file = os.path.join(save_dir, f'{client_name}_metrics.csv')
        if not os.path.exists(self.csv_file):
            with open(self.csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Epoch', 'TrainLoss', 'TrainAcc', 'TrainPrecision',
                                 'TrainRecall', 'TrainF1', 'TestLoss', 'TestAcc',
                                 'TestPrecision', 'TestRecall', 'TestF1'])

    def log_epoch(self, epoch, train_loss, train_acc, train_precision, train_recall, train_f1,
                  test_loss, test_acc, test_precision, test_recall, test_f1):
        # 记录所有指标
        self.metrics['Epoch'].append(epoch)
        self.metrics['TrainLoss'].append(train_loss)
        self.metrics['TrainAcc'].append(train_acc)
        self.metrics['TrainPrecision'].append(train_precision)
        self.metrics['TrainRecall'].append(train_recall)
        self.metrics['TrainF1'].append(train_f1)
        self.metrics['TestLoss'].append(test_loss)
        self.metrics['TestAcc'].append(test_acc)
        self.metrics['TestPrecision'].append(test_precision)
        self.metrics['TestRecall'].append(test_recall)
        self.metrics['TestF1'].append(test_f1)

        # 更新最佳指标
        current_test_f1 = test_f1
        if current_test_f1 > self.best_metrics['test_f1']:
            self.best_metrics.update({
                'epoch': epoch,
                'test_acc': test_acc,
                'test_f1': current_test_f1,
                'test_precision': test_precision,
                'test_recall': test_recall,
                'train_acc': train_acc,
                'train_f1': train_f1,
                'train_precision': train_precision,
                'train_recall': train_recall
            })

        # 写入CSV
        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, train_acc, train_precision,
                             train_recall, train_f1, test_loss, test_acc,
                             test_precision, test_recall, test_f1])

    def save_best_metrics(self):
        # 保存最佳指标到单独文件
        best_file = os.path.join(save_dir, f'{self.client_name}_best_metrics.csv')
        with open(best_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Metric', 'Value', 'Epoch'])
            for metric, value in self.best_metrics.items():
                if metric != 'epoch':
                    writer.writerow([metric, value, self.best_metrics['epoch']])


warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())
torch.backends.cudnn.benchmark = True

from sklearn.preprocessing import LabelEncoder

# 读取6个客户端的数据
df1 = pd.read_csv('data/NSL-KDD/client_1_data.csv')
df2 = pd.read_csv('data/NSL-KDD/client_2_data.csv')
df3 = pd.read_csv('data/NSL-KDD/client_3_data.csv')
df4 = pd.read_csv('data/NSL-KDD/client_4_data.csv')
df5 = pd.read_csv('data/NSL-KDD/client_5_data.csv')
df6 = pd.read_csv('data/NSL-KDD/client_6_data.csv')
df_test = pd.read_csv('data/NSL-KDD/global_test_data.csv')

# 打印列名以检查
print("Client 1 列名:", df1.columns.tolist())
print("Client 2 列名:", df2.columns.tolist())
print("Client 3 列名:", df3.columns.tolist())
print("Client 4 列名:", df4.columns.tolist())
print("Client 5 列名:", df5.columns.tolist())
print("Client 6 列名:", df6.columns.tolist())
print("Test 列名:", df_test.columns.tolist())

label_encoder = LabelEncoder()

# 确定哪些分类列实际存在
categorical_cols = []
for col in ['protocol_type', 'service', 'flag']:  # KDD常见分类列
    if col in df1.columns:
        categorical_cols.append(col)

# 只处理存在的列
for col in categorical_cols:
    df1[col] = label_encoder.fit_transform(df1[col].astype(str))
    df2[col] = label_encoder.fit_transform(df2[col].astype(str))
    df3[col] = label_encoder.fit_transform(df3[col].astype(str))
    df4[col] = label_encoder.fit_transform(df4[col].astype(str))
    df5[col] = label_encoder.fit_transform(df5[col].astype(str))
    df6[col] = label_encoder.fit_transform(df6[col].astype(str))
    df_test[col] = label_encoder.fit_transform(df_test[col].astype(str))

# 客户端1数据处理
df_train1 = pd.read_csv('data/NSL-KDD/client_1_data.csv')
scaler = StandardScaler()
X_train1 = df_train1.iloc[:, :-1]
X_train1 = X_train1 * 255
y_train1 = df_train1.iloc[:, -1]
y_train1 = y_train1.astype(float)
y_train1 = y_train1.astype(int)
y_train1 = list(y_train1)
y_train1 = [1 if x == 1 else 0 for x in y_train1]
y_train1 = pd.DataFrame(y_train1)
X_test1 = df_test.iloc[:, :-1]
X_test1 = X_test1 * 255
y_test1 = df_test.iloc[:, -1]
y_test1 = y_test1.astype(float)
y_test1 = y_test1.astype(int)
y_test1 = list(y_test1)
y_test1 = [1 if x == 1 else 0 for x in y_test1]
y_test1 = pd.DataFrame(y_test1)
print("Client 1 训练数据分布:")
print(y_train1.value_counts())
clf = RandomOverSampler()
X_train1, y_train1 = clf.fit_resample(X_train1, y_train1)

# 客户端2数据处理
df_train2 = pd.read_csv('data/NSL-KDD/client_2_data.csv')
X_train2 = df_train2.iloc[:, :-1]
X_train2 = X_train2 * 255
y_train2 = df_train2.iloc[:, -1]
y_train2 = y_train2.astype(float)
y_train2 = y_train2.astype(int)
y_train2 = list(y_train2)
y_train2 = [1 if x == 1 else 0 for x in y_train2]
y_train2 = pd.DataFrame(y_train2)
X_test2 = df_test.iloc[:, :-1]
X_test2 = X_test2 * 255
y_test2 = df_test.iloc[:, -1]
y_test2 = y_test2.astype(float)
y_test2 = y_test2.astype(int)
y_test2 = list(y_test2)
y_test2 = [1 if x == 1 else 0 for x in y_test2]
y_test2 = pd.DataFrame(y_test2)
print("Client 2 训练数据分布:")
print(y_train2.value_counts())
clf = RandomOverSampler()
X_train2, y_train2 = clf.fit_resample(X_train2, y_train2)

# 客户端3数据处理
df_train3 = pd.read_csv('data/NSL-KDD/client_3_data.csv')
X_train3 = df_train3.iloc[:, :-1]
X_train3 = X_train3 * 255
y_train3 = df_train3.iloc[:, -1]
y_train3 = y_train3.astype(float)
y_train3 = y_train3.astype(int)
y_train3 = list(y_train3)
y_train3 = [1 if x == 1 else 0 for x in y_train3]
y_train3 = pd.DataFrame(y_train3)
X_test3 = df_test.iloc[:, :-1]
X_test3 = X_test3 * 255
y_test3 = df_test.iloc[:, -1]
y_test3 = y_test3.astype(float)
y_test3 = y_test3.astype(int)
y_test3 = list(y_test3)
y_test3 = [1 if x == 1 else 0 for x in y_test3]
y_test3 = pd.DataFrame(y_test3)
print("Client 3 训练数据分布:")
print(y_train3.value_counts())
clf = RandomOverSampler()
X_train3, y_train3 = clf.fit_resample(X_train3, y_train3)

# 客户端4数据处理
df_train4 = pd.read_csv('data/NSL-KDD/client_4_data.csv')
X_train4 = df_train4.iloc[:, :-1]
X_train4 = X_train4 * 255
y_train4 = df_train4.iloc[:, -1]
y_train4 = y_train4.astype(float)
y_train4 = y_train4.astype(int)
y_train4 = list(y_train4)
y_train4 = [1 if x == 1 else 0 for x in y_train4]
y_train4 = pd.DataFrame(y_train4)
X_test4 = df_test.iloc[:, :-1]
X_test4 = X_test4 * 255
y_test4 = df_test.iloc[:, -1]
y_test4 = y_test4.astype(float)
y_test4 = y_test4.astype(int)
y_test4 = list(y_test4)
y_test4 = [1 if x == 1 else 0 for x in y_test4]
y_test4 = pd.DataFrame(y_test4)
print("Client 4 训练数据分布:")
print(y_train4.value_counts())
clf = RandomOverSampler()
X_train4, y_train4 = clf.fit_resample(X_train4, y_train4)

# 客户端5数据处理
df_train5 = pd.read_csv('data/NSL-KDD/client_5_data.csv')
X_train5 = df_train5.iloc[:, :-1]
X_train5 = X_train5 * 255
y_train5 = df_train5.iloc[:, -1]
y_train5 = y_train5.astype(float)
y_train5 = y_train5.astype(int)
y_train5 = list(y_train5)
y_train5 = [1 if x == 1 else 0 for x in y_train5]
y_train5 = pd.DataFrame(y_train5)
X_test5 = df_test.iloc[:, :-1]
X_test5 = X_test5 * 255
y_test5 = df_test.iloc[:, -1]
y_test5 = y_test5.astype(float)
y_test5 = y_test5.astype(int)
y_test5 = list(y_test5)
y_test5 = [1 if x == 1 else 0 for x in y_test5]
y_test5 = pd.DataFrame(y_test5)
print("Client 5 训练数据分布:")
print(y_train5.value_counts())
clf = RandomOverSampler()
X_train5, y_train5 = clf.fit_resample(X_train5, y_train5)

# 客户端6数据处理
df_train6 = pd.read_csv('data/NSL-KDD/client_6_data.csv')
X_train6 = df_train6.iloc[:, :-1]
X_train6 = X_train6 * 255
y_train6 = df_train6.iloc[:, -1]
y_train6 = y_train6.astype(float)
y_train6 = y_train6.astype(int)
y_train6 = list(y_train6)
y_train6 = [1 if x == 1 else 0 for x in y_train6]
y_train6 = pd.DataFrame(y_train6)
X_test6 = df_test.iloc[:, :-1]
X_test6 = X_test6 * 255
y_test6 = df_test.iloc[:, -1]
y_test6 = y_test6.astype(float)
y_test6 = y_test6.astype(int)
y_test6 = list(y_test6)
y_test6 = [1 if x == 1 else 0 for x in y_test6]
y_test6 = pd.DataFrame(y_test6)
print("Client 6 训练数据分布:")
print(y_train6.value_counts())
clf = RandomOverSampler()
X_train6, y_train6 = clf.fit_resample(X_train6, y_train6)

# 示例：对所有客户端的数据处理
categorical_cols = ['proto', 'service', 'state', 'attack_cat']  # 替换为你的分类列名

for col in categorical_cols:
    if col in X_train1.columns:
        le = LabelEncoder()
        X_train1[col] = le.fit_transform(X_train1[col].astype(str))
        X_test1[col] = le.transform(X_test1[col].astype(str))
        X_train2[col] = le.fit_transform(X_train2[col].astype(str))
        X_test2[col] = le.transform(X_test2[col].astype(str))
        X_train3[col] = le.fit_transform(X_train3[col].astype(str))
        X_test3[col] = le.transform(X_test3[col].astype(str))
        X_train4[col] = le.fit_transform(X_train4[col].astype(str))
        X_test4[col] = le.transform(X_test4[col].astype(str))
        X_train5[col] = le.fit_transform(X_train5[col].astype(str))
        X_test5[col] = le.transform(X_test5[col].astype(str))
        X_train6[col] = le.fit_transform(X_train6[col].astype(str))
        X_test6[col] = le.transform(X_test6[col].astype(str))

# 初始化结果记录器
logger1 = ResultLogger('client1')
logger2 = ResultLogger('client2')
logger3 = ResultLogger('client3')
logger4 = ResultLogger('client4')
logger5 = ResultLogger('client5')
logger6 = ResultLogger('client6')
global_logger = ResultLogger('global_model')


# 查看每个数据集的类型分布
def label_distribution(y_train, y_test, dataset_name):
    print(f"Label Distribution in {dataset_name} Training Data:")
    print(y_train.value_counts())

    print(f"\nLabel Distribution in {dataset_name} Test Data:")
    print(y_test.value_counts())


# 对每个数据集调用函数
label_distribution(y_train1, y_test1, "UNSW Client 1")
label_distribution(y_train2, y_test2, "UNSW Client 2")
label_distribution(y_train3, y_test3, "UNSW Client 3")
label_distribution(y_train4, y_test4, "UNSW Client 4")
label_distribution(y_train5, y_test5, "UNSW Client 5")
label_distribution(y_train6, y_test6, "UNSW Client 6")


class LoadData(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        X = torch.tensor(self.X.iloc[index], dtype=torch.float32)
        X = F.pad(X, (0, 196 - X.shape[0]))  # 填充到196维
        X = X.view(1, 14, 14)  # 改为 [1, 14, 14] 形状
        y = torch.tensor(self.y.iloc[index], dtype=torch.long)
        return X, y


# 创建数据加载器
batch_size = 64

# 客户端1
train_data1 = LoadData(X_train1, y_train1)
test_data1 = LoadData(X_test1, y_test1)
train_loader1 = Data.DataLoader(train_data1, batch_size=batch_size, shuffle=True)
test_loader1 = Data.DataLoader(test_data1, batch_size=batch_size, shuffle=True)

# 客户端2
train_data2 = LoadData(X_train2, y_train2)
test_data2 = LoadData(X_test2, y_test2)
train_loader2 = Data.DataLoader(train_data2, batch_size=batch_size, shuffle=True)
test_loader2 = Data.DataLoader(test_data2, batch_size=batch_size, shuffle=True)

# 客户端3
train_data3 = LoadData(X_train3, y_train3)
test_data3 = LoadData(X_test3, y_test3)
train_loader3 = Data.DataLoader(train_data3, batch_size=batch_size, shuffle=True)
test_loader3 = Data.DataLoader(test_data3, batch_size=batch_size, shuffle=True)

# 客户端4
train_data4 = LoadData(X_train4, y_train4)
test_data4 = LoadData(X_test4, y_test4)
train_loader4 = Data.DataLoader(train_data4, batch_size=batch_size, shuffle=True)
test_loader4 = Data.DataLoader(test_data4, batch_size=batch_size, shuffle=True)

# 客户端5
train_data5 = LoadData(X_train5, y_train5)
test_data5 = LoadData(X_test5, y_test5)
train_loader5 = Data.DataLoader(train_data5, batch_size=batch_size, shuffle=True)
test_loader5 = Data.DataLoader(test_data5, batch_size=batch_size, shuffle=True)

# 客户端6
train_data6 = LoadData(X_train6, y_train6)
test_data6 = LoadData(X_test6, y_test6)
train_loader6 = Data.DataLoader(train_data6, batch_size=batch_size, shuffle=True)
test_loader6 = Data.DataLoader(test_data6, batch_size=batch_size, shuffle=True)

import torch.nn as nn
from tqdm import tqdm


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.2, stride=1):
        super().__init__()
        # 确保中间通道数足够
        mid_channels = out_channels // 4 if out_channels > 4 else 1

        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(dropout_rate)

        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, max(out_channels // 4, 1), kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(max(out_channels // 4, 1), out_channels, kernel_size=1),
            nn.Sigmoid()
        )

        # 更安全的shortcut处理
        self.shortcut = nn.Sequential()
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        attention = self.attention(out)
        out = out * attention
        out += residual
        out = F.relu(out)
        return self.dropout(out)


class Teacher(nn.Module):
    def __init__(self, num_blocks=4, input_dim=1, hidden_dim=8, output_dim=2, max_channels=64):
        super().__init__()
        self.initial_conv = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.blocks = nn.ModuleList()
        current_channels = hidden_dim
        for i in range(num_blocks):
            next_channels = min(current_channels * 2, max_channels)
            stride = 2 if i == 0 else 1
            self.blocks.append(ResidualBlock(current_channels, next_channels, stride=stride))
            current_channels = next_channels

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(current_channels, output_dim)

    def forward(self, x):
        x = x.view(x.size(0), 1, 14, 14)
        x = self.initial_conv(x)

        features = []
        for block in self.blocks:
            x = block(x)
            features.append(x.clone())

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        output = self.classifier(x)
        return features, output


class Student(nn.Module):
    def __init__(self, num_blocks=4, input_dim=1, hidden_dim=8, output_dim=2):
        super().__init__()
        # 完全复制Teacher的结构
        self.initial_conv = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # 保持完全相同的通道增长
        self.blocks = nn.ModuleList()
        current_channels = hidden_dim
        for i in range(num_blocks):
            next_channels = min(current_channels * 2, 64)  # 与Teacher相同的最大通道数
            stride = 2 if i == 0 else 1
            self.blocks.append(ResidualBlock(current_channels, next_channels, dropout_rate=0.1, stride=stride))
            current_channels = next_channels

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(current_channels, output_dim)

    def forward(self, x):
        x = x.view(x.size(0), 1, 14, 14)  # 调整输入形状
        x = self.initial_conv(x)

        features = []
        for block in self.blocks:
            x = block(x)
            features.append(x.clone())

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        output = self.classifier(x)
        return features, output


teacher_num_blocks = 4
student_num_blocks = 4
input_dim = 1
teacher_hidden_dim = 8
student_hidden_dim = 8
output_dim = 2

global_model = Teacher(teacher_num_blocks, input_dim, teacher_hidden_dim, output_dim).cuda()

# 创建模型实例
teacher1 = Teacher(teacher_num_blocks, input_dim, teacher_hidden_dim, output_dim).to(device)
teacher2 = Teacher(teacher_num_blocks, input_dim, teacher_hidden_dim, output_dim).to(device)
teacher3 = Teacher(teacher_num_blocks, input_dim, teacher_hidden_dim, output_dim).to(device)
teacher4 = Teacher(teacher_num_blocks, input_dim, teacher_hidden_dim, output_dim).to(device)
teacher5 = Teacher(teacher_num_blocks, input_dim, teacher_hidden_dim, output_dim).to(device)
teacher6 = Teacher(teacher_num_blocks, input_dim, teacher_hidden_dim, output_dim).to(device)

weight = torch.tensor([1.0, 3.0]).to(device)
criterion = nn.CrossEntropyLoss(weight=weight)

optimizer1 = optim.Adam(teacher1.parameters(), lr=0.0001, weight_decay=0.0001)
optimizer2 = optim.Adam(teacher2.parameters(), lr=0.0001, weight_decay=0.0001)
optimizer3 = optim.Adam(teacher3.parameters(), lr=0.0001, weight_decay=0.0001)
optimizer4 = optim.Adam(teacher4.parameters(), lr=0.0001, weight_decay=0.0001)
optimizer5 = optim.Adam(teacher5.parameters(), lr=0.0001, weight_decay=0.0001)
optimizer6 = optim.Adam(teacher6.parameters(), lr=0.0001, weight_decay=0.0001)

student1 = Student(student_num_blocks, input_dim, student_hidden_dim, output_dim).to(device)
student2 = Student(student_num_blocks, input_dim, student_hidden_dim, output_dim).to(device)
student3 = Student(student_num_blocks, input_dim, student_hidden_dim, output_dim).to(device)
student4 = Student(student_num_blocks, input_dim, student_hidden_dim, output_dim).to(device)
student5 = Student(student_num_blocks, input_dim, student_hidden_dim, output_dim).to(device)
student6 = Student(student_num_blocks, input_dim, student_hidden_dim, output_dim).to(device)

optimizer_s1 = optim.Adam(student1.parameters(), lr=0.0001, weight_decay=0.0001)
optimizer_s2 = optim.Adam(student2.parameters(), lr=0.0001, weight_decay=0.0001)
optimizer_s3 = optim.Adam(student3.parameters(), lr=0.0001, weight_decay=0.0001)
optimizer_s4 = optim.Adam(student4.parameters(), lr=0.0001, weight_decay=0.0001)
optimizer_s5 = optim.Adam(student5.parameters(), lr=0.0001, weight_decay=0.0001)
optimizer_s6 = optim.Adam(student6.parameters(), lr=0.0001, weight_decay=0.0001)


def distillation_loss(student_logits, teacher_logits, temperature):
    return nn.KLDivLoss(reduction='batchmean')(
        nn.functional.log_softmax(student_logits / temperature, dim=1),
        nn.functional.softmax(teacher_logits / temperature, dim=1)
    )


def feature_loss(student_features, teacher_features):
    loss = 0
    for i, (s_feat, t_feat) in enumerate(zip(student_features, teacher_features[-len(student_features):])):
        # 动态创建适配卷积层（确保每次都是新的）
        if s_feat.size(1) != t_feat.size(1):
            adjust_conv = nn.Conv2d(s_feat.size(1), t_feat.size(1), kernel_size=1).to(s_feat.device)
            s_feat = adjust_conv(s_feat)

        # 调整空间尺寸
        if s_feat.size()[2:] != t_feat.size()[2:]:
            s_feat = F.interpolate(s_feat, size=t_feat.size()[2:], mode='bilinear', align_corners=False)

        # 确保形状完全匹配
        assert s_feat.shape == t_feat.shape, f"Shape mismatch at layer {i}: {s_feat.shape} vs {t_feat.shape}"

        loss += F.mse_loss(s_feat, t_feat.detach())

    return loss / len(student_features)


def classification_loss(logits, labels, smoothing=0.1):
    n_class = logits.size(-1)
    one_hot = torch.zeros_like(logits).scatter(1, labels.unsqueeze(1), 1)
    one_hot = one_hot * (1 - smoothing) + smoothing / n_class
    return F.kl_div(F.log_softmax(logits, dim=1), one_hot, reduction='batchmean')


def train(teacher, student, teacher_optimizer, student_optimizer, dataloader, epoch, epochs):
    teacher.train()
    student.train()

    total_loss = 0
    correct = 0
    gradients = []
    max_grad_norm = 1.0

    base_temp = 3.0
    min_temp = 1.0
    current_temp = max(base_temp * (1 - epoch / epochs), min_temp)

    for data, labels in tqdm(dataloader):
        data, labels = data.to(device), labels.to(device).view(-1)

        # 1. 先计算教师模型输出
        with torch.no_grad():  # 教师模型不需要梯度
            teacher_features, teacher_logits = teacher(data)

        # 2. 计算学生模型输出（需要梯度）
        student_features, student_logits = student(data)

        # 动态温度计算
        with torch.no_grad():
            cos_sim = F.cosine_similarity(student_logits, teacher_logits, dim=1)
            dynamic_temp = current_temp * (1 + 0.5 * torch.mean(cos_sim).item())

        # 计算损失
        cls_loss = classification_loss(student_logits, labels)
        kd_loss = distillation_loss(student_logits, teacher_logits, dynamic_temp)
        feat_loss = feature_loss(student_features, teacher_features)

        total_loss = cls_loss + 0.6 * kd_loss + 0.4 * feat_loss

        # 3. 只对学生模型进行反向传播
        student_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), max_grad_norm)
        student_optimizer.step()
        gradients = [param.grad.clone() for param in student.parameters()]

        # 4. 单独计算教师模型的损失（可选）
        with torch.no_grad():
            teacher_loss = classification_loss(teacher_logits, labels)
            # 这里不再需要teacher_loss.backward()

        # 计算准确率
        _, preds = torch.max(student_logits, 1)
        correct += (preds == labels).sum().item()

    acc = correct / len(dataloader.dataset)
    return total_loss.item(), acc, gradients


def test(model, test_loader):
    model.eval()
    test_loss = 0
    test_correct = 0
    y_true = []
    y_pred = []
    # 添加类别权重
    weight = torch.tensor([1.0, 1.5]).to(device)  # 你的权重值
    criterion = nn.CrossEntropyLoss(weight=weight)
    label_smoothing = 0.1

    with torch.no_grad():
        for x, y in tqdm(test_loader):
            x = x.to(device)
            y = y.to(device)
            y = y.view(-1)
            _, student_logits = model(x)
            loss = classification_loss(student_logits, y)
            _, pred = torch.max(student_logits, 1)
            test_loss += loss.item()
            test_correct += (pred == y).sum().item()
            y_true.extend(y.tolist())
            y_pred.extend(student_logits.argmax(1).tolist())
    test_loss /= len(test_loader)
    test_acc = test_correct / len(test_loader.dataset)
    test_precision = precision_score(y_true, y_pred, )
    test_recall = recall_score(y_true, y_pred)
    test_f1 = f1_score(y_true, y_pred)
    fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=0, drop_intermediate=False)
    return test_loss, test_acc, test_precision, test_recall, test_f1, fpr, y_pred, y_true


def compute_sparsity(gradient):
    """
    计算梯度的稀疏性（非零元素比例）。

    参数:
        gradient (torch.Tensor): 梯度张量。

    返回:
        sparsity (float): 梯度的稀疏性（0 到 1 之间的值）。
    """
    non_zero_count = torch.count_nonzero(gradient).item()
    total_elements = gradient.numel()
    sparsity = 1 - (non_zero_count / total_elements)
    return sparsity


# 计算基于梯度稀疏性和数据量的权重
def compute_combined_weights(gradients_list, data_sizes):
    """
    基于梯度稀疏性和数据量计算客户端权重。

    参数:
        gradients_list (list): 包含每个客户端梯度的列表。
        data_sizes (list): 每个客户端的数据量。

    返回:
        weights (list): 每个客户端的权重。
    """
    sparsity_weights = []
    for gradients in gradients_list:

        # 计算每个客户端梯度的平均稀疏性
        client_sparsity = 0
        for gradient in gradients:
            client_sparsity += compute_sparsity(gradient)
        client_sparsity /= len(gradients)
        # 稀疏性越低，权重越高
        sparsity_weight = 1 / (client_sparsity + 1e-10)  # 避免除零
        sparsity_weights.append(sparsity_weight)

    # 将稀疏性权重和数据量结合
    combined_weights = torch.tensor(sparsity_weights) * torch.tensor(data_sizes)

    # 归一化权重
    combined_weights = combined_weights / combined_weights.sum()
    return combined_weights.tolist()


# 服务器加权聚合函数
def weighted_aggregate_gradients(gradients_list, weights):
    """
    对客户端上传的梯度进行加权平均聚合。

    参数:
        gradients_list (list): 包含每个客户端梯度的列表。
        weights (list): 每个客户端的权重。

    返回:
        avg_gradients (list): 聚合后的加权平均梯度。
    """
    if not gradients_list:
        raise ValueError("梯度列表为空，无法聚合！")

    # 初始化加权平均梯度
    avg_gradients = []

    # 对每一层梯度进行加权聚合
    for i in range(len(gradients_list[0])):
        # 获取所有客户端在该层的梯度
        layer_gradients = [gradients[i] for gradients in gradients_list]
        # 计算加权平均梯度
        weighted_layer_gradient = torch.zeros_like(layer_gradients[0])
        for grad, weight in zip(layer_gradients, weights):
            weighted_layer_gradient += grad * weight
        avg_gradients.append(weighted_layer_gradient)

    return avg_gradients


def update_global_model(global_model, avg_gradients, learning_rate=0.001):
    # 改用Adam优化器更新全局模型
    optimizer = optim.Adam(global_model.parameters(), lr=learning_rate)
    optimizer.zero_grad()

    # 手动设置梯度
    for param, grad in zip(global_model.parameters(), avg_gradients):
        param.grad = grad

    optimizer.step()


epochs = 50
# 假设6个客户端的数据量，需要根据实际数据调整
data_sizes = [14257, 17821, 21386, 17821, 33271, 29704]  # 示例数据量，请根据实际情况修改
num_clients = 6

# 初始化全局模型状态
global_model_state_dict = global_model.state_dict()
teacher1.load_state_dict(global_model_state_dict)
teacher2.load_state_dict(global_model_state_dict)
teacher3.load_state_dict(global_model_state_dict)
teacher4.load_state_dict(global_model_state_dict)
teacher5.load_state_dict(global_model_state_dict)
teacher6.load_state_dict(global_model_state_dict)

# 初始化记录列表
train_acc_list = [[] for _ in range(6)]
test_acc_list = [[] for _ in range(6)]
f1_list = [[] for _ in range(6)]
best_test_f1 = 0


def save_models(epoch):
    torch.save({
        'epoch': epoch,
        'global_model': global_model.state_dict(),
        'teacher1': teacher1.state_dict(),
        'teacher2': teacher2.state_dict(),
        'teacher3': teacher3.state_dict(),
        'teacher4': teacher4.state_dict(),
        'teacher5': teacher5.state_dict(),
        'teacher6': teacher6.state_dict(),
        'student1': student1.state_dict(),
        'student2': student2.state_dict(),
        'student3': student3.state_dict(),
        'student4': student4.state_dict(),
        'student5': student5.state_dict(),
        'student6': student6.state_dict(),
        'global_metrics': {
            'test_acc': global_test_acc,
            'test_f1': global_test_f1,
            'test_precision': global_test_precision,
            'test_recall': global_test_recall,
            'train_acc': global_train_acc,
            'train_f1': global_train_f1
        }
    }, os.path.join(save_dir, f'models_epoch_{epoch}.pth'))


# 训练循环
for epoch in range(epochs):
    print(f"\n=== Epoch {epoch + 1}/{epochs} ===")

    gradients_list = []

    # 客户端1训练
    print("Training Client 1...")
    train_loss1, train_acc1, gradient1 = train(teacher1, student1, optimizer1, optimizer_s1, train_loader1, epoch,
                                               epochs)
    test_loss1, test_acc1, test_precision1, test_recall1, test_f11, tst_fpr1, y_pred1, y_true1 = test(student1,
                                                                                                      test_loader1)
    _, train_logits = student1(next(iter(train_loader1))[0].to(device))
    print(classification_report(y_true1, y_pred1))
    print(confusion_matrix(y_true1, y_pred1))
    print(f"Client 1 - Epoch {epoch + 1}:")
    print(f"Train loss: {train_loss1:.4f}, Train accuracy: {train_acc1:.4f}")
    print(f"Test loss: {test_loss1:.4f}, Test accuracy: {test_acc1:.4f}, "
          f"Test precision: {test_precision1:.4f}, Test recall: {test_recall1:.4f}, "
          f"Test F1: {test_f11:.4f}")
    train_preds = train_logits.argmax(1).cpu().numpy()
    train_labels = next(iter(train_loader1))[1].numpy()
    train_precision1 = precision_score(train_labels, train_preds)
    train_recall1 = recall_score(train_labels, train_preds)
    train_f11 = f1_score(train_labels, train_preds)
    logger1.log_epoch(epoch + 1, train_loss1, train_acc1, train_precision1, train_recall1, train_f11, test_loss1,
                      test_acc1, test_precision1, test_recall1, test_f11)
    gradients_list.append(gradient1)

    # 客户端2训练
    print("Training Client 2...")
    train_loss2, train_acc2, gradient2 = train(teacher2, student2, optimizer2, optimizer_s2, train_loader2, epoch,
                                               epochs)
    test_loss2, test_acc2, test_precision2, test_recall2, test_f12, tst_fpr2, y_pred2, y_true2 = test(student2,
                                                                                                      test_loader2)
    _, train_logits = student2(next(iter(train_loader2))[0].to(device))
    print(classification_report(y_true2, y_pred2))
    print(confusion_matrix(y_true2, y_pred2))
    print(f"Client 2 - Epoch {epoch + 1}:")
    print(f"Train loss: {train_loss2:.4f}, Train accuracy: {train_acc2:.4f}")
    print(f"Test loss: {test_loss2:.4f}, Test accuracy: {test_acc2:.4f}, "
          f"Test precision: {test_precision2:.4f}, Test recall: {test_recall2:.4f}, "
          f"Test F1: {test_f12:.4f}")
    train_preds = train_logits.argmax(1).cpu().numpy()
    train_labels = next(iter(train_loader2))[1].numpy()
    train_precision2 = precision_score(train_labels, train_preds)
    train_recall2 = recall_score(train_labels, train_preds)
    train_f12 = f1_score(train_labels, train_preds)
    logger2.log_epoch(epoch + 1, train_loss2, train_acc2, train_precision2, train_recall2, train_f12,
                      test_loss2, test_acc2, test_precision2, test_recall2, test_f12)
    gradients_list.append(gradient2)

    # 客户端3训练
    print("Training Client 3...")
    train_loss3, train_acc3, gradient3 = train(teacher3, student3, optimizer3, optimizer_s3, train_loader3, epoch,
                                               epochs)
    test_loss3, test_acc3, test_precision3, test_recall3, test_f13, tst_fpr3, y_pred3, y_true3 = test(student3,
                                                                                                      test_loader3)
    _, train_logits = student3(next(iter(train_loader3))[0].to(device))
    print(classification_report(y_true3, y_pred3))
    print(confusion_matrix(y_true3, y_pred3))
    print(f"Client 3 - Epoch {epoch + 1}:")
    print(f"Train loss: {train_loss3:.4f}, Train accuracy: {train_acc3:.4f}")
    print(f"Test loss: {test_loss3:.4f}, Test accuracy: {test_acc3:.4f}, "
          f"Test precision: {test_precision3:.4f}, Test recall: {test_recall3:.4f}, "
          f"Test F1: {test_f13:.4f}")
    train_preds = train_logits.argmax(1).cpu().numpy()
    train_labels = next(iter(train_loader3))[1].numpy()
    train_precision3 = precision_score(train_labels, train_preds)
    train_recall3 = recall_score(train_labels, train_preds)
    train_f13 = f1_score(train_labels, train_preds)
    logger3.log_epoch(epoch + 1, train_loss3, train_acc3, train_precision3, train_recall3, train_f13,
                      test_loss3, test_acc3, test_precision3, test_recall3, test_f13)
    gradients_list.append(gradient3)

    # 客户端4训练
    print("Training Client 4...")
    train_loss4, train_acc4, gradient4 = train(teacher4, student4, optimizer4, optimizer_s4, train_loader4, epoch,
                                               epochs)
    test_loss4, test_acc4, test_precision4, test_recall4, test_f14, tst_fpr4, y_pred4, y_true4 = test(student4,
                                                                                                      test_loader4)
    _, train_logits = student4(next(iter(train_loader4))[0].to(device))
    print(classification_report(y_true4, y_pred4))
    print(confusion_matrix(y_true4, y_pred4))
    print(f"Client 4 - Epoch {epoch + 1}:")
    print(f"Train loss: {train_loss4:.4f}, Train accuracy: {train_acc4:.4f}")
    print(f"Test loss: {test_loss4:.4f}, Test accuracy: {test_acc4:.4f}, "
          f"Test precision: {test_precision4:.4f}, Test recall: {test_recall4:.4f}, "
          f"Test F1: {test_f14:.4f}")
    train_preds = train_logits.argmax(1).cpu().numpy()
    train_labels = next(iter(train_loader4))[1].numpy()
    train_precision4 = precision_score(train_labels, train_preds)
    train_recall4 = recall_score(train_labels, train_preds)
    train_f14 = f1_score(train_labels, train_preds)
    logger4.log_epoch(epoch + 1, train_loss4, train_acc4, train_precision4, train_recall4, train_f14,
                      test_loss4, test_acc4, test_precision4, test_recall4, test_f14)
    gradients_list.append(gradient4)

    # 客户端5训练
    print("Training Client 5...")
    train_loss5, train_acc5, gradient5 = train(teacher5, student5, optimizer5, optimizer_s5, train_loader5, epoch,
                                               epochs)
    test_loss5, test_acc5, test_precision5, test_recall5, test_f15, tst_fpr5, y_pred5, y_true5 = test(student5,
                                                                                                      test_loader5)
    _, train_logits = student5(next(iter(train_loader5))[0].to(device))
    print(classification_report(y_true5, y_pred5))
    print(confusion_matrix(y_true5, y_pred5))
    print(f"Client 5 - Epoch {epoch + 1}:")
    print(f"Train loss: {train_loss5:.4f}, Train accuracy: {train_acc5:.4f}")
    print(f"Test loss: {test_loss5:.4f}, Test accuracy: {test_acc5:.4f}, "
          f"Test precision: {test_precision5:.4f}, Test recall: {test_recall5:.4f}, "
          f"Test F1: {test_f15:.4f}")
    train_preds = train_logits.argmax(1).cpu().numpy()
    train_labels = next(iter(train_loader5))[1].numpy()
    train_precision5 = precision_score(train_labels, train_preds)
    train_recall5 = recall_score(train_labels, train_preds)
    train_f15 = f1_score(train_labels, train_preds)
    logger5.log_epoch(epoch + 1, train_loss5, train_acc5, train_precision5, train_recall5, train_f15,
                      test_loss5, test_acc5, test_precision5, test_recall5, test_f15)
    gradients_list.append(gradient5)

    # 客户端6训练
    print("Training Client 6...")
    train_loss6, train_acc6, gradient6 = train(teacher6, student6, optimizer6, optimizer_s6, train_loader6, epoch,
                                               epochs)
    test_loss6, test_acc6, test_precision6, test_recall6, test_f16, tst_fpr6, y_pred6, y_true6 = test(student6,
                                                                                                      test_loader6)
    _, train_logits = student6(next(iter(train_loader6))[0].to(device))
    print(classification_report(y_true6, y_pred6))
    print(confusion_matrix(y_true6, y_pred6))
    print(f"Client 6 - Epoch {epoch + 1}:")
    print(f"Train loss: {train_loss6:.4f}, Train accuracy: {train_acc6:.4f}")
    print(f"Test loss: {test_loss6:.4f}, Test accuracy: {test_acc6:.4f}, "
          f"Test precision: {test_precision6:.4f}, Test recall: {test_recall6:.4f}, "
          f"Test F1: {test_f16:.4f}")
    train_preds = train_logits.argmax(1).cpu().numpy()
    train_labels = next(iter(train_loader6))[1].numpy()
    train_precision6 = precision_score(train_labels, train_preds)
    train_recall6 = recall_score(train_labels, train_preds)
    train_f16 = f1_score(train_labels, train_preds)
    logger6.log_epoch(epoch + 1, train_loss6, train_acc6, train_precision6, train_recall6, train_f16,
                      test_loss6, test_acc6, test_precision6, test_recall6, test_f16)
    gradients_list.append(gradient6)

    # ===== 评估全局模型 =====
    global_test_loss, global_test_acc, global_test_precision, global_test_recall, global_test_f1, _, _, _ = test(
        global_model, test_loader1)
    subset_train_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(train_data1, indices=range(0, len(train_data1), 10)),  # 取1/10样本
        batch_size=batch_size
    )
    global_train_loss, global_train_acc, global_train_precision, global_train_recall, global_train_f1, _, _, _ = test(
        global_model, subset_train_loader)
    global_logger.log_epoch(
        epoch + 1,
        global_train_loss, global_train_acc, global_train_precision, global_train_recall, global_train_f1,
        global_test_loss, global_test_acc, global_test_precision, global_test_recall, global_test_f1
    )

    # 记录指标
    test_acc_list[0].append(test_acc1)
    test_acc_list[1].append(test_acc2)
    test_acc_list[2].append(test_acc3)
    test_acc_list[3].append(test_acc4)
    test_acc_list[4].append(test_acc5)
    test_acc_list[5].append(test_acc6)

    f1_list[0].append(test_f11)
    f1_list[1].append(test_f12)
    f1_list[2].append(test_f13)
    f1_list[3].append(test_f14)
    f1_list[4].append(test_f15)
    f1_list[5].append(test_f16)

    train_acc_list[0].append(train_acc1)
    train_acc_list[1].append(train_acc2)
    train_acc_list[2].append(train_acc3)
    train_acc_list[3].append(train_acc4)
    train_acc_list[4].append(train_acc5)
    train_acc_list[5].append(train_acc6)

    # ===== 模型聚合阶段 =====
    weights = compute_combined_weights(gradients_list, data_sizes)
    avg_gradients = weighted_aggregate_gradients(gradients_list, weights)
    update_global_model(global_model, avg_gradients, learning_rate=0.001)

    # 执行模型聚合
    global_model_state_dict = global_model.state_dict()
    for key in global_model_state_dict:
        global_model_state_dict[key] = (teacher1.state_dict()[key] * data_sizes[0] +
                                        teacher2.state_dict()[key] * data_sizes[1] +
                                        teacher3.state_dict()[key] * data_sizes[2] +
                                        teacher4.state_dict()[key] * data_sizes[3] +
                                        teacher5.state_dict()[key] * data_sizes[4] +
                                        teacher6.state_dict()[key] * data_sizes[5]) / sum(data_sizes)
    global_model.load_state_dict(global_model_state_dict)

    # 更新本地模型
    global_model_state_dict = global_model.state_dict()
    teacher1.load_state_dict(global_model_state_dict)
    teacher2.load_state_dict(global_model_state_dict)
    teacher3.load_state_dict(global_model_state_dict)
    teacher4.load_state_dict(global_model_state_dict)
    teacher5.load_state_dict(global_model_state_dict)
    teacher6.load_state_dict(global_model_state_dict)

    # ===== 保存模型 =====
    current_avg_f1 = (test_f11 + test_f12 + test_f13 + test_f14 + test_f15 + test_f16) / 6
    if current_avg_f1 > best_test_f1:
        best_test_f1 = current_avg_f1
        save_models(epoch)
        print(f"保存最佳模型在epoch {epoch + 1}, 平均F1: {current_avg_f1:.4f}")

# 保存最佳指标
logger1.save_best_metrics()
logger2.save_best_metrics()
logger3.save_best_metrics()
logger4.save_best_metrics()
logger5.save_best_metrics()
logger6.save_best_metrics()
global_logger.save_best_metrics()

# 绘制结果
plt.figure(figsize=(15, 10))

# Accuracy plot
plt.subplot(2, 1, 1)
colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown']
for i in range(6):
    plt.plot(range(1, epochs + 1), train_acc_list[i], label=f'Client {i + 1} Train Accuracy', linestyle='--',
             color=colors[i])
    plt.plot(range(1, epochs + 1), test_acc_list[i], label=f'Client {i + 1} Test Accuracy', color=colors[i])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Test Accuracy')
plt.legend()

# F1 Score plot
plt.subplot(2, 1, 2)
for i in range(6):
    plt.plot(range(1, epochs + 1), f1_list[i], label=f'Client {i + 1} F1 Score', color=colors[i])
plt.xlabel('Epoch')
plt.ylabel('F1 Score')
plt.title('Test F1 Scores')
plt.legend()
plt.tight_layout()
plt.show()

print("训练完成！")