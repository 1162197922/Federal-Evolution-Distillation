import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE, RandomOverSampler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import recall_score, accuracy_score, precision_score
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
import csv
import os
from collections import defaultdict
from sklearn.metrics import roc_curve, accuracy_score, precision_score, recall_score, f1_score
import warnings
import torch.nn.functional as F
from tqdm import tqdm

# 创建结果保存目录
save_dir = './result/UNSW(进化蒸馏)'
os.makedirs(save_dir, exist_ok=True)


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
print(f"Using device: {device}")
torch.backends.cudnn.benchmark = True

# 加载6个客户端数据
df1 = pd.read_csv('data/UNSW-NB15/UNSW_NB15_client_1_train.csv')
df2 = pd.read_csv('data/UNSW-NB15/UNSW_NB15_client_2_train.csv')
df3 = pd.read_csv('data/UNSW-NB15/UNSW_NB15_client_3_train.csv')
df4 = pd.read_csv('data/UNSW-NB15/UNSW_NB15_client_4_train.csv')
df5 = pd.read_csv('data/UNSW-NB15/UNSW_NB15_client_5_train.csv')
df6 = pd.read_csv('data/UNSW-NB15/UNSW_NB15_client_6_train.csv')
df_test = pd.read_csv('data/UNSW-NB15/UNSW_NB15_global_test.csv')

label_encoder = LabelEncoder()

# 处理分类列
categorical_cols = ['proto', 'service', 'state', 'attack_cat']
for col in categorical_cols:
    for df in [df1, df2, df3, df4, df5, df6, df_test]:
        if col in df.columns:
            df[col] = label_encoder.fit_transform(df[col])


# 数据预处理函数
def preprocess_data(df_train, df_test):
    scaler = StandardScaler()

    X_train = df_train.iloc[:, :-1]
    X_train = X_train * 255
    y_train = df_train.iloc[:, -1]
    y_train = y_train.astype(float).astype(int)
    y_train = [1 if x == 1 else 0 for x in y_train]
    y_train = pd.DataFrame(y_train)

    X_test = df_test.iloc[:, :-1]
    X_test = X_test * 255
    y_test = df_test.iloc[:, -1]
    y_test = y_test.astype(float).astype(int)
    y_test = [1 if x == 1 else 0 for x in y_test]
    y_test = pd.DataFrame(y_test)

    # 处理分类特征
    for col in categorical_cols:
        if col in X_train.columns:
            le = LabelEncoder()
            X_train[col] = le.fit_transform(X_train[col].astype(str))
            X_test[col] = le.transform(X_test[col].astype(str))

    # 过采样
    print(f"过采样前分布: {y_train.value_counts()}")
    clf = RandomOverSampler()
    X_train, y_train = clf.fit_resample(X_train, y_train)
    print(f"过采样后分布: {y_train.value_counts()}")

    return X_train, y_train, X_test, y_test


# 预处理所有客户端数据
X_train1, y_train1, X_test1, y_test1 = preprocess_data(df1, df_test)
X_train2, y_train2, X_test2, y_test2 = preprocess_data(df2, df_test)
X_train3, y_train3, X_test3, y_test3 = preprocess_data(df3, df_test)
X_train4, y_train4, X_test4, y_test4 = preprocess_data(df4, df_test)
X_train5, y_train5, X_test5, y_test5 = preprocess_data(df5, df_test)
X_train6, y_train6, X_test6, y_test6 = preprocess_data(df6, df_test)

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

train_data1 = LoadData(X_train1, y_train1)
test_data1 = LoadData(X_test1, y_test1)
train_loader1 = Data.DataLoader(train_data1, batch_size=batch_size, shuffle=True)
test_loader1 = Data.DataLoader(test_data1, batch_size=batch_size, shuffle=True)

train_data2 = LoadData(X_train2, y_train2)
test_data2 = LoadData(X_test2, y_test2)
train_loader2 = Data.DataLoader(train_data2, batch_size=batch_size, shuffle=True)
test_loader2 = Data.DataLoader(test_data2, batch_size=batch_size, shuffle=True)

train_data3 = LoadData(X_train3, y_train3)
test_data3 = LoadData(X_test3, y_test3)
train_loader3 = Data.DataLoader(train_data3, batch_size=batch_size, shuffle=True)
test_loader3 = Data.DataLoader(test_data3, batch_size=batch_size, shuffle=True)

train_data4 = LoadData(X_train4, y_train4)
test_data4 = LoadData(X_test4, y_test4)
train_loader4 = Data.DataLoader(train_data4, batch_size=batch_size, shuffle=True)
test_loader4 = Data.DataLoader(test_data4, batch_size=batch_size, shuffle=True)

train_data5 = LoadData(X_train5, y_train5)
test_data5 = LoadData(X_test5, y_test5)
train_loader5 = Data.DataLoader(train_data5, batch_size=batch_size, shuffle=True)
test_loader5 = Data.DataLoader(test_data5, batch_size=batch_size, shuffle=True)

train_data6 = LoadData(X_train6, y_train6)
test_data6 = LoadData(X_test6, y_test6)
train_loader6 = Data.DataLoader(train_data6, batch_size=batch_size, shuffle=True)
test_loader6 = Data.DataLoader(test_data6, batch_size=batch_size, shuffle=True)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.2, stride=1):
        super().__init__()
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
        self.initial_conv = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.blocks = nn.ModuleList()
        current_channels = hidden_dim
        for i in range(num_blocks):
            next_channels = min(current_channels * 2, 64)
            stride = 2 if i == 0 else 1
            self.blocks.append(ResidualBlock(current_channels, next_channels, dropout_rate=0.1, stride=stride))
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


# 模型参数
teacher_num_blocks = 4
student_num_blocks = 4
input_dim = 1
teacher_hidden_dim = 8
student_hidden_dim = 8
output_dim = 2

# 初始化全局模型
global_model = Teacher(teacher_num_blocks, input_dim, teacher_hidden_dim, output_dim).to(device)

# 创建模型实例
teacher1 = Teacher(teacher_num_blocks, input_dim, teacher_hidden_dim, output_dim).to(device)
teacher2 = Teacher(teacher_num_blocks, input_dim, teacher_hidden_dim, output_dim).to(device)
teacher3 = Teacher(teacher_num_blocks, input_dim, teacher_hidden_dim, output_dim).to(device)
teacher4 = Teacher(teacher_num_blocks, input_dim, teacher_hidden_dim, output_dim).to(device)
teacher5 = Teacher(teacher_num_blocks, input_dim, teacher_hidden_dim, output_dim).to(device)
teacher6 = Teacher(teacher_num_blocks, input_dim, teacher_hidden_dim, output_dim).to(device)

weight = torch.tensor([1.0, 3.0]).to(device)
criterion = nn.CrossEntropyLoss(weight=weight)

student1 = Student(student_num_blocks, input_dim, student_hidden_dim, output_dim).to(device)
student2 = Student(student_num_blocks, input_dim, student_hidden_dim, output_dim).to(device)
student3 = Student(student_num_blocks, input_dim, student_hidden_dim, output_dim).to(device)
student4 = Student(student_num_blocks, input_dim, student_hidden_dim, output_dim).to(device)
student5 = Student(student_num_blocks, input_dim, student_hidden_dim, output_dim).to(device)
student6 = Student(student_num_blocks, input_dim, student_hidden_dim, output_dim).to(device)

# 优化器
optimizer1 = optim.Adam(teacher1.parameters(), lr=0.0001, weight_decay=0.0001)
optimizer2 = optim.Adam(teacher2.parameters(), lr=0.0001, weight_decay=0.0001)
optimizer3 = optim.Adam(teacher3.parameters(), lr=0.0001, weight_decay=0.0001)
optimizer4 = optim.Adam(teacher4.parameters(), lr=0.0001, weight_decay=0.0001)
optimizer5 = optim.Adam(teacher5.parameters(), lr=0.0001, weight_decay=0.0001)
optimizer6 = optim.Adam(teacher6.parameters(), lr=0.0001, weight_decay=0.0001)

optimizer_s1 = optim.Adam(student1.parameters(), lr=0.0001, weight_decay=0.0001)
optimizer_s2 = optim.Adam(student2.parameters(), lr=0.0001, weight_decay=0.0001)
optimizer_s3 = optim.Adam(student3.parameters(), lr=0.0001, weight_decay=0.0001)
optimizer_s4 = optim.Adam(student4.parameters(), lr=0.0001, weight_decay=0.0001)
optimizer_s5 = optim.Adam(student5.parameters(), lr=0.0001, weight_decay=0.0001)
optimizer_s6 = optim.Adam(student6.parameters(), lr=0.0001, weight_decay=0.0001)


def distillation_loss(student_logits, teacher_logits, temperature):
    return nn.KLDivLoss(reduction='batchmean')(
        F.log_softmax(student_logits / temperature, dim=1),
        F.softmax(teacher_logits / temperature, dim=1)
    )


def feature_loss(student_features, teacher_features):
    loss = 0
    for i, (s_feat, t_feat) in enumerate(zip(student_features, teacher_features[-len(student_features):])):
        if s_feat.size(1) != t_feat.size(1):
            adjust_conv = nn.Conv2d(s_feat.size(1), t_feat.size(1), kernel_size=1).to(s_feat.device)
            s_feat = adjust_conv(s_feat)

        if s_feat.size()[2:] != t_feat.size()[2:]:
            s_feat = F.interpolate(s_feat, size=t_feat.size()[2:], mode='bilinear', align_corners=False)

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

        with torch.no_grad():
            teacher_features, teacher_logits = teacher(data)

        student_features, student_logits = student(data)

        with torch.no_grad():
            cos_sim = F.cosine_similarity(student_logits, teacher_logits, dim=1)
            dynamic_temp = current_temp * (1 + 0.5 * torch.mean(cos_sim).item())

        cls_loss = classification_loss(student_logits, labels)
        kd_loss = distillation_loss(student_logits, teacher_logits, dynamic_temp)
        feat_loss = feature_loss(student_features, teacher_features)

        total_loss = cls_loss + 0.6 * kd_loss + 0.4 * feat_loss

        student_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), max_grad_norm)
        student_optimizer.step()
        gradients = [param.grad.clone() for param in student.parameters()]

        with torch.no_grad():
            teacher_loss = classification_loss(teacher_logits, labels)

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
    weight = torch.tensor([1.0, 1.5]).to(device)

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
    test_precision = precision_score(y_true, y_pred, zero_division=0)
    test_recall = recall_score(y_true, y_pred, zero_division=0)
    test_f1 = f1_score(y_true, y_pred, zero_division=0)
    fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=0, drop_intermediate=False)
    return test_loss, test_acc, test_precision, test_recall, test_f1, fpr, y_pred, y_true


def compute_sparsity(gradient):
    non_zero_count = torch.count_nonzero(gradient).item()
    total_elements = gradient.numel()
    sparsity = 1 - (non_zero_count / total_elements)
    return sparsity


def compute_combined_weights(gradients_list, data_sizes):
    sparsity_weights = []
    for gradients in gradients_list:
        client_sparsity = 0
        for gradient in gradients:
            client_sparsity += compute_sparsity(gradient)
        client_sparsity /= len(gradients)
        sparsity_weight = 1 / (client_sparsity + 1e-10)
        sparsity_weights.append(sparsity_weight)

    combined_weights = torch.tensor(sparsity_weights) * torch.tensor(data_sizes)
    combined_weights = combined_weights / combined_weights.sum()
    return combined_weights.tolist()


def weighted_aggregate_gradients(gradients_list, weights):
    if not gradients_list:
        raise ValueError("梯度列表为空，无法聚合！")

    avg_gradients = []
    for i in range(len(gradients_list[0])):
        layer_gradients = [gradients[i] for gradients in gradients_list]
        weighted_layer_gradient = torch.zeros_like(layer_gradients[0])
        for grad, weight in zip(layer_gradients, weights):
            weighted_layer_gradient += grad * weight
        avg_gradients.append(weighted_layer_gradient)

    return avg_gradients


def update_global_model(global_model, avg_gradients, learning_rate=0.001):
    optimizer = optim.Adam(global_model.parameters(), lr=learning_rate)
    optimizer.zero_grad()

    for param, grad in zip(global_model.parameters(), avg_gradients):
        param.grad = grad

    optimizer.step()


# 训练参数
epochs = 50
data_sizes = [44980, 77440, 83218, 39246, 32450, 75371]  # 6个客户端的数据量
num_clients = 6

# 初始化全局模型状态
global_model_state_dict = global_model.state_dict()
teacher1.load_state_dict(global_model_state_dict)
teacher2.load_state_dict(global_model_state_dict)
teacher3.load_state_dict(global_model_state_dict)
teacher4.load_state_dict(global_model_state_dict)
teacher5.load_state_dict(global_model_state_dict)
teacher6.load_state_dict(global_model_state_dict)

# 记录列表
train_acc_lists = [[] for _ in range(6)]
test_acc_lists = [[] for _ in range(6)]
f1_lists = [[] for _ in range(6)]
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


def create_summary(loggers):
    with open(os.path.join(save_dir, 'summary.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Client', 'BestEpoch', 'TestAcc', 'TestF1', 'TestPrecision', 'TestRecall',
                         'TrainAcc', 'TrainF1', 'TrainPrecision', 'TrainRecall'])
        for i, logger in enumerate(loggers, 1):
            writer.writerow([
                f'Client{i}',
                logger.best_metrics['epoch'],
                logger.best_metrics['test_acc'],
                logger.best_metrics['test_f1'],
                logger.best_metrics['test_precision'],
                logger.best_metrics['test_recall'],
                logger.best_metrics['train_acc'],
                logger.best_metrics['train_f1'],
                logger.best_metrics['train_precision'],
                logger.best_metrics['train_recall']
            ])
        writer.writerow([
            'GlobalModel',
            global_logger.best_metrics['epoch'],
            global_logger.best_metrics['test_acc'],
            global_logger.best_metrics['test_f1'],
            global_logger.best_metrics['test_precision'],
            global_logger.best_metrics['test_recall'],
            global_logger.best_metrics['train_acc'],
            global_logger.best_metrics['train_f1'],
            global_logger.best_metrics['train_precision'],
            global_logger.best_metrics['train_recall']
        ])


# 训练循环
for epoch in range(epochs):
    print(f"\n=== Epoch {epoch + 1}/{epochs} ===")

    gradients_list = []
    client_results = []

    # 客户端1-6训练
    for client_idx, (teacher, student, train_loader, test_loader, optimizer_t, optimizer_s, logger) in enumerate([
        (teacher1, student1, train_loader1, test_loader1, optimizer1, optimizer_s1, logger1),
        (teacher2, student2, train_loader2, test_loader2, optimizer2, optimizer_s2, logger2),
        (teacher3, student3, train_loader3, test_loader3, optimizer3, optimizer_s3, logger3),
        (teacher4, student4, train_loader4, test_loader4, optimizer4, optimizer_s4, logger4),
        (teacher5, student5, train_loader5, test_loader5, optimizer5, optimizer_s5, logger5),
        (teacher6, student6, train_loader6, test_loader6, optimizer6, optimizer_s6, logger6)
    ], 1):
        train_loss, train_acc, gradient = train(teacher, student, optimizer_t, optimizer_s, train_loader, epoch, epochs)
        test_loss, test_acc, test_precision, test_recall, test_f1, _, y_pred, y_true = test(student, test_loader)
        _, train_logits = student(next(iter(train_loader))[0].to(device))

        print(f"\nClient {client_idx} ---------- Epoch {epoch + 1}:")
        print(f"Train loss: {train_loss:.4f}, Train accuracy: {train_acc:.4f}")
        print(f"Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.4f}, "
              f"Test precision: {test_precision:.4f}, Test recall: {test_recall:.4f}, "
              f"Test F1: {test_f1:.4f}")

        train_preds = train_logits.argmax(1).cpu().numpy()
        train_labels = next(iter(train_loader))[1].numpy()
        train_precision = precision_score(train_labels, train_preds, zero_division=0)
        train_recall = recall_score(train_labels, train_preds, zero_division=0)
        train_f1 = f1_score(train_labels, train_preds, zero_division=0)

        logger.log_epoch(epoch + 1, train_loss, train_acc, train_precision, train_recall, train_f1,
                         test_loss, test_acc, test_precision, test_recall, test_f1)

        gradients_list.append(gradient)
        client_results.append((train_acc, test_acc, test_f1))

        train_acc_lists[client_idx - 1].append(train_acc)
        test_acc_lists[client_idx - 1].append(test_acc)
        f1_lists[client_idx - 1].append(test_f1)

    # 评估全局模型
    global_test_loss, global_test_acc, global_test_precision, global_test_recall, global_test_f1, _, _, _ = test(
        global_model, test_loader1)
    subset_train_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(train_data1, indices=range(0, len(train_data1), 10)),
        batch_size=batch_size
    )
    global_train_loss, global_train_acc, global_train_precision, global_train_recall, global_train_f1, _, _, _ = test(
        global_model, subset_train_loader)

    global_logger.log_epoch(
        epoch + 1,
        global_train_loss, global_train_acc, global_train_precision, global_train_recall, global_train_f1,
        global_test_loss, global_test_acc, global_test_precision, global_test_recall, global_test_f1
    )

    # 模型聚合
    weights = compute_combined_weights(gradients_list, data_sizes)
    avg_gradients = weighted_aggregate_gradients(gradients_list, weights)
    update_global_model(global_model, avg_gradients, learning_rate=0.001)

    # 更新本地教师模型
    global_model_state_dict = global_model.state_dict()
    teacher1.load_state_dict(global_model_state_dict)
    teacher2.load_state_dict(global_model_state_dict)
    teacher3.load_state_dict(global_model_state_dict)
    teacher4.load_state_dict(global_model_state_dict)
    teacher5.load_state_dict(global_model_state_dict)
    teacher6.load_state_dict(global_model_state_dict)

    # 保存最佳模型
    current_avg_f1 = sum(result[2] for result in client_results) / len(client_results)
    if current_avg_f1 > best_test_f1:
        best_test_f1 = current_avg_f1
        save_models(epoch)
        print(f"保存最佳模型在epoch {epoch + 1}, 平均F1: {current_avg_f1:.4f}")

# 保存最终结果
create_summary([logger1, logger2, logger3, logger4, logger5, logger6])
logger1.save_best_metrics()
logger2.save_best_metrics()
logger3.save_best_metrics()
logger4.save_best_metrics()
logger5.save_best_metrics()
logger6.save_best_metrics()
global_logger.save_best_metrics()

# 绘制结果
plt.figure(figsize=(15, 10))
colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown']

# Accuracy plot
plt.subplot(2, 1, 1)
for i in range(6):
    plt.plot(range(1, epochs + 1), train_acc_lists[i], label=f'Client {i + 1} Train Accuracy', linestyle='--',
             color=colors[i])
    plt.plot(range(1, epochs + 1), test_acc_lists[i], label=f'Client {i + 1} Test Accuracy', color=colors[i])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Test Accuracy')
plt.legend()

# F1 Score plot
plt.subplot(2, 1, 2)
for i in range(6):
    plt.plot(range(1, epochs + 1), f1_lists[i], label=f'Client {i + 1} F1 Score', color=colors[i])
plt.xlabel('Epoch')
plt.ylabel('F1 Score')
plt.title('Test F1 Scores')
plt.legend()
plt.tight_layout()
plt.show()

print("训练完成！")