import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
#评价函数
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from math import sqrt
import matplotlib.pyplot as plt
from tensorflow.keras import layers
import keras.backend as K
from tensorflow import reduce_sum
from keras.layers import Input,LSTM,Dropout,Dense,Reshape,Reshape,GRU,Conv1D, GlobalMaxPooling1D,MaxPooling1D,Flatten,concatenate,BatchNormalization,Permute
from keras.models import Model
from tensorflow.keras.layers import LSTM,Dense,Permute,Input,Reshape,Multiply,dot
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense, Activation
import tensorflow as tf
import keras
import networkx as nx
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import numpy as np
import itertools
import math
import random
from joblib import Parallel, delayed
from tqdm import trange
import warnings
warnings.filterwarnings('ignore')


def mape(y_true, y_pred):
    """
    参数:
    y_true -- 测试集目标真实值
    y_pred -- 测试集目标预测值

    返回:
    mape -- MAPE 评价指标
    """

    n = len(y_true)
    mape = sum(np.abs((y_true - y_pred) / y_true)) / n * 100
    return mape


def easy_result(y_train, y_train_predict, train_index, model_index, col_index):
    # 画图进行展示
    plt.figure(figsize=(10, 5))
    plt.plot(y_train[:])
    plt.plot(y_train_predict[:])
    plt.legend(('real', 'predict'), fontsize='15')
    plt.title("%s Data" % train_index, fontsize='20')  # 添加标题
    plt.show()
    print('\n')

    plot_begin, plot_end = min(min(y_train), min(y_train_predict)), max(max(y_train), max(y_train_predict))
    plot_x = np.linspace(plot_begin, plot_end, 10)
    plt.figure(figsize=(5, 5))
    plt.plot(plot_x, plot_x)
    plt.plot(y_train, y_train_predict, 'o')
    plt.title("%s Data" % train_index, fontsize='20')  # 添加标题
    plt.show()

    # 输出结果
    print('%s上的MAE/RMSE/MAPE/R^2' % train_index)
    print(mean_absolute_error(y_train, y_train_predict))
    print(np.sqrt(mean_squared_error(y_train, y_train_predict)))
    print(mape(y_train, y_train_predict))
    print(r2_score(y_train, y_train_predict))

    pred_data = np.vstack([y_train, y_train_predict])
    pred_data = pd.DataFrame(pred_data).T
    return mean_absolute_error(y_train, y_train_predict), np.sqrt(mean_squared_error(y_train, y_train_predict)), mape(
        y_train, y_train_predict), r2_score(y_train, y_train_predict)


data_1 = df = pd.read_csv('C:/Users/DELL/Desktop/画图/1.csv').iloc[:, 2:6]
data_21 = data_1.iloc[:, [1, 2, 3]]
data_22 = data_1.iloc[:, [0, 2, 3]]
data_23 = data_21.iloc[:, 0:1]
data_24 = data_22.iloc[:, 0:1]
data_3 = data_1.iloc[:, 0:2]
print('Before Processing\n', data_21.head(6))
print('Before Processing\n', data_22.head(6))
test_ratio = 0.2

L = []
win = [0, 1, 2, 3, 4, 5]
for i in range(len(df)):
    L.append(i)
x1 = []
for i in range(len(L)):
    for j in win:
        x1.append([c for c in range(i, i + j + 1, 1)])
y = []
for i in x1:
    x = []
    if len(i) >= 2:
        y.append((x1.index(i), i[0], i[-1]))
    elif len(i) == 1:
        y.append((x1.index(i), i[0], i[0]))
z = []
for i in y:
    for j in y:
        if i[2] + 1 == j[1]:
            z.append((i[0], j[0]))
Lon_in = []
Lat_in = []
Lon_out = []
Lat_out = []
for i in y:
    # Lon_in.append(np.mean(data_21.iloc[i[1]:i[2]+1,0:1].values))
    Lon_in.append([np.mean(data_21.iloc[i[1]:i[2] + 1, 0:1].values), np.mean(data_21.iloc[i[1]:i[2] + 1, 1:2].values),
                   np.mean(data_21.iloc[i[1]:i[2] + 1, 2:3].values)])
for i in y:
    # Lat_in.append(np.mean(data_22.iloc[i[1]:i[2]+1,0:1].values))
    Lat_in.append([np.mean(data_22.iloc[i[1]:i[2] + 1, 0:1].values), np.mean(data_22.iloc[i[1]:i[2] + 1, 1:2].values),
                   np.mean(data_22.iloc[i[1]:i[2] + 1, 2:3].values)])
for i in y:
    Lon_out.append(np.mean(data_23.iloc[i[1]:i[2] + 1, 0:1].values))
for i in y:
    Lat_out.append(np.mean(data_24.iloc[i[1]:i[2] + 1, 0:1].values))
Lon_in = pd.DataFrame(Lon_in)
Lat_in = pd.DataFrame(Lat_in)
Lon_out = pd.DataFrame(Lon_out)
Lat_out = pd.DataFrame(Lat_out)
kk1 = []
kk2 = []
for i in y:
    kk1.append(np.mean(data_3.iloc[i[1]:i[2] + 1, 1:2].values))
for i in y:
    kk2.append(np.mean(data_3.iloc[i[1]:i[2] + 1, 0:1].values))
mm_x = MinMaxScaler()
mm_y = MinMaxScaler()
mm_z1 = MinMaxScaler()
mm_z2 = MinMaxScaler()
Lon_in1 = mm_x.fit_transform(Lon_in)
Lat_in1 = mm_y.fit_transform(Lat_in)
Lon_out1 = mm_z1.fit_transform(Lon_out)
Lat_out1 = mm_z2.fit_transform(Lat_out)
np.savetxt('C:/Users/DELL/Desktop/画图/test1_2_5.txt', z, fmt='%d')
G = nx.read_edgelist('C:/Users/DELL/Desktop/画图/test1_2_5.txt', create_using=nx.DiGraph(), nodetype=None,
                     data=[('weight', int)])


def partition_num(num, workers):
    if num % workers == 0:
        return [num // workers] * workers
    else:
        return [num // workers] * workers + [num % workers]


class RandomWalker:
    def __init__(self, G):

        self.G = G

    def deepwalk_walk(self, walk_length, start_node):

        walk = [start_node]

        while len(walk) < walk_length:
            cur = walk[-1]  # 当前序列的最后一个值
            cur_nbrs = list(self.G.neighbors(cur))  # 获取所有的邻居节点
            if len(cur_nbrs) > 0:
                walk.append(random.choice(cur_nbrs))  # 随机选择一个邻居节点加入队列
            else:
                break
        return walk

    def simulate_walks(self, num_walks, walk_length, workers=1, verbose=0):

        G = self.G

        nodes = list(G.nodes())

        results = Parallel(n_jobs=workers, verbose=verbose, )(
            delayed(self._simulate_walks)(nodes, num, walk_length) for num in
            partition_num(num_walks, workers))

        walks = list(itertools.chain(*results))

        return walks

    def _simulate_walks(self, nodes, num_walks, walk_length, ):
        walks = []
        for _ in range(num_walks):
            random.shuffle(nodes)
            for v in nodes:
                walks.append(self.deepwalk_walk(
                    walk_length=walk_length, start_node=v))

        return walks

num_walks = 1
walk_length = 15
workers =1
rw = RandomWalker(G)
sentences = rw.simulate_walks(num_walks=num_walks, walk_length=walk_length, workers=workers, verbose=1)
L = []
for i in sentences:
    if len(i) == walk_length:
        L.append(i)
xxx1 = []
xxx2 = []
xxx3 = []
xxx4 = []
#yyy = []
for i in L:
    print(i)
    lon = []
    lat = []
    lon1 = []
    lat1 = []
    for j in i:
        lon.append(Lon_in1[eval(j)])
        lat.append(Lat_in1[eval(j)])
        lon1.append(Lon_out1[eval(j)])
        lat1.append(Lat_out1[eval(j)])
    xxx1.append(lon)
    xxx2.append(lat)
    xxx3.append(lon1)
    xxx4.append(lat1)
ppp1 = []
ppp2 = []
for i in L:
    print(i)
    o = []
    k = []
    for j in i:
        o.append(kk1[eval(j)])
        k.append(kk2[eval(j)])
    ppp1.append(o)
    ppp2.append(k)
cut = round(0.2* len(xxx1))
lstm_input = []
lstm_output = []
xxx1 = np.array(xxx1)
xxx3 = np.array(xxx3)
for i in range(len(xxx1)):
    lstm_input.append(xxx1[i][0:10])
    lstm_output.append(xxx3[i][10:15])
lstm_input=np.array(lstm_input)
lstm_output=np.array(lstm_output)
x_train,y_train,x_test,y_test=\
lstm_input[:-cut,:],lstm_output[:-cut:],lstm_input[-cut:,:],lstm_output[-cut:]
y_train = y_train.reshape(-1,5)
y_test=y_test.reshape(-1,5)
print('x_train.shape',x_train.shape)
print('x_test.shape',x_test.shape)
print('y_train.shape',y_train.shape)
print('y_test.shape',y_test.shape)
cut = round(0.2* len(xxx2))
lstm_input = []
lstm_output = []
xxx2 = np.array(xxx2)
xxx4 = np.array(xxx4)
for i in range(len(xxx1)):
    lstm_input.append(xxx2[i][0:10])
    lstm_output.append(xxx4[i][10:15])
lstm_input=np.array(lstm_input)
lstm_output=np.array(lstm_output)
x_train1,y_train1,x_test1,y_test1=\
lstm_input[:-cut,:],lstm_output[:-cut:],lstm_input[-cut:,:],lstm_output[-cut:]
y_train1 = y_train1.reshape(-1,5)
y_test1=y_test1.reshape(-1,5)
print('x_train1.shape',x_train1.shape)
print('x_test1.shape',x_test1.shape)
print('y_train1.shape',y_train1.shape)
print('y_test1.shape',y_test1.shape)

def  attention_3d_block(inputs,SINGLE_ATTENTION_VECTOR  =  False):
    input_dim =  int(inputs.shape[1]) # shape = (batch_size, time_steps, input_dim)
    a = Permute((2, 1))(inputs) # shape = (batch_size, input_dim, time_steps)
    #a = Reshape((input_dim, 3))(a) # this line is not useful. It's just to know which dimension is what.
    a_probs = tf.keras.layers.Dense(10, activation='softmax', name='attention_vec')(a)# 为了让输出的维数和时间序列数相同（这样才能生成各个时间点的注意力值）
    #a_probs= tf.sigmoid(a)
    #a_probs= layers.Softmax(a)
    #a_probs = Permute((2, 1), name='attention_vec')(a) # shape = (batch_size, time_steps, input_dim)
    output_attention_mul = Multiply()([a_probs,a]) #把注意力值和输入按位相乘，权重乘以输入
    return output_attention_mul
def  attention_3d_block1(inputs,SINGLE_ATTENTION_VECTOR  =  False):
    input_dim =  int(inputs.shape[1]) # shape = (batch_size, time_steps, input_dim)
    a = Permute((2, 1))(inputs) # shape = (batch_size, input_dim, time_steps)
    #a = Reshape((input_dim, 10))(a) # this line is not useful. It's just to know which dimension is what.
    a = tf.keras.layers.Dense(3, activation='softmax')(a)# 为了让输出的维数和时间序列数相同（这样才能生成各个时间点的注意力值）
    #a_probs= layers.ReLU(a)
    #a_probs= layers.Softmax(a)
    a_probs = Permute((2, 1), name='attention_vec1')(a) # shape = (batch_size, time_steps, input_dim)
    #output_attention_mul = K.batch_dot(a_probs,inputs)
    output_attention_mul = tf.keras.layers.Dot(axes=(2, 2))([inputs, a_probs]) #把注意力值和输入按位相乘，权重乘以输入
    output_attention_mul = reduce_sum(output_attention_mul,axis=1)
    return output_attention_mul
def  model_attention_applied_before_lstm(inputs):
    attention_mul = attention_3d_block(inputs)
    #attention_mul = LSTM(10, return_sequences=True)(attention_mul)
    attention_mul = LSTM(30, return_sequences=True)(attention_mul)
    attention_mul = LSTM(20, return_sequences=True)(attention_mul)
    #attention_mul = LSTM(40, return_sequences=True)(attention_mul)
    #attention_mul = LSTM(50, return_sequences=True)(attention_mul)
    attention_mul = attention_3d_block1(attention_mul)
    #attention_mul = Flatten()(attention_mul)
    #output = Dense(1, activation='sigmoid')(attention_mul)
    #attention_mul = LSTM(40, return_sequences=False)(attention_mul)
    #attention_mul = Dropout(0.05)(attention_mul)
    outputs = Dense(5, activation='sigmoid')(attention_mul)
    return outputs
input_data = Input(shape=(10,3))
label_data = model_attention_applied_before_lstm(input_data)
model = Model(input_data,label_data)
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
model.summary()
from tensorflow.keras.utils import plot_model
plot_model(model, to_file='.\model.png') #保存模型结构图片
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=100, verbose=2)
history = model.fit(x_train,
                    y_train,
                      batch_size=64,
                      epochs=10000,
                      verbose=2,
                      validation_split=0.1,
                      shuffle=False,
                      callbacks=[early_stopping])
    #迭代图像
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(len(loss))
plt.plot(epochs_range, loss, label='Train Loss')
plt.plot(epochs_range, val_loss, label='Val Loss')
plt.legend(loc='upper right')
plt.title('Train and Val Loss')
plt.show()

y_test_predict=model.predict(x_test)#预测结果
y_test_predict= mm_z1.inverse_transform(y_test_predict)
y_test_inverse=mm_z1.inverse_transform(y_test)
easy_result(y_test_inverse[:,0], y_test_predict[:,0], 'Train', 'TransF', 'longitude')
easy_result(y_test_inverse[:,1], y_test_predict[:,1], 'Train', 'TransF', 'longitude')
easy_result(y_test_inverse[:,2], y_test_predict[:,2], 'Train', 'TransF', 'longitude')
easy_result(y_test_inverse[:,3], y_test_predict[:,3], 'Train', 'TransF', 'longitude')
easy_result(y_test_inverse[:,4], y_test_predict[:,4], 'Train', 'TransF', 'longitude')

from tensorflow import reduce_sum
from tensorflow.keras import layers
import keras.backend as K
def  attention_3d_block(inputs,SINGLE_ATTENTION_VECTOR  =  False):
    input_dim =  int(inputs.shape[1]) # shape = (batch_size, time_steps, input_dim)
    #a = Permute((2, 1))(inputs) # shape = (batch_size, input_dim, time_steps)
    a = Reshape((input_dim, 3))(inputs) # this line is not useful. It's just to know which dimension is what.
    a_probs = tf.keras.layers.Dense(3, activation='softmax', name='attention_vec')(a)# 为了让输出的维数和时间序列数相同（这样才能生成各个时间点的注意力值）
    #a_probs= tf.sigmoid(a)
    #a_probs= layers.Softmax(a)
    #a_probs = Permute((2, 1), name='attention_vec')(a) # shape = (batch_size, time_steps, input_dim)
    output_attention_mul = Multiply()([inputs, a_probs]) #把注意力值和输入按位相乘，权重乘以输入
    return output_attention_mul
def  attention_3d_block1(inputs,SINGLE_ATTENTION_VECTOR  =  False):
    input_dim =  int(inputs.shape[2]) # shape = (batch_size, time_steps, input_dim)
    a = Permute((2, 1))(inputs) # shape = (batch_size, input_dim, time_steps)
    a = Reshape((input_dim, 10))(a) # this line is not useful. It's just to know which dimension is what.
    a = tf.keras.layers.Dense(10, activation='softmax')(a)# 为了让输出的维数和时间序列数相同（这样才能生成各个时间点的注意力值）
    #a_probs= layers.ReLU(a)
    #a_probs= layers.Softmax(a)
    a_probs = Permute((2, 1), name='attention_vec1')(a) # shape = (batch_size, time_steps, input_dim)
    #output_attention_mul = K.batch_dot(a_probs,inputs)
    output_attention_mul = Multiply()([inputs, a_probs]) #把注意力值和输入按位相乘，权重乘以输入
    output_attention_mul = reduce_sum(output_attention_mul,axis=1)
    return output_attention_mul
def  model_attention_applied_before_lstm(inputs):
    attention_mul = attention_3d_block(inputs)
    attention_mul = LSTM(30, return_sequences=True)(attention_mul)
    attention_mul = LSTM(20, return_sequences=True)(attention_mul)
    #attention_mul = LSTM(20, return_sequences=True)(attention_mul)
    #attention_mul = LSTM(40, return_sequences=True)(attention_mul)
    #attention_mul = LSTM(50, return_sequences=True)(attention_mul)
    attention_mul = attention_3d_block1(attention_mul)
    #attention_mul = Flatten()(attention_mul)
    #output = Dense(1, activation='sigmoid')(attention_mul)
    #attention_mul = LSTM(40, return_sequences=False)(attention_mul)
    #attention_mul = Dropout(0.05)(attention_mul)
    outputs = Dense(5, activation='sigmoid')(attention_mul)
    return outputs
input_data = Input(shape=(10,3))
label_data = model_attention_applied_before_lstm(input_data)
model = Model(input_data,label_data)
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
model.summary()
from tensorflow.keras.utils import plot_model
plot_model(model, to_file='.\model.png') #保存模型结构图片
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=100, verbose=2)
history = model.fit(x_train1,
                    y_train1,
                      batch_size=64,
                      epochs=10000,
                      verbose=2,
                      validation_split=0.1,
                      shuffle=False,
                      callbacks=[early_stopping])
    #迭代图像
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(len(loss))
plt.plot(epochs_range, loss, label='Train Loss')
plt.plot(epochs_range, val_loss, label='Val Loss')
plt.legend(loc='upper right')
plt.title('Train and Val Loss')
plt.show()

y_test_predict1=model.predict(x_test1)#预测结果
y_test_predict1= mm_z2.inverse_transform(y_test_predict1)
y_test_inverse1=mm_z2.inverse_transform(y_test1)
easy_result(y_test_inverse1[:,0], y_test_predict1[:,0], 'Train', 'TransF', 'longitude')
easy_result(y_test_inverse1[:,1], y_test_predict1[:,1], 'Train', 'TransF', 'longitude')
easy_result(y_test_inverse1[:,2], y_test_predict1[:,2], 'Train', 'TransF', 'longitude')
easy_result(y_test_inverse1[:,3], y_test_predict1[:,3], 'Train', 'TransF', 'longitude')

def get_ade(forecasted_trajectory_Lon,forecasted_trajectory_Lat,gt_trajectory_Lon,gt_trajectory_Lat):
    pred_len = forecasted_trajectory_Lon.shape[0] #预测单条轨迹的行数==每个轨迹包含轨迹点的个数
    ade = float(  #单条轨迹中所有轨迹点坐标与真值的欧氏距离的平均值
        sum(
            math.sqrt(
                (forecasted_trajectory_Lon[i] - gt_trajectory_Lon[i]) ** 2
                + (forecasted_trajectory_Lat[i] - gt_trajectory_Lat[i]) ** 2
            )
            for i in range(pred_len)
        )
        / pred_len
    )
    return ade
def get_fde(forecasted_trajectory_Lon,forecasted_trajectory_Lat,gt_trajectory_Lon,gt_trajectory_Lat):

    fde = math.sqrt( #单条轨迹中最后一个轨迹点坐标与真值的欧氏距离
        (forecasted_trajectory_Lon[-1] - gt_trajectory_Lon[-1]) ** 2
        + (forecasted_trajectory_Lat[-1] - gt_trajectory_Lat[-1]) ** 2
    )
    return fde
ade_1 = 0
for index1 in range(len(y_test_predict)):
    ade_1 = ade_1+get_ade(np.array(y_test_predict[index1]),np.array(y_test_predict1[index1]),np.array(y_test_inverse[index1]),np.array(y_test_inverse1[index1]))
fde_1 = 0
for index2 in range(len(y_test_predict)):
    fde_1 = fde_1+get_fde(np.array(y_test_predict[index2]),np.array(y_test_predict1[index2]),np.array(y_test_inverse[index2]),np.array(y_test_inverse1[index2]))
MAE1 = 0
for i in range(0,5):
    MAE1 = MAE1 + mean_squared_error(y_test_inverse[:,i], y_test_predict[:,i])
MAE2 = 0
for i in range(0,5):
    MAE2 = MAE2 + mean_squared_error(y_test_inverse1[:,i], y_test_predict1[:,i])
print('ADE:',ade_1/len(y_test_predict))
print('FDE:',fde_1/len(y_test_predict))
print('MSE:',(MAE1+MAE2)/10)