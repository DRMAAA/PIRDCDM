import torch
from torch import nn
import numpy as np
import h5py
import cv2
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset
from torchvision import transforms
from torch.utils.data import _utils

import matplotlib.pyplot as plt
from visdom import Visdom
import time

import RDDM

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def data_change(data):
    data = np.where(data == 1,0,data)
    data = np.where(data == 128,1,data)
    data = np.where(data == 254,2,data)
    data = np.where(data == 3,0.01,data)
    return data

def matrix_upsample(input_matrix):
    temp_matrix = np.zeros((64, 64))
    for i in range(16):
        for j in range(16):
            temp_matrix[4*i:4*i+4, 4*j:4*j+4] = input_matrix[i, j]
    return temp_matrix

def normalization_data(data):
    data_max = data.max()
    data_min = data.min()
    gap = data_max - data_min
    temp_data = data - data_min
    temp_data = temp_data / gap
    return temp_data

def normalization_divide255(data):
    data_max = 255
    data_min = 0
    gap = data_max - data_min
    temp_data = data - data_min
    temp_data = temp_data / gap
    return temp_data

def VFM(vih,ele_num):
    matrix = np.ones((ele_num, ele_num))
    for i in range(ele_num):
        for j in range(ele_num):
            if i == j:
                matrix[i, j] = 0
                if i > 0 :
                    matrix[i-1, j] = 0
                if i < ele_num-1 :
                    matrix[i+1, j] = 0
    matrix[0, ele_num-1] = 0
    matrix[ele_num-1, 0] = 0
    # vh = vh.reshape(40, 1)
    k = 0
    for i in range(ele_num):
        for j in range(ele_num):
            if matrix[i, j] == 1:
                matrix[i, j] = vih[0, k]
                k += 1
    return matrix

def inter_new(vih,ele_num):
    temp_arr = vih
    input_arr = np.zeros([ele_num*2-3,ele_num])
    #扩展为(16,61)
    for i in range(temp_arr.shape[0]):
        if i >= 2 and i<=ele_num-3:
            # if i >= 2:
            # print(i)
            temp_arr[i] = np.roll(temp_arr[i],-i+1)
    temp_arr = cv2.resize(temp_arr,input_arr.shape,interpolation=cv2.INTER_LINEAR)
    # temp_arr = cv2.resize(temp_arr,[ele_num*2-3,ele_num*2],interpolation=cv2.INTER_LINEAR)

    res_arr = np.zeros([ele_num*2,ele_num*2-3])
    for i in range(temp_arr.shape[1]):
        y_temp = np.zeros(ele_num+1)
        y_temp[:ele_num] = temp_arr[:,i]
        y_temp[ele_num] = y_temp[0]
        # print(y_temp)

        # x_temp = [0,2,4,6,8,10,12,14,17]  #电极编号
        x_temp = np.arange(0,ele_num*2+1,2)
        # print(x_temp)
        x_temp[ele_num] = ele_num*2+1
        x_new = np.arange(0,ele_num*2)

        y_new = np.interp(x_new,x_temp,y_temp)
        res_arr[:,i] = y_new

    for i in range(res_arr.shape[0]):
        if i >= 2 and i<=ele_num*2-3:
            # if i >= 2:
            #     print(i)
            res_arr[i] = np.roll(res_arr[i],(i)-1)
    return res_arr

def inter(vih):
    temp_arr = vih
    input_arr = np.zeros([16,61])
    #扩展为(16,61)
    for i in range(temp_arr.shape[0]):
        y_temp = temp_arr[i]
        # x_temp = [0,3,6,9,12]  #电极编号
        x_temp = np.arange(0,61,5)
        # print(y_temp.shape)
        x_new = np.arange(0,61)
        y_new = np.interp(x_new,x_temp,y_temp)
        input_arr[i] = y_new
    res_arr = np.zeros([64,61])

    #扩展为(64,61)
    for i in range(input_arr.shape[1]):
        y_temp = np.zeros(17)
        y_temp[:16] = input_arr[:,i]
        y_temp[16] = y_temp[0]
        # print(y_temp)

        # x_temp = [0,2,4,6,8,10,12,14,17]  #电极编号
        x_temp = np.arange(0,65,4)
        # print(x_temp)
        x_temp[16] = 65
        x_new = np.arange(0,64)

        y_new = np.interp(x_new,x_temp,y_temp)
        res_arr[:,i] = y_new

    return res_arr

def count_elements(matrix):
    unique_elements, counts = np.unique(matrix, return_counts=True)
    return dict(zip(unique_elements,counts))

batch = 8

path_bg = h5py.File("dataset/back_ground.mat")
data_bg = np.array(path_bg['back_ground'])
# path = h5py.File("dataset/cir&tri&rect_train_8216_02.mat")
# path = h5py.File("dataset/cir&tri&rect_test_8216_02.mat")
path = h5py.File("dataset/cir&tri&rect_test_16264_03_16&64.mat")
data = path['dataset']

# pre_result = np.zeros([1, 64, 64])
Y_temp = np.zeros([1, 64, 64])
Y_data = np.zeros([len(data['rec_data']), 64, 64])
X_temp = np.zeros([1, 208])
X_data = np.zeros([len(data['rec_data']), 208])
Z_data = np.zeros([len(data['rec_data']), 64, 64])
Z_temp = np.zeros([1, 64, 64])

#X:16测量电压 Y:gt Z:bg
# for i in range(2400):
for i in range(len(data['rec_data'])):
    print(i)
    Y_data[i] = data[data['rec_data'][i][0]]
    X_temp = np.array(data[data['v_ih_16'][i][0]])
    # X_temp = np.array(data[data['v_ih_16'][i][0]])-np.array(data[data['v_h_16'][i][0]])
    # X_data[i] = X_temp
    # X_data[i] = VFM(inter(X_temp.reshape(16,13)).reshape(1,3904),64).reshape(1,64,64)
    X_data[i] = VFM(inter_new(inter_new(X_temp.reshape(16,13),16),32).reshape(1,3904),64).reshape(1,64,64)
    Z_data[i] = data_bg.reshape(1,64,64)
    # X_data[i] = cv2.resize(VFM(X_temp,16).reshape(16,16),[64,64],interpolation = cv2.INTER_LINEAR).reshape(1,64,64)



Z = torch.from_numpy(Z_data).type(torch.FloatTensor)
Y = torch.from_numpy(Y_data).type(torch.FloatTensor)
X = torch.from_numpy(X_data).type(torch.FloatTensor)
train_dataset = TensorDataset(X, Y, Z)
# train_dataloader = torch.utils.data.DataLoader(
#     train_dataset,
#     batch_size=batch,
#     shuffle=True,
#     drop_last=True
# )
train_dataloader_01 = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=1,
    shuffle=True,
    drop_last=True
)

Y_temp = np.zeros([1, 64, 64])
Y_data = np.zeros([len(data['rec_data']), 64, 64])
X_temp = np.zeros([1, 208])
X_data = np.zeros([len(data['rec_data']), 64, 64])
Z_data = np.zeros([len(data['rec_data']), 64, 64])
Z_temp = np.zeros([1, 64, 64])
i = 0
for X, Y ,Z in train_dataloader_01:
    print(i)
    Y_data[i] = data[data['rec_data'][i][0]]
    X_temp = np.array(data[data['v_ih_16'][i][0]])-np.array(X).reshape(1,208)
    # X_temp = np.array(data[data['v_ih_16'][i][0]])-np.array(data[data['v_h_16'][i][0]])
    X_data[i] = VFM(inter(X_temp.reshape(16,13)).reshape(1,3904),64).reshape(1,64,64)
    # X_data[i] = VFM(inter(X_temp.reshape(16,13)).reshape(1,3904),64).reshape(1,64,64)
    # X_data[i] = VFM(inter_new(inter_new(X_temp.reshape(16,13),16),32).reshape(1,3904),64).reshape(1,64,64)
    Z_data[i] = np.array(Y)
    i += 1
Z = torch.from_numpy(Z_data).type(torch.FloatTensor)
Y = torch.from_numpy(Y_data).type(torch.FloatTensor)
X = torch.from_numpy(X_data).type(torch.FloatTensor)
train_dataset_02 = TensorDataset(X, Y, Z)
train_dataloader_02 = torch.utils.data.DataLoader(
    train_dataset_02,
    batch_size=batch,
    shuffle=True,
    drop_last=True
)


model_unet = RDDM.UnetRes(
    dim=64,
    # dim=16,
    dim_mults=(1, 2, 4, 8),
    num_unet=2,
    # condition=False,
    condition=True,
    # input_condition=False,
    input_condition=True,
    objective='pred_res_noise',
    # objective='pred_noise',
    test_res_or_noise = "res_noise"
)

# 初始化模型，定义优化器，损失函数
model_path = 'D:\project\EIT_03_V2C\\result'
# pt_path = 'D:\project\EIT_03_V2C\\result\\20model.pt'
numOfBatch = len(data['rec_data']) // batch
model = RDDM.ResidualDiffusion(model=model_unet,
                               image_size=64,
                               input_condition=True,
                               # input_condition=False,
                               condition=True,
                               # input_condition_mask=True,
                               input_condition_mask=False,
                               loss_type='l2',
                               ).to(device)
opt0 = torch.optim.RAdam(model.model.unet0.parameters(), lr=0.0001)
opt1 = torch.optim.RAdam(model.model.unet1.parameters(), lr=0.0001)

# #  instantiate the window class （实例化监听窗口类）
# viz = Visdom()
# #  create a window and initialize it （创建监听窗口）
# viz.line([[0.,0.]], [0], win='train', opts=dict(title='loss_0&loss_1', legend=['loss_0', 'loss_1']))

EPOCHS = 200
for epoch in range(EPOCHS):
    # for X, Y in train_dataloader:
    for X, Y ,Z in tqdm(train_dataloader_02):

        #X:16测量电压 Y:gt Z:bg
        x = normalization_data(X.reshape(batch, 1, 64, 64))     #VFM
        y = normalization_divide255(Y.reshape(batch, 1, 64, 64))     #gt
        z = normalization_divide255(Z.reshape(batch, 1, 64, 64))     #bg或者是预重建图像
        x, y, z = x.to(device), y.to(device), z.to(device)

        loss = model(y, z, x)

        opt0.zero_grad()
        loss[0].backward(retain_graph=True)
        opt0.step()
        opt1.zero_grad()
        loss[1].backward(retain_graph=False)
        opt1.step()
        # break
    with torch.no_grad():
        print('[EPOCH:{}/{}]  [loss-unet0:{}]   [loss-unet1:{}]'.
              format(epoch+1, EPOCHS, loss[0], loss[1]))
        # viz.line([[loss[0].cpu().detach().numpy(), loss[1].cpu().detach().numpy()]], [epoch], win='train', update='append')
        if (epoch+1) % 20 == 0:
            path = model_path + "//" + str(epoch+1) + "model.pt"
            # path_unet0 = model_path + "//" + str(epoch+1) + "unet0_model.pt"
            # path_unet1 = model_path + "//" + str(epoch+1) + "unet1_model.pt"
            torch.save(model.state_dict(), path)
            # torch.save(model_unet.unet0.state_dict(), path_unet0)
            # torch.save(model_unet.unet1.state_dict(), path_unet1)
