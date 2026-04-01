import time

import serial
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import h5py
import torch.nn as nn
import csv

import V2A_net
import eletrans_net
import RDDM

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

def normalization_divide255(data):
    data_max = 255
    data_min = 0
    gap = data_max - data_min
    temp_data = data - data_min
    temp_data = temp_data / gap
    return temp_data

def normalization_data(data):
    data_max = data.max()
    data_min = data.min()
    gap = data_max - data_min
    temp_data = data - data_min
    temp_data = temp_data / gap
    return temp_data

def count_elements(matrix):
    unique_elements, counts = np.unique(matrix, return_counts=True)
    return dict(zip(unique_elements,counts))

def assign_values(matrix):
    new_matrix = [[0]*64 for _ in range(64)]
    for i in range(64):
        for j in range(64):
            value = matrix[i][j]
            if 0.75 <= value <= 0.965:
                new_matrix[i][j] = 255
            elif -0.965 < value <= -0.75:
                new_matrix[i][j] = 255
            else:
                new_matrix[i][j] = 0

    return new_matrix

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
# 初始化端口
ser = serial.Serial('COM4',115200)

# 初始化模型
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'

pre = eletrans_net.Unet().to(device)
pre.load_state_dict(torch.load('result-8216_02/25model.pt'))

model_unet = RDDM.UnetRes(
    dim=64,
    # dim=16,
    dim_mults=(1, 2, 4, 8),
    num_unet=2,
    condition=True,
    input_condition=True,
    # input_condition=False,
    objective='pred_res_noise',
    # objective='pred_noise',
    test_res_or_noise = "res_noise",
    # test_res_or_noise = "noise",
    # test_res_or_noise = "res",
)

model = RDDM.ResidualDiffusion(model=model_unet,
                               image_size=64,
                               input_condition=True,
                               # input_condition=False,
                               condition=True,
                               # input_condition_mask=True,
                               input_condition_mask=False,
                               test_res_or_noise = "res_noise",
                               # test_res_or_noise="noise",
                               # test_res_or_noise = "res",
                               sampling_timesteps=5,
                               ).to(device)
model.load_state_dict(torch.load("D:\project\EIT_03_V2C\\result_cha_near_16ele_1-3circle\\200model.pt"))
# model.load_state_dict(torch.load("result_cha_near_8ele/400model.pt"))
model.eval()
model_unet.eval()
model_unet.unet0.eval()
model_unet.unet1.eval()

path_bg = h5py.File("dataset/back_ground.mat")
data_bg = np.array(path_bg['back_ground'])
Z_data = np.zeros([1, 64, 64])
Z_data[0] = data_bg.reshape(1,64,64)
Z = torch.from_numpy(Z_data).type(torch.FloatTensor)

pre_path = 'D:\project\EIT_03_V2C\live_pre'

# input_vh = input("v_h=")
# numbers = [float(num) for num in input_vh.split(',') if num not in ('A', 'B')]
# v_h = np.array(numbers).reshape(1, 40)

judge = 0
count = 0

while True:
    # 读取串口数据
    # time.sleep(1)
    data = ser.readline()
    print(data)
    # if count < 50:
    #     count += 1
    #     print(count)
    #     continue
    if data and len(data)>100 and (count % 20 == 0):
        if judge == 0:
            print("v_h---->")
            v_h = [int(x) for x in data.decode('utf-8').strip().split(',') if x not in ('A', 'B')]
            print(np.array(v_h).shape)
            v_h = np.array(v_h).reshape(1, 40)
            judge = 1
            continue

        # print(len(data))
        # data_list = data.decode('utf-8').split(',')
        # data_list.pop()
        # data_np = np.array(data_list, dtype=float).reshape(1,40)
        # data = ser.readline()
        # print(data.decode('utf-8'))
        # data_list = [int(x) for x in data.decode('utf-8').strip().split(',') if x not in ('A', 'B')]
        # data_np = np.array(data_list, dtype=float).reshape(1,40)
        data_np = [int(x) for x in data.decode('utf-8').strip().split(',') if x not in ('A', 'B')]
        data_np = np.array(data_np)

        # print(data_np)
        # v_ih = VFM((data_np-v_h), 8)
        # print(v_ih)
        # input = np.array(v_ih.reshape(8,8))

        # loss_fun = nn.MSELoss()
        # cha_mse = loss_fun(torch.tensor(data_np),torch.tensor(v_h))
        # print(cha_mse)
        # if cha_mse < 10 :
        #     input_cha = np.zeros([8,5])
        # else:
        #     input_cha = (data_np-v_h).reshape(8,5)

        input_cha = (data_np-v_h).reshape(8,5)
        # input = cv2.resize(VFM(input_cha.reshape(1,208),16),[64,64],interpolation=cv2.INTER_NEAREST).reshape(1,1,64,64)
        input = cv2.resize(VFM(input_cha.reshape(1,40),8),[64,64],interpolation=cv2.INTER_NEAREST).reshape(1,1,64,64)
        # print(input)
        # input = VFM(inter_new(inter_new(inter_new(input_cha,8),16),32).reshape(1,3904),64).reshape(64,64)
        data_tensor = torch.from_numpy(input).type(torch.FloatTensor)


        imgs = []
        imgs.append(normalization_divide255(Z[0,:,:].reshape(1, 1, 64, 64)).to(device))
        # imgs.append(normalization_divide255(torch.zeros([1, 1, 64, 64])).to(device))
        # imgs.append(Z[j,:,:].reshape(1, 1, 64, 64).to(device))
        imgs.append(normalization_data(data_tensor.reshape(1, 1, 64, 64)).to(device))
        # imgs.append(X[j,:,:].reshape(1, 1, 64, 64).to(device))
        images_list = list(model.sample(imgs, batch_size=1, last=True))
        # print(len(images_list))
        # print(images_list[0].shape)

        if np.count_nonzero(imgs[1].cpu().numpy()) < 50:
            out = normalization_data(imgs[0].cpu().numpy().reshape(64,64))
        else:
            out = normalization_data(images_list[1].detach().cpu().numpy().reshape(64,64,1))*255
            # out = np.array(assign_values(images_list[1].detach().cpu().numpy().reshape(64,64)))
        # print(count_elements(out))
        # print(count_elements(np.array(Z[0,:,:].reshape(64,64))))
        cv2.imwrite(pre_path+'//real.png',out,[int(cv2.IMWRITE_PNG_COMPRESSION), 0])

        # with open("D:\project\EIT_03_V2C\\test.csv", 'w', newline='') as csvfile:
        #     csv_writer = csv.writer(csvfile)
        #     for row in out:
        #         csv_writer.writerow(row)

        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(VFM(data_np.reshape(1,40),8).reshape(8,8,1))
        # ax1.imshow(VFM(data_np.reshape(1,208),16).reshape(16,16,1))
        # ax1.imshow(VFM(data_np.reshape(1,208),16).reshape(16,16))
        ax2.imshow(out.reshape(64,64,1))
        fig.show()

    count += 1
