import torch
from torch import nn
import torchvision
import os
import matplotlib.pyplot as plt
import h5py
import numpy as np
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset
import cv2

import RDDM

from noise_scheduler import Scheduler
from noise_scheduler import Cosine_Scheduler
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

path = h5py.File("dataset/cir&tri&rect_test_8216_02.mat")
# path = h5py.File("dataset/cir_combine_train_8216_02_tuoyuan.mat")
data = path['dataset']
path_bg = h5py.File("dataset/back_ground.mat")
data_bg = np.array(path_bg['back_ground'])
# print(data.keys())

pre_result = np.zeros([1, 64, 64])
Y_temp = np.zeros([1, 64, 64])
Y_data = np.zeros([len(data['rec_data']), 64, 64])
X_temp = np.zeros([1,208])
X_data = np.zeros([len(data['rec_data']), 64, 64])
Z_data = np.zeros([len(data['rec_data']), 64, 64])

for i in range(len(data['rec_data'])):
# for i in range(1200)):
    print(i)
    Y_data[i] = data[data['rec_data'][i][0]]
    # X_temp = np.array(data[data['v_ih_16'][i][0]])-np.array(data[data['v_h_16'][i][0]])
    X_temp = np.array(data[data['v_ih_16'][i][0]])
    # X_temp = np.roll(X_temp.reshape(16,13), shift=-1, axis=0).reshape(16,13)

    # X_data[i] = cv2.resize(VFM(X_temp,8),[64,64],interpolation=cv2.INTER_NEAREST).reshape(1,64,64)
    # X_data[i] = VFM(inter_new(inter_new(X_temp.reshape(16,13),16),32).reshape(1,3904),64).reshape(1,64,64)
    X_data[i] = VFM(inter(X_temp.reshape(16,13)).reshape(1,3904),64).reshape(1,64,64)
    Z_data[i] = data_bg.reshape(1,64,64)
    # Z_data[i] = np.full([64,64],128).reshape(1,64,64)
    # Z_data[i] = np.array(data[data['rec_data'][0][0]]).reshape(1,64,64)


Y = torch.from_numpy(Y_data).type(torch.FloatTensor)
# Y = normalization_data(Y)*255
X = torch.from_numpy(X_data).type(torch.FloatTensor)
Z = torch.from_numpy(Z_data).type(torch.FloatTensor)
# train_dataset = TensorDataset(X, Y, Z)
# train_dataloader = torch.utils.data.DataLoader(
#     train_dataset,
#     batch_size=1,
#     shuffle=True,
#     drop_last=True
# )

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
                               sampling_timesteps=1000,
                               ).to(device)
model.load_state_dict(torch.load("model/best.pt"))
# model.load_state_dict(torch.load("result_cha_newinter_16ele_1-3circle/400model.pt"))
# model_unet.unet0.load_state_dict(torch.load("D:\\project\\EIT_03_V2C\\result\\unet0_model.pt"))
# model_unet.unet1.load_state_dict(torch.load("D:\\project\\EIT_03_V2C\\result\\unet1_model.pt"))
# model.load_state_dict(torch.load("D:\\project\\EIT_03_V2C\\result\\100model_02.pt"))
model.eval()
model_unet.eval()
model_unet.unet0.eval()
model_unet.unet1.eval()

for j in range(len(data['rec_data'])):
    print(j)
    # j = 768
    # input = normalization_data(X[j,:,:].reshape(1, 1, 16, 16)).to(device)
    # input = normalization_data(Z[j,:,:].reshape(1, 1, 64, 64)).to(device)

    # trans_input = prerec(normalization_data(X[j,:,:].reshape(1, 1, 64, 64)).to(device))

    imgs = []
    imgs.append(normalization_divide255(Z[j,:,:].reshape(1, 1, 64, 64)).to(device))
    # imgs.append(Z[j,:,:].reshape(1, 1, 64, 64).to(device))
    imgs.append(normalization_data(X[j,:,:].reshape(1, 1, 64, 64)).to(device))
    # imgs.append(trans_input)
    # imgs.append(X[j,:,:].reshape(1, 1, 64, 64).to(device))
    images_list = list(model.sample(imgs, batch_size=1, last=True))
    # print(len(images_list))
    # print(images_list[0].shape)

    out = normalization_data(images_list[1].detach().cpu().numpy().reshape(64,64,1))*255
    cv2.imwrite('predict//'+str(j).zfill(6)+'.png',out,[int(cv2.IMWRITE_PNG_COMPRESSION), 0])

    # output_size =64
    # print(i,"----",count_elements(x0_pred.detach().cpu().numpy()))
    # out = (normalization_data(x0_pred.detach().cpu().numpy().reshape(output_size,output_size,1))*255).astype('uint8')
    # # out = (normalization_data(xt.detach().cpu().numpy().reshape(64,64,1))*255).astype('uint8')
    # cv2.imwrite('predict//'+str(i).zfill(6)+'.png',out,[int(cv2.IMWRITE_PNG_COMPRESSION), 0])
    # cv2.imwrite('gt//'+str(j).zfill(6)+'.png',normalization_data(Y[j].numpy().reshape(64,64,1))*255,[int(cv2.IMWRITE_PNG_COMPRESSION), 0])
    # cv2.imwrite('predict//''condition.png',normalization_data(X[temp].numpy().reshape(output_size,output_size,1))*255,[int(cv2.IMWRITE_PNG_COMPRESSION), 0])
    # break


# # j = 588
# # while(j<1200):
# while(True):
#     j = int(input("j="))
#     # input = normalization_data(X[j,:,:].reshape(1, 1, 16, 16)).to(device)
#     # input = normalization_data(Z[j,:,:].reshape(1, 1, 64, 64)).to(device)
#     imgs = []
#     imgs.append(normalization_divide255(Z[j,:,:].reshape(1, 1, 64, 64)).to(device))
#     # imgs.append(Z[j,:,:].reshape(1, 1, 64, 64).to(device))
#     imgs.append(normalization_data(X[j,:,:].reshape(1, 1, 64, 64)).to(device))
#     # imgs.append(X[j,:,:].reshape(1, 1, 64, 64).to(device))
#     images_list = list(model.sample(imgs, batch_size=1, last=True))
#     # print(len(images_list))
#     # print(images_list[0].shape)
#
#     out = normalization_data(images_list[1].detach().cpu().numpy().reshape(64,64,1))*255
#
#     fig, (ax1) = plt.subplots(1, 1)
#     ax1.imshow(out.reshape(64,64))
#     fig.show()
#     cv2.imwrite('predict//'+str(j).zfill(6)+'.png',out,[int(cv2.IMWRITE_PNG_COMPRESSION), 0])
#     cv2.imwrite('predict//'+str(j).zfill(6)+'gt.png',normalization_data(Y[j].numpy().reshape(64,64,1))*255,[int(cv2.IMWRITE_PNG_COMPRESSION), 0])
#     # print(j)
#     # j = j+1
#     # output_size =64
#     # print(i,"----",count_elements(x0_pred.detach().cpu().numpy()))
#     # out = (normalization_data(x0_pred.detach().cpu().numpy().reshape(output_size,output_size,1))*255).astype('uint8')
#     # # out = (normalization_data(xt.detach().cpu().numpy().reshape(64,64,1))*255).astype('uint8')
#     # cv2.imwrite('predict//'+str(i).zfill(6)+'.png',out,[int(cv2.IMWRITE_PNG_COMPRESSION), 0])
#     # cv2.imwrite('predict//''gt.png',normalization_data(Y[temp].numpy().reshape(output_size,output_size,1))*255,[int(cv2.IMWRITE_PNG_COMPRESSION), 0])
#     # cv2.imwrite('predict//''condition.png',normalization_data(X[temp].numpy().reshape(output_size,output_size,1))*255,[int(cv2.IMWRITE_PNG_COMPRESSION), 0])
