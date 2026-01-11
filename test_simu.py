# 信道为lognormal跨水面信道，加入非线性和shot noise, 加入人工噪声和Willie网络
# 验证集结果, 仿真信道


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from torch.utils.data import DataLoader
import argparse
import matplotlib.pyplot as plt
import os
import copy
import numpy as np
import random

from func.plot_func import accuracies_chart, bler_chart, losses_chart, TSNE_plot, constellation_image, PDFdiff_chart
from func.accuracy_func import willie_accuracy, alice_accuracy, bob_accuracy, bob_ber
from model.model_lognormal import Autoencoder, Willie, gradient_penalty  #########################################
from scipy.optimize import linear_sum_assignment
from func.MMD_JS_loss_func import MMD_loss, JS_div


accs = {'willie_test': [], 'bob_test': [],'alice_test':[]}
# PDF_diff = {'JS_test':[], 'MMD_test':[]}
PDF_diff = {'JS_test':[]}
ber = {'bob_test': []}

# 训练 ———————————————————————————————————————————————————————————————————————————————————————
def verify(args, models, test_ds, snr):
    test_x, test_y = test_ds[:][0], test_ds[:][1]
    device = test_x.device
    ae, w = models
    # snr = torch.tensor(30, requires_grad=False).to(device) # 产生[5,30]的均匀分布的随机整数
    with torch.no_grad():
        ae.eval()
        w.eval()
        t = torch.randn(test_x.size()[0], args.n_channel).to(device) # n长度的随机输入向量t
        bob_decode_test, covert_aftchannel_test, noise_data_test = ae(test_x, t, snr) # 默认调用forward函数
        # noise_data_test = torch.randn(test_x.size()[0], args.n_channel).to(device)
        bob_acc_test = bob_accuracy(bob_decode_test, test_y)
        bob_ber_test = bob_ber(bob_decode_test, test_y, args)  # 计算Bob按bit计算的ber
        w_covert = w(covert_aftchannel_test)
        willie_acc_test = willie_accuracy(w_covert, w(noise_data_test)) 
        alice_acc_test = alice_accuracy(w_covert)


        JS_test = JS_div(covert_aftchannel_test.reshape(-1).detach().cpu().numpy(), noise_data_test.reshape(-1).detach().cpu().numpy(), num_bins=20)
        # MMD_model = MMD_loss()
        # MMD_test = MMD_model(covert_aftchannel_test.reshape(-1).detach(), noise_data_test.reshape(-1).detach())

        PDF_diff['JS_test'].append(JS_test)
        # PDF_diff['MMD_test'].append(MMD_test)

        accs['bob_test'].append(bob_acc_test) # 存储成numpy形式
        accs['willie_test'].append(willie_acc_test) # 存储成numpy形式
        accs['alice_test'].append(alice_acc_test) # 存储成numpy形式

        ber['bob_test'].append(bob_ber_test) # 存储成numpy形式

        epoch = 'test'
        TSNE_plot(covert_aftchannel_test, noise_data_test, args, epoch)   # 绘制输出是否发送数据的分布差异可视化
        constellation_image(covert_aftchannel_test, noise_data_test, args, epoch) # 二维星座点图

        print('Test, Acc_B:{:.6f}, Acc_W:{:.6f}, Acc_A:{:.6f}, JS:{:.6e}, ber:{:.6e},  SNR:{}'.format( bob_acc_test, willie_acc_test, alice_acc_test,
                             JS_test, bob_ber_test, snr.numpy()))

    return PDF_diff, accs, ber


# 主函数 ——————————————————————————————————————————————————————————————————————————————————————————————————
def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    print('Device: ', device)  

    # 隐蔽测试信号 ————————————————————————————————————————————————————————————————————————————————
    test_labels = (torch.rand(int(args.test_size)) * args.tx).long() # 8192:0-m之间
    test_data = torch.eye(args.tx).index_select(dim=0, index=test_labels) # 对应位置为1
    test_labels = test_labels.to(device)
    test_data = test_data.to(device)
    test_ds = Data.TensorDataset(test_data, test_labels)

    # 初始化模型，生成实例
    # AE = Autoencoder(args.tx, args.n_channel, args.channel_type,args.ebno, args.r) ###############################
    AE = Autoencoder(args.tx, args.n_channel, args.channel_type, args.r)
    W = Willie(args.n_channel)
    
    # load 已训练好模型参数 ——————————————————————————————————————————————————————————————————————————
    state_dict_path = os.path.join(args.modelpara_dir, 'autoencoder_lognormal{}_{}.pth'.format(args.n_channel, args.k))
    if os.path.exists(state_dict_path):
        info = AE.load_state_dict(torch.load(state_dict_path))
        print(info)
        
    state_dict_path = os.path.join(args.modelpara_dir, 'willie_lognormal{}_{}.pth'.format(args.n_channel, args.k))
    if os.path.exists(state_dict_path):
        info = W.load_state_dict(torch.load(state_dict_path))
        print(info)  
    
    AE = AE.to(device)  
    W = W.to(device)
    models = (AE, W)

    # for para in model.parameters():
    #     print(para)

    # snr =  torch.tensor([5, 10, 15, 20, 25, 30], requires_grad=False).to(device) # 产生[5,30]的均匀分布的随机整数
    snr =  torch.tensor([10, 12, 15, 17, 20], requires_grad=False).to(device) # 产生[5,30]的均匀分布的随机整数
    # snr =  torch.tensor([15], requires_grad=False).to(device) # 产生[5,30]的均匀分布的随机整数
    
    for snr0 in snr:
    
        PDF_diff, accs, ber = verify(args, models, test_ds, snr0)

    # # 保存输出数据字典到文件——————————————————————————————————————————————————————————————————————————
    # save_output(PDF_diff, args.saveoutput_dir, 
    #             'PDF_diff_lognormal{}_{}.npy'.format(args.n_channel, args.k))
    # save_output(accs, args.saveoutput_dir, 
    #             'accs_lognormal{}_{}.npy'.format(args.n_channel, args.k)) 


    # 读取文件
    # new_dict = np.load('file.npy', allow_pickle=True)    # 输出即为Dict 类型

    # 绘图 ————————————————————————————————————————————————————————————————————————————————————————
    # PDFdiff_chart(PDF_diff,args)
    # accuracies_chart(accs, args)
   
    
def save_output( output_var, output_dir, output_name): # 保存模型参数
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    np.save(os.path.join(output_dir, output_name), output_var) 


    
if __name__ == '__main__': # 默认整个脚本从此处开始运行
    def parse_args():
        parser = argparse.ArgumentParser(description = "AutoEocoder")
        parser.add_argument('--channel_type', type=str, default = 'lognormal') # awgn, rayleigh，lognormal
        parser.add_argument('--train_size', type=int, default = 1024 * 10) #1024 * 8
        parser.add_argument('--test_size', type=int, default = 1024*300*4/2) # 1024*10###############################
        parser.add_argument('--batch_size', type=int, default = 1024) 
        parser.add_argument('--n_channel', type=int, default = 8) # 信道编码后的码长n
        parser.add_argument('--k', type=int, default = 2) # k是发送的message的比特数 #####################################
        parser.add_argument('--lr', type=int, default = 2e-4) # 2e-4
        parser.add_argument('--result_dir', type=str, default = 'results')
        parser.add_argument('--saveoutput_dir', type=str, default = 'saveoutput')
        parser.add_argument('--modelpara_dir', type=str, default = 'modelparameter')
        parser.add_argument('--seed', type=int, default = 10) # 10 #########################################
        parser.add_argument('--total_epoch', type=int, default = 500)  # epoch总数
        parser.add_argument('--plot_interval', type=int, default = 50) # 绘图epoch间隔
        args = parser.parse_args()
        return args
    args = parse_args()
    if args.channel_type == 'awgn':
        args.ebno = 8 
    if args.channel_type == 'rayleigh':
        args.ebno = 16 
    if args.channel_type == 'lognormal':
        args.ebno = 16

    args.tx = 2 ** args.k # one-hot编码长度
    args.r = args.k / args.n_channel # 速率： bit/channel
    main(args)