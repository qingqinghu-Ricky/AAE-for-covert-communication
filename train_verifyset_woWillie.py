# 信道为lognormal跨水面信道，加入非线性和shot noise, 加入人工噪声和Willie网络
# （n,k）=(8,4)
# 固定SNR训练, 没有Willie

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

from func.plot_func import accuracies_chart, bler_chart, losses_chart, TSNE_plot, constellation_image, PDFdiff_chart, ber_chart
from func.accuracy_func import willie_accuracy, alice_accuracy, bob_accuracy, bob_ber
from model.model_lognormal import Autoencoder, Willie, gradient_penalty
from scipy.optimize import linear_sum_assignment
from func.MMD_JS_loss_func import MMD_loss, JS_div




# 训练 ———————————————————————————————————————————————————————————————————————————————————————
def train(args, models, data_loader, verify_ds, lambdas=(1, 1), optim_fn=torch.optim.Adam):

    verify_x, verify_y = verify_ds[:][0], verify_ds[:][1]
    # 初始化优化器
    ae, w = models
    # w_optimizer = optim_fn(w.parameters(), lr=args.lr, amsgrad=True, eps=1e-8) # weight_decay=0.00001
    ae_encoder_optimizer = optim_fn(ae.encoder.parameters(), lr=args.lr, amsgrad=True, eps=1e-8)
    ae_decoder_optimizer = optim_fn(ae.decoder.parameters(), lr=args.lr, amsgrad=True, eps=1e-8)
 

    # 初始化 损失函数criterion 及系数lambda
    device = verify_x.device
    ae_criterion = nn.CrossEntropyLoss().to(device) #  CrossEntropyLoss计算预测结果向量 和 真实值label 的交叉熵损失,已经内置了softmax处理
    w_criterion = nn.BCELoss().to(device)
    bob_lambda, willie_lambda = lambdas

    losses = { 'bob_tra': []}
    accs = {'bob_val': []}
    PDF_diff = {'JS_val': []}
    ber = {'bob_val': []}


    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(w_optimizer, args.total_epoch, eta_min=0, last_epoch=-1) # 学习率lr随epoch增加呈cos形状减小
    scheduler_ae_encoder = torch.optim.lr_scheduler.CosineAnnealingLR(ae_encoder_optimizer, args.total_epoch, eta_min=0, last_epoch=-1) # 学习率lr随epoch增加呈cos形状减小
    scheduler_ae_decoder = torch.optim.lr_scheduler.CosineAnnealingLR(ae_decoder_optimizer, args.total_epoch, eta_min=0, last_epoch=-1)

    # 计算未训练前分布的差异
    with torch.no_grad():
        ae.eval()
        w.eval()
        t = torch.randn(verify_x.size()[0], args.n_channel).to(device) # n长度的随机输入向量t
        snr = torch.tensor(15, requires_grad=False).to(device) ####################################################
        bob_decode0, covert_aftchannel0, noise_data0 = ae(verify_x, t, snr) # 默认调用forward函数

        bob_acc0 = bob_accuracy(bob_decode0, verify_y)
        bob_ber0 = bob_ber(bob_decode0, verify_y,args)  # 计算Bob按bit计算的ber

        willie_output0 = w(covert_aftchannel0)
        willie_acc0 = willie_accuracy(willie_output0, w(noise_data0)) 
        alice_acc0 = alice_accuracy(willie_output0)

        JS0 = JS_div(covert_aftchannel0.reshape(-1).detach().cpu().numpy(), noise_data0.reshape(-1).detach().cpu().numpy(), num_bins=20)

        PDF_diff['JS_val'].append(JS0)

        accs['bob_val'].append(bob_acc0) 
        ber['bob_val'].append(bob_ber0) # 存储成numpy形式
        # accs['willie_val'].append(willie_acc0) 
        # accs['alice_val'].append(alice_acc0)
 
        
        epoch = 0
        print('Train Epoch[{}/{}],  Acc_B:{:.3f},   Acc_W:{:.3f},  JS:{:.3e}, ber: {:.3e},  SNR:{}'.format(
                    epoch, args.total_epoch, bob_acc0, willie_acc0, JS0, bob_ber0, snr.numpy()))
        TSNE_plot(covert_aftchannel0[0:args.batch_size], noise_data0[0:args.batch_size], args, epoch)   # 绘制输出是否发送数据的分布差异可视化
        constellation_image(covert_aftchannel0[0:args.batch_size], noise_data0[0:args.batch_size], args, epoch) # 二维星座点图

    # 开始训练 ——————————————————————————————————————————————————————————————————————————————————————————————————————————————————————

    for epoch in range(args.total_epoch):
        ae.train()
        w.train()

        train_willie_loss=[]
        train_alice_loss=[]
        train_bob_loss=[]


        w_num = 20
        a_num = 1
        
        
        # snr = torch.randint(5, 31, (1,), requires_grad=False).to(device) # 产生[5,30]的均匀分布的随机整数
        snr = torch.tensor(15, requires_grad=False).to(device) # 产生[5,30]的均匀分布的随机整数#############################################

        for i, (x, y) in enumerate(data_loader):
            x = x.to(device)
            y = y.to(device)


            # 缓存梯度清零
            ae_encoder_optimizer.zero_grad()
            ae_decoder_optimizer.zero_grad()
            # ae_sigma_optimizer.zero_grad()
            # w_optimizer.zero_grad()
            
            # 训练 decoder 网络————————————————————————————————————————————————————————————————————————
            t = torch.randn(args.batch_size, args.n_channel).to(device) # 2n长度的随机输入向量t
            bob_decode, covert_aftchannel, noise_data = ae(x, t, snr) # 默认调用forward函数

            bob_loss = ae_criterion(bob_decode, y)

            bob_loss = 1.0 * bob_loss 
            if np.isnan(bob_loss.detach().numpy()) == False:   # 判断输出是否有NaN
                bob_loss.backward()
                ae_decoder_optimizer.step()
                ae_encoder_optimizer.step()
            else:
                import pdb; pdb.set_trace()

            with torch.no_grad():
                bob_acc = bob_accuracy(bob_decode, y)


            train_bob_loss.append(bob_loss.item())
                
                
            


        # 每1个epoch的loss
        # losses['willie_tra'].append(np.mean(train_willie_loss)) # .item 用于在只包含一个元素的tensor中提取值, 返回这个张量的值作为一个标准的Python数字,在浮点数结果上使用 .item() 函数可以提高显示精度
        losses['bob_tra'].append(np.mean(train_bob_loss))
        # losses['alice_tra'].append(np.mean(train_alice_loss)) 


        
 
        #  绘图 计算随epoch的变化情况————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
        if (epoch == args.total_epoch-1): # 最后一个epoch再画图
            TSNE_plot(covert_aftchannel, noise_data, args, epoch+1)   # 绘制输出是否发送数据的分布差异可视化
            constellation_image(covert_aftchannel, noise_data, args, epoch+1) # 二维星座点图


        if (epoch+1) % args.plot_interval == 0 or epoch == 0:

            # 验证集结果 ———————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
            with torch.no_grad():
                ae.eval()
                w.eval()
                t = torch.randn(verify_x.size()[0], args.n_channel).to(device) # n长度的随机输入向量t
                bob_decode_verify, covert_aftchannel_verify, noise_data_verify = ae(verify_x, t, snr) # 默认调用forward函数

                bob_acc_verify = bob_accuracy(bob_decode_verify, verify_y)
                bob_ber_verify = bob_ber(bob_decode_verify, verify_y,args)  # 计算Bob按bit计算的ber

                # w_verify = w(covert_aftchannel_verify)
                # willie_acc_verify = willie_accuracy(w_verify, w(noise_data_verify)) 
                # alice_acc_verify = alice_accuracy(w_verify)

       
                JS_verify = JS_div(covert_aftchannel_verify.reshape(-1).detach().cpu().numpy(), noise_data_verify.reshape(-1).detach().cpu().numpy(), num_bins=20)

                PDF_diff['JS_val'].append(JS_verify)
                accs['bob_val'].append(bob_acc_verify) # 存储成numpy形式
                ber['bob_val'].append(bob_ber_verify) # 存储成numpy形式
                # accs['willie_val'].append(willie_acc_verify) # 存储成numpy形式
                # accs['alice_val'].append(alice_acc_verify) # 存储成numpy形式
            
            print('Train Epoch[{}/{}], Loss_B:{:.4f}, Acc_B:{:.3f},    JS:{:.3e}, ber: {:.3e}, SNR:{}'.format(
            epoch + 1, args.total_epoch,  np.mean(train_bob_loss), bob_acc_verify,    JS_verify, bob_ber_verify, snr.numpy()))
            print('w_num: {}, a_num: {}'.format(w_num, a_num))



        # scheduler.step()
        scheduler_ae_encoder.step()
        scheduler_ae_decoder.step()
        # scheduler_ae_sigma.step()
    return losses, PDF_diff, accs, ber


# 主函数 ——————————————————————————————————————————————————————————————————————————————————————————————————
def main(args):
    torch.manual_seed(args.seed)
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    print('Device: ', device)

    # 隐蔽训练信号————————————————————————————————————————————————————————————————————————————————————
    train_labels = (torch.rand(args.train_size) * args.tx).long() # 8192:0-2^k之间
    train_data = torch.sparse.torch.eye(args.tx).index_select(dim=0, index=train_labels) # 对应位置为1
    train_labels = train_labels.to(device)
    train_data = train_data.to(device)
    train_ds = Data.TensorDataset(train_data, train_labels)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True) # 打乱后按batch分组的训练数据


    # 隐蔽验证信号 ————————————————————————————————————————————————————————————————————————————————
    verify_labels = (torch.rand(args.verify_size) * args.tx).long() # 8192:0-m之间
    verify_data = torch.eye(args.tx).index_select(dim=0, index=verify_labels) # 对应位置为1
    verify_labels = verify_labels.to(device)
    verify_data = verify_data.to(device)
    verify_ds = Data.TensorDataset(verify_data, verify_labels)

    # 初始化模型，生成实例
    AE = Autoencoder(args.tx, args.n_channel, args.channel_type, args.r)
    W = Willie(args.n_channel)
    
    # load 已训练好模型参数 ——————————————————————————————————————————————————————————————————————————
    # state_dict_path = os.path.join(args.modelpara_dir, 'autoencoder_lognormal{}_{}.pth'.format(args.n_channel, args.k))
    # if os.path.exists(state_dict_path):
    #     info = AE.load_state_dict(torch.load(state_dict_path))
    #     print(info)
    # state_dict_path = os.path.join(args.modelpara_dir, 'willie_lognormal{}_{}.pth'.format(args.n_channel, args.k))
    # if os.path.exists(state_dict_path):
    #     info = W.load_state_dict(torch.load(state_dict_path))
    #     print(info)  
    
    AE = AE.to(device)  
    W = W.to(device)
    models = (AE, W)

    bob_lambda, willie_lambda = 0.6, 3

    # for para in model.parameters():
    #     print(para)
    
    losses, PDF_diff, accs, ber = train(args, models, train_dl, verify_ds, lambdas=(bob_lambda, willie_lambda))

    # 保存输出数据字典到文件——————————————————————————————————————————————————————————————————————————
    save_output(losses, args.saveoutput_dir, 
                'loss_lognormal{}_{}.npy'.format(args.n_channel, args.k))
    save_output(PDF_diff, args.saveoutput_dir, 
                'PDF_diff_lognormal{}_{}.npy'.format(args.n_channel, args.k))
    save_output(accs, args.saveoutput_dir, 
                'accs_lognormal{}_{}.npy'.format(args.n_channel, args.k)) 
    save_output(ber, args.saveoutput_dir, 
        'ber_lognormal{}_{}.npy'.format(args.n_channel, args.k)) 


    # 读取文件
    # new_dict = np.load('file.npy', allow_pickle=True)    # 输出即为Dict 类型

    # 绘图 ————————————————————————————————————————————————————————————————————————————————————————
    losses_chart(losses,args)
    PDFdiff_chart(PDF_diff,args)
    bler_chart(accs, args)
    ber_chart(ber, args)

    # 保存现在的模型参数————————————————————————————————————————————————————————————————————————
    save_checkpoint(models[0].state_dict(), args.modelpara_dir, 
                    'autoencoder_lognormal{}_{}.pth'.format(args.n_channel, args.k))
    save_checkpoint(models[1].state_dict(), args.modelpara_dir, 
                    'willie_lognormal{}_{}.pth'.format(args.n_channel, args.k))    
    
def save_output( output_var, output_dir, output_name): # 保存模型参数
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    np.save(os.path.join(output_dir, output_name), output_var) 

def save_checkpoint(state_dict, model_dir, model_name): # 保存模型参数
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    torch.save(state_dict, os.path.join(model_dir, model_name))   

    
if __name__ == '__main__': # 默认整个脚本从此处开始运行
    def parse_args():
        parser = argparse.ArgumentParser(description = "AutoEocoder")
        parser.add_argument('--channel_type', type=str, default = 'lognormal') # awgn, rayleigh，lognormal
        parser.add_argument('--train_size', type=int, default = 1024 * 10) #1024 * 8
        parser.add_argument('--verify_size', type=int, default = 1024*10)
        parser.add_argument('--batch_size', type=int, default = 1024) # 1024
        parser.add_argument('--n_channel', type=int, default = 8) # 信道编码后的码长n
        parser.add_argument('--k', type=int, default = 4) # k是发送的message的比特数
        parser.add_argument('--lr', type=int, default = 2e-4) # 2e-4, 1e-4 ########################################################
        parser.add_argument('--result_dir', type=str, default = 'results')
        parser.add_argument('--saveoutput_dir', type=str, default = 'saveoutput')
        parser.add_argument('--modelpara_dir', type=str, default = 'modelparameter')
        parser.add_argument('--seed', type=int, default = 100)
        parser.add_argument('--total_epoch', type=int, default = 4000)  # epoch总数 4000
        parser.add_argument('--plot_interval', type=int, default = 3000) # 绘图epoch间隔 100
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