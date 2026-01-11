# 绘图的函数——————————————————————————————————————————————————————————————————————————————————————
import math
import os

import torch
import numpy as np
# from torch.utils.data import TensorDataset
# from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
# from sklearn import decomposition
# from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import argparse

plt.rcParams['font.family']='Times New Roman, SimSun'# 设置字体族，中文为SimSun，英文为Times New Roman
plt.rcParams['mathtext.fontset'] = 'stix' # 设置数学公式字体为stix
# plt.rc('font',family='Times New Roman')
# del matplotlib.font_manager.weight_dict['roman']
# matplotlib.font_manager._rebuild()

# 绘制准确度的图——————————————————————————————————————————————————————————————————————————————————————————————
def accuracies_chart(accs, args):    
    plt.figure(figsize=(40, 30))   # 打印图片的大小，尺寸是英寸（1英寸对应100像素）
    fig, ax = plt.subplots()
    # plots = ['bob_tra', 'willie_tra','bob_test', 'willie_test']
    plots = list(accs.keys()) # 取出字典中的键
    colors_map = {'bob_val': 'r', 'willie_val': 'g',  'alice_val': 'b', 'bob_tra': 'r', 'willie_tra': 'g',  'alice_tra': 'b', 'bob_test': 'r', 'willie_test': 'g','alice_test': 'y'}
    markers_map = {'bob_val': 'o', 'willie_val': '^','alice_val': 's', 'bob_tra': 'o', 'willie_tra': '^','alice_tra': 's', 'bob_test': '+','willie_test':'x','alice_test': '>'}
    line_map = {'bob_val': '-', 'willie_val': '-.','alice_val': '--'}


    for plot in plots:
        # plt.plot(range(0, len(accs[plot])), accs[plot], color = colors_map[plot], marker=markers_map[plot], markerfacecolor='none', label=str(plot).capitalize() + " Acc")
        # plt.plot(range(0, len(accs[plot])), accs[plot], color = colors_map[plot], label=str(plot).capitalize() + " Acc")
        plt.plot([0, 1]+[args.plot_interval * i for i in list(range(1, len(accs[plot])-1))], accs[plot], color = colors_map[plot], linestyle = line_map[plot], label=str(plot).capitalize())

    # plt.title("Models Accuracies")
    plt.tick_params(top='on', right='on', which='both') # 显示上侧和右侧的刻度
    plt.rcParams['xtick.direction'] = 'in' #将x轴的刻度线方向设置向内
    plt.rcParams['ytick.direction'] = 'in' #将y轴的刻度方向设置向内
    plt.xlabel("Epoch", fontsize=18)
    plt.ylabel("Accuracy", fontsize=18)
    plt.legend(loc="lower right", fontsize=16, facecolor=None,edgecolor=None, shadow=False,framealpha=1) # 给出图例，默认是plot中每根线条的label参数
    # ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: round(x * 10))) # 设置主坐标轴的刻度格式(小数位数),每10个epoch输出一个验证结果，所以横坐标要*10
    # plt.xticks(list(range(0, len(accs[plot]))), [args.plot_interval * i for i in list(range(0, len(accs[plot])))], fontsize=16)  # Set locations and labels
    plt.xticks(fontsize=16) # 设置刻度线格式
    plt.yticks(fontsize=16)
    plt.tight_layout()
    if not os.path.exists('results/png'):
        os.makedirs('results/png')
    if not os.path.exists('results/eps'):
        os.makedirs('results/eps')
    
    plt.savefig("{}/png/acc_{}_({},{}).png".format(args.result_dir, args.channel_type, args.n_channel, args.k)) # 默认输出8*6比例，即800*600dpi, 这里是8*500，6*500像素
    plt.savefig("{}/eps/acc_{}_({},{}).eps".format(args.result_dir, args.channel_type, args.n_channel, args.k))
    plt.close()


# 绘制BLER的图——————————————————————————————————————————————————————————————————————————————————————————————
def bler_chart(accs, args):    
    plt.figure(figsize=(40, 30))   # 打印图片的大小，尺寸是英寸（1英寸对应100像素）
    fig, ax = plt.subplots()
    plots = ['bob_val']
    # plots = ['bob_val']
    # plots = list(accs.keys()) # 取出字典中的键
    colors_map = {'bob_val': 'g', 'willie_val': 'g',  'alice_val': 'b', 'bob_tra': 'r', 'willie_tra': 'g',  'alice_tra': 'b', 'bob_test': 'r', 'willie_test': 'g','alice_test': 'y'}
    markers_map = {'bob_val': 'o', 'willie_val': '^','alice_val': 's', 'bob_tra': 'o', 'willie_tra': '^','alice_tra': 's', 'bob_test': '+','willie_test':'x','alice_test': '>'}
    line_map = {'bob_val': '-', 'willie_val': '-.','alice_val': '--'}


    for plot in plots:
        # plt.plot(range(0, len(accs[plot])), accs[plot], color = colors_map[plot], marker=markers_map[plot], markerfacecolor='none', label=str(plot).capitalize() + " Acc")
        # plt.plot(range(0, len(accs[plot])), accs[plot], color = colors_map[plot], label=str(plot).capitalize() + " Acc")
        plt.semilogy([0, 1]+[args.plot_interval * i for i in list(range(1, len(accs[plot])-1))], [1-x for x in accs[plot]], color = colors_map[plot], linestyle = line_map[plot])
    
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0),useMathText=True)   # 纵轴是科学计数法,1e6
    ax.xaxis.get_offset_text().set_fontsize(16)#设置1e6的大小与位置
    plt.tick_params(top='on', right='on', which='both') # 显示上侧和右侧的刻度
    plt.rcParams['xtick.direction'] = 'in' #将x轴的刻度线方向设置向内
    plt.rcParams['ytick.direction'] = 'in' #将y轴的刻度方向设置向内
    # plt.title("Models Accuracies")
    plt.xlabel("Epoch", fontsize=18)
    plt.ylabel("BLER", fontsize=18)
    # plt.legend(loc="upper right", fontsize=16, facecolor=None,edgecolor=None, shadow=False,framealpha=1) # 给出图例，默认是plot中每根线条的label参数
    # ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: round(x * 10))) # 设置主坐标轴的刻度格式(小数位数),每10个epoch输出一个验证结果，所以横坐标要*10
    # plt.xticks(list(range(0, len(accs[plot]))), [args.plot_interval * i for i in list(range(0, len(accs[plot])))], fontsize=16)  # Set locations and labels
    plt.xticks(fontsize=16) # 设置刻度线格式
    plt.yticks(fontsize=16)
    plt.tight_layout()
    if not os.path.exists('results/png'):
        os.makedirs('results/png')
    if not os.path.exists('results/eps'):
        os.makedirs('results/eps')
    
    plt.savefig("{}/png/bler_{}_({},{}).png".format(args.result_dir, args.channel_type, args.n_channel, args.k)) # 默认输出8*6比例，即800*600dpi, 这里是8*500，6*500像素
    plt.savefig("{}/eps/bler_{}_({},{}).eps".format(args.result_dir, args.channel_type, args.n_channel, args.k))
    plt.close()

# 绘制BER的图——————————————————————————————————————————————————————————————————————————————————————————————
def ber_chart(ber, args):    
    plt.figure(figsize=(40, 30))   # 打印图片的大小，尺寸是英寸（1英寸对应100像素）
    fig, ax = plt.subplots()
    # plots = ['bob_tra', 'willie_tra','bob_test', 'willie_test']
    plots = list(ber.keys()) # 取出字典中的键
    colors_map = {'bob_val': 'b', 'willie_val': 'g',  'alice_val': 'b', 'bob_tra': 'r', 'willie_tra': 'g',  'alice_tra': 'b', 'bob_test': 'r', 'willie_test': 'g','alice_test': 'y'}
    markers_map = {'bob_val': 'o', 'willie_val': '^','alice_val': 's', 'bob_tra': 'o', 'willie_tra': '^','alice_tra': 's', 'bob_test': '+','willie_test':'x','alice_test': '>'}
    line_map = {'bob_val': '-', 'willie_val': '-.','alice_val': '--'}


    for plot in plots:
        # plt.plot(range(0, len(accs[plot])), accs[plot], color = colors_map[plot], marker=markers_map[plot], markerfacecolor='none', label=str(plot).capitalize() + " Acc")
        # plt.plot(range(0, len(accs[plot])), accs[plot], color = colors_map[plot], label=str(plot).capitalize() + " Acc")
        plt.semilogy([0, 1]+[args.plot_interval * i for i in list(range(1, len(ber[plot])-1))], ber[plot], color = colors_map[plot], linestyle = line_map[plot])

    # plt.title("Models Accuracies")
        
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0),useMathText=True)   # 纵轴是科学计数法,1e6
    ax.xaxis.get_offset_text().set_fontsize(16)#设置1e6的大小与位置
    plt.tick_params(top='on', right='on', which='both') # 显示上侧和右侧的刻度
    plt.rcParams['xtick.direction'] = 'in' #将x轴的刻度线方向设置向内
    plt.rcParams['ytick.direction'] = 'in' #将y轴的刻度方向设置向内
    plt.xlabel("Epoch", fontsize=18)
    plt.ylabel("BER", fontsize=18)
    # plt.legend(loc="upper right", fontsize=16, facecolor=None,edgecolor=None, shadow=False,framealpha=1) # 给出图例，默认是plot中每根线条的label参数
    # ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: round(x * 10))) # 设置主坐标轴的刻度格式(小数位数),每10个epoch输出一个验证结果，所以横坐标要*10
    # plt.xticks(list(range(0, len(accs[plot]))), [args.plot_interval * i for i in list(range(0, len(accs[plot])))], fontsize=16)  # Set locations and labels
    plt.xticks(fontsize=16) # 设置刻度线格式
    plt.yticks(fontsize=16)
    plt.tight_layout()
    if not os.path.exists('results/png'):
        os.makedirs('results/png')
    if not os.path.exists('results/eps'):
        os.makedirs('results/eps')
    
    plt.savefig("{}/png/ber_{}_({},{}).png".format(args.result_dir, args.channel_type, args.n_channel, args.k)) # 默认输出8*6比例，即800*600dpi, 这里是8*500，6*500像素
    plt.savefig("{}/eps/ber_{}_({},{}).eps".format(args.result_dir, args.channel_type, args.n_channel, args.k))
    plt.close()

# 绘制loss——————————————————————————————————————————————————————————————————————————————————————————————————————
def losses_chart(losses,args):    
    plt.figure(figsize=(40, 30))
    fig, ax = plt.subplots()
    # plots = ['bob_tra', 'willie_tra','alice_tra']
    plots = list(losses.keys()) # 取出字典中的键
    colors_map = {'bob_tra': 'r', 'willie_tra': 'g', 'alice_tra': 'b'}
    line_map = {'bob_tra': '-', 'willie_tra': '-.','alice_tra': '--'}

    for plot in plots:
        plt.plot(range(0, len(losses[plot])), losses[plot], colors_map[plot], linestyle = line_map[plot], label=str(plot).capitalize())
    # plt.title("Models Accuracies")
    plt.tick_params(top='on', right='on', which='both') # 显示上侧和右侧的刻度
    plt.rcParams['xtick.direction'] = 'in' #将x轴的刻度线方向设置向内
    plt.rcParams['ytick.direction'] = 'in' #将y轴的刻度方向设置向内
    plt.xlabel("Epoch", fontsize=18)
    plt.ylabel("Loss", fontsize=18)
    plt.legend(loc="upper right", fontsize=16, facecolor=None,edgecolor=None, shadow=False,framealpha=1) # 给出图例，默认是plot中每根线条的label参数
    # ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: round(x * 10))) # 设置主坐标轴的刻度格式(小数位数),每10个epoch输出一个验证结果，所以横坐标要*10
    plt.xticks(fontsize=16) # 设置刻度线格式
    plt.yticks(fontsize=16)
    plt.tight_layout()
    if not os.path.exists('results/png'):
        os.makedirs('results/png')
    if not os.path.exists('results/eps'):
        os.makedirs('results/eps')
    
    plt.savefig("{}/png/loss_{}_({},{}).png".format(args.result_dir, args.channel_type, args.n_channel, args.k))
    plt.savefig("{}/eps/loss_{}_({},{}).eps".format(args.result_dir, args.channel_type, args.n_channel, args.k))

    plt.close()

# 绘制JS&MMD分布差异指标————————————————————————————————————————————————————————————————————————————————————————————

def PDFdiff_chart(PDF_diff,args):    
    plt.figure(figsize=(40, 30))
    fig, ax = plt.subplots()
    # plots = ['JS_tra','MMD_tra','JS_test','MMD_test']
    plots = list(PDF_diff.keys()) # 取出字典中的键
    colors_map = {'JS_val': 'r', 'JS_tra': 'r', 'MMD_tra': 'g','JS_test': 'r','MMD_test':'g'}
    markers_map = {'JS_val': 'o', 'JS_tra': 'o', 'MMD_tra': '^','JS_test': '+','MMD_test':'x'}

    for plot in plots:
        # plt.plot(range(0, len(PDF_diff[plot])), PDF_diff[plot], color = colors_map[plot], label=str(plot) + " PDF_diff")
        # plt.plot([0, 1]+[args.plot_interval * i for i in list(range(1, len(PDF_diff[plot])-1))], PDF_diff[plot], color = colors_map[plot], label=str(plot) + " PDF_diff")
        plt.plot([0, 1]+[args.plot_interval * i for i in list(range(1, len(PDF_diff[plot])-1))], PDF_diff[plot], color = colors_map[plot])
        plt.ticklabel_format(axis="both", style="sci", scilimits=(0,0),useMathText=True)   # 纵轴是科学计数法,1e6
        # plt.gca().ticklabel_format(useMathText=True) # 改成10为底的表示
        ax.get_yaxis().get_offset_text().set(va='bottom', ha='left')
        ax.yaxis.get_offset_text().set_fontsize(16)#设置1e6的大小与位置
        ax.xaxis.get_offset_text().set_fontsize(16)#设置1e6的大小与位置
       
        # markerfacecolor='none' 绘制空心marker
    # plt.title("Models Accuracies") 
    plt.tick_params(top='on', right='on', which='both') # 显示上侧和右侧的刻度
    plt.rcParams['xtick.direction'] = 'in' #将x轴的刻度线方向设置向内
    plt.rcParams['ytick.direction'] = 'in' #将y轴的刻度方向设置向内
    plt.xlabel("Epoch", fontsize=18)
    plt.ylabel("JS散度", fontsize=18)
    # plt.legend(loc="upper right", fontsize=16,facecolor=None,edgecolor=None, shadow=False,framealpha=1) # 给出图例，默认是plot中每根线条的label参数
    # ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: round(x * 10))) # 设置主坐标轴的刻度格式(小数位数),每10个epoch输出一个验证结果，所以横坐标要*10
    # plt.xticks(list(range(0, len(PDF_diff[plot]))), [args.plot_interval * i for i in list(range(0, len(PDF_diff[plot])))], fontsize=16)  # Set locations and labels
    plt.xticks(fontsize=16) # 设置刻度线格式
    plt.yticks(fontsize=16)
    plt.tight_layout()
    if not os.path.exists('results/png'):
        os.makedirs('results/png')
    if not os.path.exists('results/eps'):
        os.makedirs('results/eps')
    
    plt.savefig("{}/png/MMD&JS_{}_({},{}).png".format(args.result_dir, args.channel_type, args.n_channel, args.k))
    plt.savefig("{}/eps/MMD&JS_{}_({},{}).eps".format(args.result_dir, args.channel_type, args.n_channel, args.k))
    plt.close()

# 绘制区分两个分布的可视化图——————————————————————————————————————————————————————————————————————————————————————
def plot_embedding_2D(data, label, epoch,len_covert):
    x_min, x_max = np.min(data), np.max(data)
    data = (data - x_min) / (x_max - x_min) # 归一化到0-1
    fig = plt.figure(figsize=(4, 3.5))
    # for i in range(data.shape[0]):
    #     plt.text(data[i, 0], data[i, 1], str(label[i]),
    #              color=plt.cm.Set1(label[i]), # 根据输入i不同，输出不同颜色
    #              fontdict={'weight': 'bold', 'size': 9})

    plt.scatter(data[0:len_covert,0], data[0:len_covert,1], c="none", marker="o", edgecolors='red', s=12, label="Data")
    plt.scatter(data[len_covert:,0], data[len_covert:,1], c="none", marker="^", edgecolors='gray',s=12, label="Noise")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xticks(fontsize=8) # 设置刻度线格式
    plt.yticks(fontsize=8)
    plt.tick_params(top='on', right='on', which='both') # 显示上侧和右侧的刻度
    plt.rcParams['xtick.direction'] = 'in' #将x轴的刻度线方向设置向内
    plt.rcParams['ytick.direction'] = 'in' #将y轴的刻度方向设置向内
    plt.xlabel("Dimension 0", fontsize=10)
    plt.ylabel("Dimension 1", fontsize=10)
    # plt.title('Epoch:{}'.format(epoch))
    plt.legend(loc="upper left", framealpha=1, fontsize=8)
    return fig

def plot_embedding_3D(data,label,title): 
    x_min, x_max = np.min(data,axis=0), np.max(data,axis=0) 
    data = (data- x_min) / (x_max - x_min) 
    ax = plt.figure().add_subplot(111,projection='3d') 
    for i in range(data.shape[0]): 
        ax.text(data[i, 0], data[i, 1], data[i,2],str(label[i]), color=plt.cm.Set1(label[i]),fontdict={'weight': 'bold', 'size': 9}) 
    return ax


def TSNE_plot(covert_aftchannel, noise_aftchannel, args, epoch): 
    data = torch.cat([covert_aftchannel, noise_aftchannel], dim=0).detach().numpy()
    label = np.concatenate((np.zeros((len(covert_aftchannel),)), np.ones((len(noise_aftchannel),))), axis=0)
    n_samples, n_features = data.shape
    
    tsne_2D = TSNE(n_components = 2, init = 'pca', random_state = 0) #调用TSNE， 2D可视化, random_state=0
    result_2D = tsne_2D.fit_transform(data)
    # tsne_3D = TSNE(n_components=3, init='pca', random_state=0) # 3D可视化
    # result_3D = tsne_3D.fit_transform(data)
    fig1 = plot_embedding_2D(result_2D, label, epoch,len(covert_aftchannel))
    # plt.show()
    if not os.path.exists('results/png'):
        os.makedirs('results/png')
    if not os.path.exists('results/eps'):
        os.makedirs('results/eps')
    plt.savefig("{}/png/TSNE_epoch_{}_({},{}).png".format(args.result_dir, epoch, args.n_channel, args.k))
    plt.savefig("{}/eps/TSNE_epoch_{}_({},{}).eps".format(args.result_dir, epoch, args.n_channel, args.k))

    # fig2 = plot_embedding_3D(result_3D, label,'tSNE-3D')
    # plt.show()
    plt.close()

# 绘制两维星座图 ————————————————————————————————————————————————————————————————————————————————————————————————
def constellation_image(sample_data, real_data, args, epoch):
    sample_data = sample_data.detach().cpu().numpy()
    real_data = real_data.detach().cpu().numpy()
    plt.figure(figsize=(4,3.5)) 
    plt.scatter(sample_data[:,0], sample_data[:,1], c="red", marker="+", s=10, label="Data")
    plt.scatter(real_data[:,0], real_data[:,1], c="gray", marker=".", s=10, label="Noise")


    # plt.title('Distribution of Real data and Fake data from' + model_name, fontsize=17)
    plt.xticks(fontsize=8) # 设置刻度线格式
    plt.yticks(fontsize=8)
    plt.tick_params(top='on', right='on', which='both') # 显示上侧和右侧的刻度
    plt.rcParams['xtick.direction'] = 'in' #将x轴的刻度线方向设置向内
    plt.rcParams['ytick.direction'] = 'in' #将y轴的刻度方向设置向内
    plt.xlabel("y[0]", fontsize=10)
    plt.ylabel("y[1]", fontsize=10)
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)
    # plt.title('Epoch:{}'.format(epoch))
    # plt.rcParams.update({'font.size': 15})
    plt.legend(loc="upper left", framealpha=1, fontsize=8)

    plt.tight_layout()
    if not os.path.exists('results/png'):
        os.makedirs('results/png')
    if not os.path.exists('results/eps'):
        os.makedirs('results/eps')

    plt.savefig("{}/png/scatter2D_epoch_{}_({},{}).png".format(args.result_dir, epoch, args.n_channel, args.k))
    plt.savefig("{}/eps/scatter2D_epoch_{}_({},{}).eps".format(args.result_dir, epoch, args.n_channel, args.k))
    # plt.show()
    plt.close()

#——————————————————————————————————————————————————————————————————————————————————————————————————————
def main(args):
    # 读取文件
    accs_dict_path = os.path.join(args.saveoutput_dir, 'accs_lognormal{}_{}.npy'.format(args.n_channel, args.k))
    accs = np.load(accs_dict_path, allow_pickle=True)    # 输出即为Dict 类型
    accs=dict(accs.tolist())

    loss_dict_path = os.path.join(args.saveoutput_dir, 'loss_lognormal{}_{}.npy'.format(args.n_channel, args.k))
    loss = np.load(loss_dict_path, allow_pickle=True)   # 输出即为Dict 类型
    loss=dict(loss.tolist())

    PDF_diff_dict_path = os.path.join(args.saveoutput_dir, 'PDF_diff_lognormal{}_{}.npy'.format(args.n_channel, args.k))
    PDF_diff = np.load(PDF_diff_dict_path,allow_pickle=True)   # 输出即为Dict 类型
    PDF_diff=dict(PDF_diff.tolist())

    ber_dict_path = os.path.join(args.saveoutput_dir, 'ber_lognormal{}_{}.npy'.format(args.n_channel, args.k))
    ber = np.load(ber_dict_path,allow_pickle=True)   # 输出即为Dict 类型
    ber=dict(ber.tolist())

    # 绘图 ————————————————————————————————————————————————————————————————————————————————————————
    losses_chart(loss,args)
    PDFdiff_chart(PDF_diff,args)
    bler_chart(accs, args)
    ber_chart(ber, args)


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
        parser.add_argument('--plot_interval', type=int, default = 50) # 绘图epoch间隔 100
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

    
    
