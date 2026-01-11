# 同一训练后的model在多个snr下的测试结果

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

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from torch.utils.data import DataLoader
import argparse
import matplotlib.pyplot as plt
import copy


# 绘制准确度的图——————————————————————————————————————————————————————————————————————————————————————————————
def test_accuracies_chart(SNR, accs, args):    
    plt.figure(figsize=(40, 30))
    fig, ax = plt.subplots()

    plots = list(accs.keys()) # 取出字典中的键
    # colors_map = {'bob_test': 'r', 'willie_test': 'g','alice_test': 'b'}
    # markers_map = {'bob_test': 'o','willie_test':'^','alice_test': 's'}

    # colors_map = {'8_2': 'r', '8_4': 'g','8_6': 'b'}
    # markers_map = {'8_2': 'o','8_4':'^','8_6': 's'}

    # colors_map = {'8_4': 'r', '10_5': 'g','12_6': 'b'}
    # markers_map = {'8_4': 'o','10_5':'^','12_6': 's'}
    
    # colors_map = {'8_1': 'r', '8_4': 'g'}
    # markers_map = {'8_1': 'o','8_4':'^'}
    
    # colors_map = {'8_4': 'r', '12_6': 'g'}
    # markers_map = {'8_4': 'o','12_6':'^'}
    
    colors_map = {'15db': 'r', '10db': 'g','mix': 'b', 'increase':'k', 'decrease': 'y'}
    markers_map = {'15db': 'o','10db':'^','mix': 's','increase':'>', 'decrease': '<'}
    

    for plot in plots:
        plt.plot(SNR, accs[plot], color = colors_map[plot], marker=markers_map[plot], markerfacecolor='none', label=str(plot).capitalize() + " Acc")
        # plt.plot(SNR, accs[plot], color = colors_map[plot], label=str(plot).capitalize() + " Acc")

    # plt.title("Models Accuracies")
    plt.tick_params(top='on', right='on', which='both') # 显示上侧和右侧的刻度
    plt.rcParams['xtick.direction'] = 'in' #将x轴的刻度线方向设置向内
    plt.rcParams['ytick.direction'] = 'in' #将y轴的刻度方向设置向内
    plt.xlabel("SNR (dB)", fontsize=18)
    plt.ylabel("Accuracy of Bob", fontsize=18)
    plt.legend(loc="center right", fontsize=16, facecolor=None,edgecolor=None, shadow=False,framealpha=1) # 给出图例，默认是plot中每根线条的label参数
    # ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: round(x * 10))) # 设置主坐标轴的刻度格式(小数位数),每10个epoch输出一个验证结果，所以横坐标要*10
    # plt.xticks(list(range(0, len(accs[plot]))), [args.plot_interval * i for i in list(range(0, len(accs[plot])))], fontsize=16)  # Set locations and labels
    plt.xticks(fontsize=16) # 设置刻度线格式
    plt.yticks(fontsize=16)
    plt.tight_layout()
    if not os.path.exists('results/png'):
        os.makedirs('results/png')
    if not os.path.exists('results/eps'):
        os.makedirs('results/eps')
    
    plt.savefig("{}/png/test_acc_{}_({},{}).png".format(args.result_dir, args.channel_type, args.n_channel, args.k))
    plt.savefig("{}/eps/test_acc_{}_({},{}).eps".format(args.result_dir, args.channel_type, args.n_channel, args.k))
    plt.close()

# 绘制BLER的图——————————————————————————————————————————————————————————————————————————————————————————————
def test_BLER_chart(SNR, BLER, args): 

    plt.rcParams['xtick.direction'] = 'in' #将x轴的刻度线方向设置向内   # 设置坐标轴刻度朝外的代码必须放到前面
    plt.rcParams['ytick.direction'] = 'in' #将y轴的刻度方向设置向内 

    plt.figure(figsize=(40, 30))
    fig, ax = plt.subplots()
    plt.tick_params(top='on', right='on', which='both') # 显示上侧和右侧的刻度

    plots = list(BLER.keys()) # 取出字典中的键
    
    # colors_map = {'-10dB': 'r', '-15dB': 'g','-20dB': 'b', '-10~-20dB':'k'}
    # markers_map = {'-10dB': 'o','-15dB':'^','-20dB': 's','-10~-20dB':'>'}

    # colors_map = {'(8,4)': 'r', '(10,5)': 'g','(12,6)': 'b'}
    # markers_map = {'(8,4)': 'o','(10,5)':'^','(12,6)': 's'}

    colors_map = {'(8,2)': 'r', '(8,4)': 'g','(8,6)': 'b'}
    markers_map = {'(8,2)': 'o','(8,4)':'^','(8,6)': 's'}


    
    for plot in plots:
        plt.semilogy(SNR, BLER[plot], color = colors_map[plot], marker=markers_map[plot], markerfacecolor='none', label=str(plot))


    # plt.title("Models Accuracies")

    plt.xlabel("Noise power (dB)", fontsize=18)
    plt.ylabel("BLER", fontsize=18)
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
    
    plt.savefig("{}/png/test_bler_{}_({},{}).png".format(args.result_dir, args.channel_type, args.n_channel, args.k))
    plt.savefig("{}/eps/test_bler_{}_({},{}).eps".format(args.result_dir, args.channel_type, args.n_channel, args.k))
    plt.close()

# 绘制BER的图——————————————————————————————————————————————————————————————————————————————————————————————
def test_BER_chart(SNR, BER, args): 

    plt.rcParams['xtick.direction'] = 'in' #将x轴的刻度线方向设置向内   # 设置坐标轴刻度朝外的代码必须放到前面
    plt.rcParams['ytick.direction'] = 'in' #将y轴的刻度方向设置向内 

    plt.figure(figsize=(40, 30))
    fig, ax = plt.subplots()
    plt.tick_params(top='on', right='on', which='both') # 显示上侧和右侧的刻度

    plots = list(BER.keys()) # 取出字典中的键
    
    # colors_map = {'-10dB': 'r', '-15dB': 'g','-20dB': 'b', '-10~-20dB':'k'}
    # markers_map = {'-10dB': 'o','-15dB':'^','-20dB': 's','-10~-20dB':'>'}

    # colors_map = {'(8,4)': 'r', '(10,5)': 'g','(12,6)': 'b'}
    # markers_map = {'(8,4)': 'o','(10,5)':'^','(12,6)': 's'}

    colors_map = {'(8,2)': 'r', '(8,4)': 'g','(8,6)': 'b'}
    markers_map = {'(8,2)': 'o','(8,4)':'^','(8,6)': 's'}


    
    for plot in plots:
        plt.semilogy(SNR, BER[plot], color = colors_map[plot], marker=markers_map[plot], markerfacecolor='none', label=str(plot))


    # plt.title("Models Accuracies")

    plt.xlabel("Noise power (dB)", fontsize=18)
    plt.ylabel("BER", fontsize=18)
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
    
    plt.savefig("{}/png/test_ber_{}_({},{}).png".format(args.result_dir, args.channel_type, args.n_channel, args.k))
    plt.savefig("{}/eps/test_ber_{}_({},{}).eps".format(args.result_dir, args.channel_type, args.n_channel, args.k))
    plt.close()

# 绘制PDFdiff的图——————————————————————————————————————————————————————————————————————————————————————————————
def test_PDFdiff_chart(SNR, PDF_diff, args):    
    plt.figure(figsize=(40, 30))
    fig, ax = plt.subplots()

    plots = list(PDF_diff.keys()) # 取出字典中的键
    # colors_map = {'JS_test': 'r' }
    # markers_map = {'JS_test': 'o'}

    # colors_map = {'8_2': 'r', '8_4': 'g','8_6': 'b'}
    # markers_map = {'8_2': 'o','8_4':'^','8_6': 's'}

    # colors_map = {'8_4': 'r', '10_5': 'g','12_6': 'b'}
    # markers_map = {'8_4': 'o','10_5':'^','12_6': 's'}
    
    # colors_map = {'8_1': 'r', '8_4': 'g'}
    # markers_map = {'8_1': 'o','8_4':'^'}
    
    # colors_map = {'8_4': 'r', '12_6': 'g'}
    # markers_map = {'8_4': 'o','12_6':'^'}
    
    # colors_map = {'15db': 'r', '10db': 'g','mix': 'b', 'increase':'k', 'decrease': 'y'}
    # markers_map = {'15db': 'o','10db':'^','mix': 's','increase':'>', 'decrease': '<'}

    # colors_map = {'-10dB': 'r', '-15dB': 'g','-20dB': 'b', '-10~-20dB':'k'}
    # markers_map = {'-10dB': 'o','-15dB':'^','-20dB': 's','-10~-20dB':'>'}


    # colors_map = {'(8,4)': 'r', '(10,5)': 'g','(12,6)': 'b'}
    # markers_map = {'(8,4)': 'o','(10,5)':'^','(12,6)': 's'}

    colors_map = {'(8,2)': 'r', '(8,4)': 'g','(8,6)': 'b'}
    markers_map = {'(8,2)': 'o','(8,4)':'^','(8,6)': 's'}

    for plot in plots:
        plt.plot(SNR, PDF_diff[plot], color = colors_map[plot], marker=markers_map[plot], markerfacecolor='none', label=str(plot))
        plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))   # 纵轴是科学计数法,1e6
        plt.gca().ticklabel_format(useMathText=True) # 改成10为底的表示
        ax.get_yaxis().get_offset_text().set(va='bottom', ha='left')
        ax.yaxis.get_offset_text().set_fontsize(16)#设置1e6的大小与位置
        # plt.plot(SNR, PDF_diff[plot], color = colors_map[plot], label=str(plot).capitalize() + " JS div")

    # plt.title("Models Accuracies")
    plt.tick_params(top='on', right='on', which='both') # 显示上侧和右侧的刻度
    plt.rcParams['xtick.direction'] = 'in' #将x轴的刻度线方向设置向内
    plt.rcParams['ytick.direction'] = 'in' #将y轴的刻度方向设置向内
    plt.xlabel("Noise power (dB)", fontsize=18)
    plt.ylabel("JS divergence", fontsize=18)
    plt.legend(loc="upper right", fontsize=16, facecolor=None,edgecolor=None, shadow=False,framealpha=1) # 给出图例，默认是plot中每根线条的label参数
    # ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: round(x * 10))) # 设置主坐标轴的刻度格式(小数位数),每10个epoch输出一个验证结果，所以横坐标要*10
    # plt.xticks(list(range(0, len(accs[plot]))), [args.plot_interval * i for i in list(range(0, len(accs[plot])))], fontsize=16)  # Set locations and labels
    plt.xticks(fontsize=16) # 设置刻度线格式
    plt.yticks(fontsize=16)
    plt.tight_layout()
    if not os.path.exists('results/png'):
        os.makedirs('results/png')
    if not os.path.exists('results/eps'):
        os.makedirs('results/eps')
    
    plt.savefig("{}/png/test_js_{}_({},{}).png".format(args.result_dir, args.channel_type, args.n_channel, args.k))
    plt.savefig("{}/eps/test_js_{}_({},{}).eps".format(args.result_dir, args.channel_type, args.n_channel, args.k))
    plt.close()


def main(args):
    # 测试acc 随snr变化曲线——————————————————————————————————————————————————————————————————————
    # # accs = {'willie_test': [], 'bob_test': [],'alice_test':[]}  # 不同snr下的测试准确度
    # accs = {'willie_test': [], 'bob_test': []}  # 不同snr下的测试准确度
    # accs['willie_test']=[0.500,0.500,0.500,0.500,0.500,0.500]
    # # accs['bob_test']=[0.295703, 0.564941, 0.895703, 0.993945, 1.000000, 1.000000] # 15db
    # accs['bob_test']=[0.412988, 0.770117, 0.983398, 0.999902, 1, 1] # 10 db
    # # accs['alice_test']=[1.000,1.000,1.000,1.000,1.000,1.000]


    # # PDF_diff = {'JS_test':[1.472648e-04, 6.967022e-05, 2.023999e-04, 6.813169e-04, 8.915341e-04, 9.741958e-04]}
    # PDF_diff = {'JS_test':[2.414768e-4, 8.211769e-4, 4.294218e-3, 9.495356e-3, 1.776279e-2, 2.351209e-2]}

    # SNR = [5,10,15,20,25,30]
    # test_accuracies_chart(SNR, accs, args)
    # test_PDFdiff_chart(SNR, PDF_diff, args)

    # 不同码率测试acc 随snr变化曲线————————————————————————————————————————————————————————————————————
    # accs0 = {'8_1': [], '8_4': []}  # 不同snr下的测试准确度
    # accs0['8_1']=[0.698242, 0.834766, 0.964063, 0.998633, 1.000000, 1.000000]
    # accs0['8_4']=[0.295703, 0.564941, 0.895703, 0.993945, 1.000000, 1.000000]

    # PDF_diff0 = {'8_1': [], '8_4': []}  # 不同snr下的测试准确度
    # PDF_diff0['8_1']=[1.036853e-04, 6.833291e-05, 3.239240e-04, 6.171240e-04, 5.268980e-04, 6.002429e-04]
    # PDF_diff0['8_4']=[1.472648e-04, 6.967022e-05, 2.023999e-04, 6.813169e-04, 8.915341e-04, 9.741958e-04]
  

    # SNR = [5,10,15,20,25,30]
    # test_accuracies_chart(SNR, accs0, args)
    # test_PDFdiff_chart(SNR, PDF_diff0, args)

    # 不同码长测试acc 随snr变化曲线————————————————————————————————————————————————————————————————————
    # accs0 = {'8_4': [], '12_6':[]}  # 不同snr下的测试准确度
    # accs0['8_4']=[0.295703, 0.564941, 0.895703, 0.993945, 1.000000, 1.000000]  # acc_b
    # accs0['12_6']=[0.233887, 0.588379, 0.943262, 0.998535, 1.000000, 1.000000]

    # PDF_diff0 = {'8_4': [], '12_6':[]}  # 不同snr下的测试准确度
    # PDF_diff0['8_4']=[1.472648e-04, 6.967022e-05, 2.023999e-04, 6.813169e-04, 8.915341e-04, 9.741958e-04]
    # PDF_diff0['12_6']=[7.799751e-05, 8.799976e-05, 9.176550e-05, 2.675610e-04, 4.180905e-04, 5.361716e-04]

    # SNR = [5,10,15,20,25,30]
    # test_accuracies_chart(SNR, accs0, args)
    # test_PDFdiff_chart(SNR, PDF_diff0, args)
    
    # 不同固定snr下的训练模型在测试集的acc 随snr变化曲线————————————————————————————————————————————————————————————————————
    # accs0 = {'15db': [], '10db': [], 'mix': [], 'increase':[], 'decrease':[]}  # 不同snr下的测试准确度
    # accs0['15db']=[0.295703, 0.564941, 0.895703, 0.993945, 1.000000, 1.000000]
    # accs0['10db']=[0.412988, 0.770117, 0.983398, 0.999902, 1, 1]
    # accs0['mix']=[0.334473, 0.635449, 0.945117, 0.998437, 1.000000, 1.000000] # 15,10db随机选择训练
    # accs0['increase']=[0.304883, 0.589746, 0.916699, 0.996973, 1.000000, 1.000000] # 10db到15db 递增训练
    # accs0['decrease']=[0.363770, 0.691406, 0.963672, 0.999609, 1.000000, 1.000000] # 10db到15db 递减训练

    # PDF_diff0 = {'15db': [], '10db': [], 'mix': [], 'increase':[], 'decrease':[]}   # 不同snr下的测试准确度
    # PDF_diff0['15db']=[1.472648e-04, 6.967022e-05, 2.023999e-04, 6.813169e-04, 8.915341e-04, 9.741958e-04]
    # PDF_diff0['10db']=[2.414768e-4, 8.211769e-4, 4.294218e-3, 9.495356e-3, 1.776279e-2, 2.351209e-2]
    # PDF_diff0['mix']=[1.200775e-04, 2.068888e-04, 1.198518e-03, 3.344199e-03, 4.893267e-03, 5.798990e-03]
    # PDF_diff0['increase']=[9.478610e-05, 8.141982e-05, 1.365614e-04, 2.556815e-04, 3.360615e-04, 5.395734e-04] # 10db到15db 递增训练
    # PDF_diff0['decrease']=[1.655926e-04, 5.142833e-04, 2.479341e-03, 6.057364e-03, 8.276907e-03, 9.371150e-03]
  

    # SNR = [5,10,15,20,25,30]
    # test_accuracies_chart(SNR, accs0, args)
    # test_PDFdiff_chart(SNR, PDF_diff0, args) 


    # 不同固定snr下的训练模型在测试集的BLER 随snr变化曲线————————————————————————————————————————————————————————————————————
    # BLER0 = {'-10dB': [], '-15dB': [], '-20dB': [],'-10~-20dB': []}  # 不同snr下的测试准确度

    # # BLER0['-20dB']=np.subtract(1, [0.346875, 0.473242, 0.708496, 0.847852, 0.968750])
    # # BLER0['-15dB']=np.subtract(1,[0.542773, 0.706152, 0.900488, 0.965137, 0.995605])
    # # BLER0['-10dB']=np.subtract(1,[0.713574, 0.846094, 0.968848, 0.993848, 0.999902])
    # # BLER0['-10~-20dB']=np.subtract(1,[0.587891, 0.727051, 0.914355, 0.970313, 0.996387]) # 10db到20db随机选择训练

    # # BLER0['-20dB']=np.subtract(1, [0.350228, 0.471973, 0.710872, 0.848372, 0.965723])
    # # BLER0['-15dB']=np.subtract(1,[0.540755, 0.702897, 0.902767, 0.967871, 0.995833])
    # # BLER0['-10dB']=np.subtract(1,[0.710059, 0.851497, 0.967741, 0.993392, 0.999707])
    # # # BLER0['-10~-20dB']=np.subtract(1,[0.582292, 0.730957, 0.909180, 0.970540, 0.996582]) # 10db到20db随机选择训练,4000epoch
    # # BLER0['-10~-20dB']=np.subtract(1,[0.567350, 0.709993, 0.897526, 0.964225, 0.995573]) # 10db到20db随机选择训练,8000epoch

    # BLER0['-20dB']=np.subtract(1, [0.348434, 0.474092, 0.707008, 0.847881, 0.967096])
    # BLER0['-15dB']=np.subtract(1,[0.537415, 0.704502, 0.902122, 0.967096, 0.995518])
    # BLER0['-10dB']=np.subtract(1,[0.712565, 0.848011, 0.968210, 0.993464, 0.999785])
    # BLER0['-10~-20dB']=np.subtract(1,[0.565863, 0.714144, 0.896663, 0.963682, 0.995472]) # 10db到20db随机选择训练,8000epoch
    

    # BER0 = {'-10dB': [], '-15dB': [], '-20dB': [],'-10~-20dB': []}  # 不同snr下的测试准确度


    # # BER0['-20dB']=[1.859701e-01, 1.540853e-01, 8.715820e-02, 4.619141e-02, 1.078288e-02]
    # # BER0['-15dB']=[1.240885e-01, 8.085124e-02, 2.606608e-02, 8.683268e-03, 9.602865e-04]
    # # BER0['-10dB']=[8.912760e-02, 4.491374e-02, 1.017253e-02, 1.879883e-03, 4.882813e-05]
    # # # BLER0['-10~-20dB']=np.subtract(1,[0.582292, 0.730957, 0.909180, 0.970540, 0.996582]) # 10db到20db随机选择训练,4000epoch
    # # BER0['-10~-20dB']=[1.337891e-01, 9.140625e-02, 3.331706e-02, 1.155599e-02, 1.619466e-03] # 10db到20db随机选择训练,8000epoch

    # BER0['-20dB']=[1.851750e-01, 1.532275e-01, 8.864746e-02, 4.669434e-02, 1.032878e-02]
    # BER0['-15dB']=[1.256893e-01, 8.005208e-02, 2.609456e-02, 8.840332e-03, 1.234538e-03]
    # BER0['-10dB']=[8.720459e-02, 4.662760e-02, 9.888509e-03, 2.072754e-03, 7.649740e-05]
    # BER0['-10~-20dB']=[1.339836e-01, 8.990072e-02, 3.337402e-02, 1.208903e-02, 1.586914e-03] # 10db到20db随机选择训练,8000epoch

    # PDF_diff0 = {'-10dB': [], '-15dB': [], '-20dB': [],'-10~-20dB': []}  # 不同snr下的测试准确度
    # # PDF_diff0['-20dB']=[5.755255e-05, 7.052563e-05, 1.750413e-04, 1.337434e-04, 1.754547e-04]
    # # PDF_diff0['-15dB']=[1.899213e-04, 1.480457e-04, 3.222679e-04, 4.818234e-04, 6.744928e-04]
    # # PDF_diff0['-10dB']=[1.058175e-04, 1.772953e-04, 4.876498e-04, 7.998410e-04, 1.146039e-03]
    # # PDF_diff0['-10~-20dB']=[1.775062e-04, 3.375010e-04, 9.010840e-04, 1.259800e-03, 1.863963e-03] # 10db到20db随机选择训练

    # # PDF_diff0['-20dB']=[5.895997e-05, 4.271648e-05, 3.804943e-05, 9.697967e-05, 1.227796e-04]
    # # PDF_diff0['-15dB']=[1.543322e-04, 1.480389e-04, 2.327050e-04, 3.132881e-04, 5.763112e-04]
    # # PDF_diff0['-10dB']=[7.955056e-05, 2.179047e-04, 3.951432e-04, 6.507012e-04, 1.139166e-03]
    # # # PDF_diff0['-10~-20dB']=[1.791006e-04, 3.881797e-04, 7.787936e-04, 1.265447e-03, 1.927761e-03] # 10db到20db随机选择训练,4000epoch
    # # PDF_diff0['-10~-20dB']=[1.490300e-04, 1.841035e-04, 3.523673e-04, 5.267942e-04, 8.759377e-04] # 10db到20db随机选择训练,8000epoch

    # PDF_diff0['-20dB']=[1.995348e-05, 1.446502e-05, 4.819992e-05, 5.847482e-05, 8.870598e-05]
    # PDF_diff0['-15dB']=[9.862966e-05, 1.427813e-04, 2.759852e-04, 3.422411e-04, 5.138941e-04]
    # PDF_diff0['-10dB']=[8.268716e-05, 1.772775e-04, 4.411464e-04, 6.420026e-04, 1.003603e-03]
    # PDF_diff0['-10~-20dB']=[1.159999e-04, 1.845829e-04, 3.969095e-04, 5.518178e-04, 7.762242e-04] # 10db

    # SNR = [-10, -12,-15,-17,-20]
    # test_BLER_chart(SNR, BLER0, args)
    # test_BER_chart(SNR, BER0, args)
    # test_PDFdiff_chart(SNR, PDF_diff0, args) 


    # 不同码长测试 随snr变化曲线————————————————————————————————————————————————————————————————————
    # BLER0 = {'(8,4)': [], '(10,5)': [], '(12,6)':[]}  # 固定15db下的BLER

    # # BLER0['(8,4)']=np.subtract(1,[0.542773, 0.706152, 0.900488, 0.965137, 0.995605])  
    # # BLER0['(10,5)']=np.subtract(1,[0.508545, 0.682373, 0.904419, 0.965454, 0.997070]) 
    # # BLER0['(12,6)']=np.subtract(1,[0.495019, 0.692939, 0.913126, 0.973191, 0.997217])

    # # BLER0['(8,4)']=np.subtract(1,[0.540755, 0.702897, 0.902767, 0.967871, 0.995833])  
    # # BLER0['(10,5)']=np.subtract(1,[0.508830, 0.683146, 0.900146, 0.968750, 0.996053]) 
    # # BLER0['(12,6)']=np.subtract(1,[0.503662, 0.691455, 0.915186, 0.975244, 0.997119])

    # BLER0['(8,4)']=np.subtract(1,[0.537415, 0.704502, 0.902122, 0.967096, 0.995518])  
    # BLER0['(10,5)']=np.subtract(1,[0.504215, 0.681453, 0.899316, 0.967875, 0.996122]) 
    # BLER0['(12,6)']=np.subtract(1,[0.505854, 0.696143, 0.914858, 0.975415, 0.996851])

    # # BER0 = {'(8,4)': [], '(10,5)': [], '(12,6)':[]}  # 固定15db下的BER
    # # BER0['(8,4)']=[1.240885e-01, 8.085124e-02, 2.606608e-02, 8.683268e-03, 9.602865e-04]
    # # BER0['(10,5)']=[1.248291e-01, 7.963867e-02, 2.559408e-02, 8.300781e-03, 9.684245e-04]
    # # BER0['(12,6)']=[1.100749e-01, 6.822917e-02, 1.726888e-02, 5.460612e-03, 6.673177e-04]

    # BER0 = {'(8,4)': [], '(10,5)': [], '(12,6)':[]}  # 固定15db下的BER
    # BER0['(8,4)']=[1.256893e-01, 8.005208e-02, 2.609456e-02, 8.840332e-03, 1.234538e-03]
    # BER0['(10,5)']=[1.267684e-01, 7.998535e-02, 2.508870e-02, 8.097331e-03, 1.023763e-03]
    # BER0['(12,6)']=[1.100098e-01, 6.643311e-02, 1.811930e-02, 5.204264e-03, 7.096354e-04]
    

    # PDF_diff0 = {'(8,4)': [], '(10,5)': [], '(12,6)':[]}  # 不同snr下的BLER

    # # PDF_diff0['(8,4)']=[1.899213e-04, 1.480457e-04, 3.222679e-04, 4.818234e-04, 6.744928e-04]  
    # # PDF_diff0['(10,5)']=[1.506418e-04, 2.158479e-04, 5.260221e-04, 5.281673e-04, 6.889117e-04] 
    # # PDF_diff0['(12,6)']=[1.147916e-04, 2.442370e-04, 3.884717e-04, 4.167257e-04, 4.903106e-04]

    # # PDF_diff0['(8,4)']=[1.543322e-04, 1.480389e-04, 2.327050e-04, 3.132881e-04, 5.763112e-04]  
    # # PDF_diff0['(10,5)']=[1.085249e-04, 1.688666e-04, 2.277357e-04, 4.528795e-04, 7.221215e-04] 
    # # PDF_diff0['(12,6)']=[9.359087e-05, 1.642930e-04, 2.289719e-04, 3.160734e-04, 4.241605e-04]

    # PDF_diff0['(8,4)']=[9.862966e-05, 1.427813e-04, 2.759852e-04, 3.422411e-04, 5.138941e-04]  
    # PDF_diff0['(10,5)']=[8.576857e-05, 1.420318e-04, 3.298021e-04, 4.464386e-04, 6.799591e-04] 
    # PDF_diff0['(12,6)']=[9.044787e-05, 1.309198e-04, 2.342435e-04, 2.973527e-04, 3.534275e-04]


    

    # SNR = [-10, -12,-15,-17,-20]
    # test_BLER_chart(SNR, BLER0, args)
    # test_BER_chart(SNR, BER0, args)
    # test_PDFdiff_chart(SNR, PDF_diff0, args)

    # 不同码率测试 随snr变化曲线————————————————————————————————————————————————————————————————————
    BLER0 = {'(8,2)': [], '(8,4)': [], '(8,6)':[]}  # 固定15db下的BLER
    # BLER0['(8,2)']=np.subtract(1,[0.631641, 0.738623, 0.874658, 0.943066, 0.986133])  
    # BLER0['(8,4)']=np.subtract(1,[0.542773, 0.706152, 0.900488, 0.965137, 0.995605]) 
    # BLER0['(8,6)']=np.subtract(1,[0.456490, 0.629212, 0.870202, 0.961617, 0.995459])

    # BLER0['(8,2)']=np.subtract(1,[0.632926, 0.739095, 0.881413, 0.943148, 0.987484])  
    # BLER0['(8,4)']=np.subtract(1,[0.540755, 0.702897, 0.902767, 0.967871, 0.995833]) 
    # BLER0['(8,6)']=np.subtract(1,[0.446191, 0.625244, 0.870215, 0.956250, 0.994189])

    BLER0['(8,2)']=np.subtract(1,[0.635545, 0.739089, 0.879645, 0.942520, 0.986880])  
    BLER0['(8,4)']=np.subtract(1,[0.537415, 0.704502, 0.902122, 0.967096, 0.995518]) 
    BLER0['(8,6)']=np.subtract(1,[0.449824, 0.626831, 0.868223, 0.956304, 0.994741])


    BER0 = {'(8,2)': [], '(8,4)': [], '(8,6)':[]}  # 固定15db下的BER
    
    # BER0['(8,2)']=[1.155680e-01, 8.269857e-02, 3.815104e-02, 1.787109e-02, 3.833008e-03]
    # BER0['(8,4)']=[1.240885e-01, 8.085124e-02, 2.606608e-02, 8.683268e-03, 9.602865e-04]
    # BER0['(8,6)']=[1.236491e-01, 8.212891e-02, 2.802734e-02, 8.699544e-03, 1.212565e-03]

    BER0['(8,2)']=[1.152987e-01, 8.271159e-02, 3.828288e-02, 1.828776e-02, 3.929850e-03]
    BER0['(8,4)']=[1.256893e-01, 8.005208e-02, 2.609456e-02, 8.840332e-03, 1.234538e-03]
    BER0['(8,6)']=[1.216390e-01, 8.150146e-02, 2.842448e-02, 9.056803e-03, 1.003418e-03]


    PDF_diff0 = {'(8,2)': [], '(8,4)': [], '(8,6)':[]}  # 不同snr下的BLER
    # PDF_diff0['(8,2)']=[5.621117e-05, 1.053060e-04, 8.704270e-05, 2.126377e-04, 3.686858e-04]  
    # PDF_diff0['(8,4)']=[1.899213e-04, 1.480457e-04, 3.222679e-04, 4.818234e-04, 6.744928e-04] 
    # PDF_diff0['(8,6)']=[2.930355e-04, 5.379288e-04, 6.845096e-04, 1.549538e-03, 1.392803e-03]

    # PDF_diff0['(8,2)']=[1.719641e-05, 2.595623e-05, 1.127999e-04, 1.911526e-04, 3.559136e-04]  
    # PDF_diff0['(8,4)']=[1.543322e-04, 1.480389e-04, 2.327050e-04, 3.132881e-04, 5.763112e-04] 
    # PDF_diff0['(8,6)']=[1.960031e-04, 4.978199e-04, 9.742498e-04, 1.047303e-03, 1.534161e-03]

    PDF_diff0['(8,2)']=[1.075290e-05, 4.123605e-05, 9.341110e-05, 1.665604e-04, 2.877077e-04]  
    PDF_diff0['(8,4)']=[9.862966e-05, 1.427813e-04, 2.759852e-04, 3.422411e-04, 5.138941e-04] 
    PDF_diff0['(8,6)']=[1.681937e-04, 3.434098e-04, 7.772716e-04, 1.069960e-03, 1.588397e-03]
    

    SNR = [-10, -12,-15,-17,-20]
    test_BLER_chart(SNR, BLER0, args)
    test_BER_chart(SNR, BER0, args)
    test_PDFdiff_chart(SNR, PDF_diff0, args)



if __name__ == '__main__': # 默认整个脚本从此处开始运行
    def parse_args():
        parser = argparse.ArgumentParser(description = "AutoEocoder")
        parser.add_argument('--channel_type', type=str, default = 'lognormal') # awgn, rayleigh，lognormal
        parser.add_argument('--train_size', type=int, default = 1024 * 10) #1024 * 8
        parser.add_argument('--test_size', type=int, default = 1024*1)
        parser.add_argument('--batch_size', type=int, default = 1024) 
        parser.add_argument('--n_channel', type=int, default = 8) # 信道编码后的码长n
        parser.add_argument('--k', type=int, default = 4) # k是发送的message的比特数
        parser.add_argument('--lr', type=int, default = 0.5e-4) # 2e-4, 1e-4 ########################################################
        parser.add_argument('--result_dir', type=str, default = 'results')
        parser.add_argument('--saveoutput_dir', type=str, default = 'saveoutput')
        parser.add_argument('--modelpara_dir', type=str, default = 'modelparameter')
        parser.add_argument('--seed', type=int, default = 100)
        parser.add_argument('--total_epoch', type=int, default = 1000)  # epoch总数
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