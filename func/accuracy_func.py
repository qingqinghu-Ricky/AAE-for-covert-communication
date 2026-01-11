import torch
import copy
import numpy as np

# 计算accuracy的函数——————————————————————————————————————————————————————————————————————————————————————
def accuracy(pred, label):
    return torch.sum(pred == label).item() / len(pred)

def willie_accuracy(covert_preds_, normal_preds_): # 发送covert信号的评分，发送normal信号的评分
    covert_preds = copy.deepcopy(covert_preds_.detach()) # 深复制复刻，从而建立一个完全的新变量，改动不会影响原始变量
    normal_preds = copy.deepcopy(normal_preds_.detach())
    covert_preds = covert_preds.cpu()
    normal_preds = normal_preds.cpu()
    covert_preds[covert_preds > 0.5] = 1
    covert_preds[covert_preds <= 0.5] = 0
    normal_preds[normal_preds > 0.5] = 1
    normal_preds[normal_preds <= 0.5] = 0
    discriminator_covert_accuracy = accuracy(covert_preds, torch.zeros(len(covert_preds), 1))
    discriminator_normal_accuracy = accuracy(normal_preds, torch.ones(len(normal_preds), 1))
    # print(discriminator_covert_accuracy, discriminator_normal_accuracy)
    return (discriminator_covert_accuracy + discriminator_normal_accuracy) / 2 # 是否发送判断成功的平均准确度

def alice_accuracy(covert_preds_): # 发送covert信号的评分，发送normal信号的评分
    covert_preds = copy.deepcopy(covert_preds_.detach())
    covert_preds = covert_preds.cpu()
    covert_preds[covert_preds > 0.5] = 1
    covert_preds[covert_preds <= 0.5] = 0
    discriminator_covert_accuracy = accuracy(covert_preds, torch.ones(len(covert_preds), 1))
    return discriminator_covert_accuracy    # 发送covert信号判为1的概率

def bob_accuracy(pred, label):
    _, preds = torch.max(pred, dim=1)
    # import pdb; pdb.set_trace()
    return accuracy(preds, label)

def bob_ber(pred, label, args):
    _, preds = torch.max(pred, dim=1)
    preds_np = preds.numpy()
    label_np = label.numpy()
    preds_bool = decimal2binary(preds_np, args)
    label_bool = decimal2binary(label_np, args)
    xor_list = [x ^ y for x, y in zip(preds_bool, label_bool)]

    xor_np = np.array(xor_list)
    ber = xor_np.sum()/len(xor_np)
    return(ber)


def decimal2binary(inputs,args):   # 将十进制数组转换成二进制数组
    # 创建空白的二进制数组
    outputs = []
 
    # 遍历每个十进制数字，并将其转换为二进制形式后添加到二进制数组中
    for decimal in inputs:
        binary = bin(decimal)[2:] # 使用bin()函数获取二进制表示，然后去除前缀'0b'
        binary_bool = [bool(temp) for temp in binary]
        while(len(binary_bool)<args.k):  # 小于指定位置前面添0
            binary_bool=[False]+binary_bool
        outputs.extend(binary_bool)
    return outputs


