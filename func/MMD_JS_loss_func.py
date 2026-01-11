import torch
import torch.nn as nn
import scipy.stats
import numpy as np
import pandas as pd


class MMD_loss(nn.Module):
	def __init__(self, kernel_mul = 2.0, kernel_num = 5):
		super(MMD_loss, self).__init__()
		self.kernel_num = kernel_num
		self.kernel_mul = kernel_mul
		self.fix_sigma = None
		return
	def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
		n_samples = int(source.size()[0])+int(target.size()[0])
		total = torch.cat([source, target], dim=0)
		total = torch.unsqueeze(total, 1)

		total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1))) # 扩维和复制
		total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))

		
		L2_distance = ((total0-total1)**2).sum(2) 
		if fix_sigma:
			bandwidth = fix_sigma
		else:
			bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
		bandwidth /= kernel_mul ** (kernel_num // 2)
		bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
		kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
		return sum(kernel_val)

	def forward(self, source, target): # source: sample_size_1 * feature_size 的数据
		batch_size = int(source.size()[0])
		kernels = self.guassian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
		XX = kernels[:batch_size, :batch_size]
		YY = kernels[batch_size:, batch_size:]
		XY = kernels[:batch_size, batch_size:]
		YX = kernels[batch_size:, :batch_size]
		loss = torch.mean(XX + YY - XY -YX)
		return loss.item()


# 计算两个样本集合的JS散度——————————————————————————————————————————————————————————————————————————————
def JS_divergence(p,q):
    M=(p+q)/2
    return 0.5*scipy.stats.entropy(p, M, base=2)+0.5*scipy.stats.entropy(q, M, base=2)

def JS_div(arr1,arr2,num_bins):
    max0 = max(np.max(arr1),np.max(arr2))
    min0 = min(np.min(arr1),np.min(arr2))
    bins = np.linspace(min0-1e-4, max0-1e-4, num=num_bins)
    PDF1 = pd.cut(arr1,bins).value_counts() / len(arr1)
    PDF2 = pd.cut(arr2,bins).value_counts() / len(arr2)
    return JS_divergence(PDF1.values,PDF2.values)