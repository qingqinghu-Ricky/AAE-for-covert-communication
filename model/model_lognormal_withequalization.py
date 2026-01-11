# 设定信道的SNR
# 加入信道均衡
import torch
import torch.nn as nn
import numpy as np



# Autoencoder网络 ————————————————————————————————————————————————————————————————————————————————————————
class Autoencoder(nn.Module):
    def __init__(self, in_channel, n_channel, channel_type, r):
        super(Autoencoder, self).__init__()
        self.r = r
        # self.ebno = ebno
        self.channel_type = channel_type
        self.n_channel = n_channel
        self.count = 0
        # self.sigma_AN = nn.Parameter(torch.tensor([-5.0], requires_grad=True)) # 成为了模型中根据训练可以改动的参数 # -10.0
        # self.h_est = nn.Parameter(torch.tensor([0.0], requires_grad=True))
        # hiddle_channel = 2*n_channel
        
        self.encoder = nn.Sequential(     # 每层都多加了batchnorm层
            nn.Linear(in_channel + n_channel,  128),
            # nn.Linear(in_channel + n_channel,  4 * n_channel),
            nn.LayerNorm(128, elementwise_affine=True),
            # nn.ReLU(True),
            nn.LeakyReLU(0.2),

            nn.Linear(128, 128),
            nn.LayerNorm(128, elementwise_affine=True),
            # nn.ReLU(True),
            nn.LeakyReLU(0.2),

            nn.Linear(128, 128),
            nn.LayerNorm(128, elementwise_affine=True),
            # nn.ReLU(True),
            nn.LeakyReLU(0.2),

            nn.Linear(128, 128),
            nn.LayerNorm(128, elementwise_affine=True),
            # nn.ReLU(True),
            nn.LeakyReLU(0.2),
            
            

            nn.Linear(128, n_channel),
            # nn.LayerNorm(n_channel, elementwise_affine=False), # 输出码字功率归一化
        )


        self.decoder = nn.Sequential(         # 每层都多加了batchnorm层
            # nn.LayerNorm(n_channel, elementwise_affine=True),
            nn.Linear(n_channel, 128), # revise
            nn.LayerNorm(128, elementwise_affine=True),
            # nn.ReLU(True),
            nn.LeakyReLU(0.2),
        
            nn.Linear(128, 128), # revise
            nn.LayerNorm(128, elementwise_affine=True),
            # nn.ReLU(True),
            nn.LeakyReLU(0.2),

            nn.Linear(128, 128),
            nn.LayerNorm(128, elementwise_affine=True),
            # nn.ReLU(True),
            nn.LeakyReLU(0.2),

            nn.Linear(128, 128),
            nn.LayerNorm(128, elementwise_affine=True),
            # nn.ReLU(True),
            nn.LeakyReLU(0.2),

            nn.Linear(128, in_channel),
        )
        
        
        self.channel_est = nn.Sequential(
                nn.Linear(n_channel, n_channel* 2),
                nn.ELU(),
                nn.Linear(n_channel * 2, n_channel * 4),
                nn.Tanh(),
                nn.Linear(n_channel * 4, int(n_channel / 2)),
                nn.Tanh(),
                nn.Linear(int(n_channel / 2), 1)  # 输出信道估计参数
            )

        self.init_weight()
        
    def init_weight(self,):
        for m in self.modules():  # 不同层权重初始化不同
            if isinstance(m, nn.Conv1d):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.zero_()                
            elif isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.zero_()
            # elif isinstance(m, nn.BatchNorm1d):
            elif isinstance(m, nn.LayerNorm):
                if m.elementwise_affine == True:
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

    def forward(self, x, t, snr):   # 实例化Autoencoder后默认调用的函数
        x = self.encoder(torch.cat([x, t], dim=1)) 
        device = x.device
        
        # x1 = x- x.mean(dim=1).unsqueeze(-1)  # 每个样本经历的信道衰落不同，因此隔直是在每个样本（包含n个特征）上进行的
        # x2 = x1/(x1.std(dim=1).unsqueeze(-1)+1e-18) # 将每个样本信号进行尺度归一化
        # x = x2
        
        x1 = x- x.mean()  # 每个样本经历的信道衰落不同，因此隔直是在每个样本（包含n个特征）上进行的
        x2 = x1/(x1.std()+1e-18) # 将每个样本信号进行尺度归一化
        x = x2
        
        noise_data = torch.randn(x.size(), requires_grad=False).to(device)
        
        
        # 信号经过信道————————————————————————————————————————————————————————       
        d1 = self.lognormal(x,snr)

     
        # 噪声经过信道————————————————————————————————————————————————————————
        AN1 = self.lognormal(noise_data,snr)
        
        
        # 信道均衡————————————————————————————————————————————————————————————
        h_est = self.channel_est(d1) # 网络估计的h
        d2 = d1 / h_est
        
        x = self.decoder(d2)  # 解码
        
        return x, d1, AN1 # Bob解调后信号，发covert信号时经过channel后，不发covert信号时经过channel后
    
    def lognormal(self, x, snr, h=None): # h是可选的参数

        device = x.device

        
        # LED输出光功率非负，需加bias ——————————————————————————————————————————————————————
        bias = torch.tensor([10], requires_grad=False)
        x = x + bias 
        

        x = torch.clamp(x, min=0) # 将x中小于0的数值限制为0，防止后续开根号出现NaN

        # APD 参数 ——————————————————————————————————————————————————————————————————————
        S = torch.tensor([1.1538], requires_grad=False)
        M = torch.tensor([100], requires_grad=False)
        eta = torch.tensor([1], requires_grad=False)
        q = torch.tensor([1.602e-19], requires_grad=False)
        I0 = torch.tensor([1e-9], requires_grad=False)
        B = torch.tensor([500e6], requires_grad=False)
        k = torch.tensor([1.380649e-23], requires_grad=False)
        T = torch.tensor([300], requires_grad=False)
        Rs = torch.tensor([50], requires_grad=False)
        m = torch.tensor([0.3], requires_grad=False)


        # lognormal fading ———————————————————————————————————————————————————————————————
        if h is None:
            # Bob (0,0,1)处的lognormal分布参数 (u=-13.1548, sigma=0.330682), Willie (0.1,0.1,5) 参数（u=-15.5045, sigma=0.187588）
            u_b = torch.tensor([-13.1548], requires_grad=False)
            sigma_b = torch.tensor([0.330682], requires_grad=False)
            pdf_h = torch.distributions.log_normal.LogNormal(u_b, sigma_b)
            fading_batch = pdf_h.sample((x.size()[0],)).requires_grad_(False).to(device) #h是样本数*1维的，在n个channel上相同, sample参数sample_shape，输出是（sample_shape，1）
            # (x.size()[0],) 必须加逗号形成元组，成为可迭代对象
            # expectation_h2 =  torch.exp(2 * u_b + 2 * (sigma_b ** 2)) # E(h^2)
        else:
            fading_batch = h
        
        self.fading = fading_batch # 1e-6量级
       

        fading_batch = fading_batch * 1e3  # 使hx能落在APD输入范围内 0 到1.5e-2 W
        hx = x * fading_batch # 到达APD前的光功率 , 会broadcast扩充维度， 1e-5量级
        
        
        # APD非线性 ———————————————————————————————————————————————————————————————
        u = S/(2*M)*((1+4*(M**2)*eta/S*(hx)*1e3).sqrt()-1)*1e-3+I0
        u = u.to(device)
   
        # APD noise 在弱背景光下较小 ———————————————————————————————————————————————————————————————
        # sigma_2=2*q*I0*B+4*k*T*B/Rs+ \
        # 2*q*B*(S/(2*M)*1e-3).pow(m+2)* \
        # ((1+4*(M**2)*eta/S*(hx)*1e3).sqrt()-1).pow(m+2)*(eta*hx).pow(-m-1) # \ 实现换行

        u1 = u/(8e-4) # 将样本信号进行放大

     

        # 额外给定 无衰落SNR 的热噪声————————————————————————————————————————————————————————————
        sigma_2_add= 1/(10**(snr/10))  # 另外加上无衰落信噪比对应的噪声，热噪声与信号，衰落无关

        sigma = (sigma_2_add).sqrt()  # 1e-7量级

        noise = torch.randn(x.size(), requires_grad=False) * sigma # 作用在形参上，代表这个位置接收任意多个非关键字参数 ; * 是矩阵对应元素相乘
        noise = noise.to(device)

        y = u1 + noise
        
        # aa = hx/(5.1428e-05**(0.5))
        # y =  hx/(5.1428e-05**(0.5)) + noise

        
        y1 = y - y.mean(dim=1).unsqueeze(-1)  # 每个样本经历的信道衰落不同，因此隔直是在每个样本（包含n个特征）上进行的
        y2 = y1
        # y2 = y1/(y1.std(dim=1).unsqueeze(-1)+1e-18) # 将每个样本信号进行尺度归一化

        # import pdb; pdb.set_trace()
        
        self.count+=1
        if self.count==1:
            snr_output = 10*(u1.var()/noise.var()).log10()
            print('------snr------: {} dB'.format(snr_output))
    


        if np.isnan(torch.mean(y2).detach().numpy()) == True:   # 判断输出是否有NaN    
            import pdb; pdb.set_trace()
            
        return (y2)  

        # return (x) # 相当于不经过信道
        # return (x*torch.tensor(1e-6, requires_grad=False) + noise) # 相当于经过线性信道

        #     snr = 10*(x.var()/noise.var()).log10()
        #     print('------snr------: {} dB'.format(snr))            
        #     snr = 10*((x*1e-3).var()/noise.var()).log10()
        #     print('------snr------: {} dB'.format(snr))     
        # return (x*1e-3)*torch.exp(self.h_est) # 相当于经过线性信道
        # x = x*1
        # return x

    

# Willie 网络 ——————————————————————————————————————————————————————————————————————————————————————————————                

class Willie(nn.Module):
    def __init__(self, n_channel):
        super(Willie, self).__init__()
        self.net = nn.Sequential(
            # nn.LayerNorm(n_channel, elementwise_affine=False),
            # nn.BatchNorm1d(n_channel, affine=True),
            nn.Linear(n_channel, 128), # revise
            # nn.LayerNorm(128, elementwise_affine=True),
            # nn.ReLU(True),
            nn.LeakyReLU(0.2),

            nn.Linear(128, 128), # revise
            # nn.LayerNorm(128, elementwise_affine=True),
            # nn.ReLU(True),
            nn.LeakyReLU(0.2),
            
            nn.Linear(128, 128),
            # nn.LayerNorm(128, elementwise_affine=True),
            # nn.ReLU(True),
            nn.LeakyReLU(0.2),
            
            nn.Linear(128, 128), # revise
            # nn.LayerNorm(128, elementwise_affine=True),
            # nn.ReLU(True),
            nn.LeakyReLU(0.2),
            

            nn.Linear(128, 1),
            # nn.Sigmoid()
        )
        self.init_weight()
        
    def init_weight(self,):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.zero_()                
            elif isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.zero_()
            # elif isinstance(m, nn.BatchNorm1d):
            elif isinstance(m, nn.LayerNorm):
                if m.elementwise_affine == True:
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

    def forward(self, x, h=None):
        return self.net(x)
    
# 计算WGAN-GP的函数————————————————————————————————————————————————————————————————————————————————————————————————————
    
def gradient_penalty(real_data, fake_data, D, device):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.rand(len(real_data), 1).to(device)
    alpha = alpha.expand_as(real_data)
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_data + ((1 - alpha) * fake_data)).requires_grad_(True)
    d_interpolates = D(interpolates)

    # Calculate gradients of D(interpolates) wrt interpolates
    gradients = torch.autograd.grad(outputs=d_interpolates, inputs=interpolates,
                                    grad_outputs=torch.ones_like(d_interpolates),
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(len(real_data), -1)
    # Calculate L2 norm
    gradient_norm = gradients.norm(2, dim=1)
    # Calculate gradient penalty
    gradient_penalty = ((gradient_norm - 1) ** 2).mean()
    return gradient_penalty