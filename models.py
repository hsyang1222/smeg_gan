import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, ngpu, nz=100, ndf=64, ngf=64, nc=3, img_size=32):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        
        output_kernel_size = ngf*4*(2*(img_size//32))
        #print(ngf, img_size//32, 2*(img_size//32), output_kernel_size)
        
        kernel_up = nn.Sequential(
            nn.ConvTranspose2d( nz, output_kernel_size, 2, 1, 0, bias=False),
            nn.BatchNorm2d(output_kernel_size),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        up_size = []
        cur_size = 2
        input_kernel_size = output_kernel_size
        output_kernel_size = output_kernel_size//2
        
        while cur_size < img_size : 
            cur_size *= 2
            if cur_size == img_size : 
                up_size.append(torch.nn.ConvTranspose2d(input_kernel_size, nc, 4, 2, 1))
            else : 
                up_size.append(torch.nn.ConvTranspose2d(input_kernel_size, output_kernel_size, 4, 2, 1))
                up_size.append(nn.BatchNorm2d(output_kernel_size))
                up_size.append(nn.LeakyReLU(0.2, inplace=True))
            input_kernel_size = output_kernel_size
            output_kernel_size = output_kernel_size//2
            
        up_size.insert(0, kernel_up)
            
        self.main = torch.nn.ModuleList(up_size)

    def forward(self, x):
        for layer in self.main:
            x = layer(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, ngpu, nz=100, ndf=64, ngf=64, nc=3, img_size=32):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu

        output_kernel_size = ndf
        
        kernel_up = nn.Sequential(
            nn.Conv2d( nc, output_kernel_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(output_kernel_size),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        down_size = []
        cur_size = img_size // 2
        input_kernel_size = output_kernel_size
        output_kernel_size = output_kernel_size*2
        
        while cur_size > 1 : 
            cur_size = cur_size // 2
            if cur_size == 1 : 
                down_size.append(torch.nn.Conv2d(input_kernel_size, 1, 2, 1, 0))
                down_size.append(nn.Sigmoid())
            else : 
                down_size.append(torch.nn.Conv2d(input_kernel_size, output_kernel_size, 4, 2, 1))
                down_size.append(nn.BatchNorm2d(output_kernel_size))
                down_size.append(nn.LeakyReLU(0.2, inplace=True))
            input_kernel_size = output_kernel_size
            output_kernel_size = output_kernel_size*2
            
        down_size.insert(0, kernel_up)
            
        self.main = torch.nn.ModuleList(down_size)

    def forward(self, x):
        for layer in self.main:
            x = layer(x)
        return x.view(-1, 1)

class Encoder(nn.Module):
    
    def __init__(self, ngpu, nz=100,ndf=64, ngf=64, nc=3, img_size=32):
        super(Encoder, self).__init__()
        self.ngpu = ngpu
        
        output_kernel_size = ndf
        
        kernel_up = nn.Sequential(
            nn.Conv2d( nc, output_kernel_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(output_kernel_size),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        down_size = []
        cur_size = img_size // 2
        input_kernel_size = output_kernel_size
        output_kernel_size = output_kernel_size*2
        
        while cur_size > 1 : 
            cur_size = cur_size // 2
            if cur_size == 1 : 
                down_size.append(torch.nn.Conv2d(input_kernel_size, nz, 2, 1, 0))
            else : 
                down_size.append(torch.nn.Conv2d(input_kernel_size, output_kernel_size, 4, 2, 1))
                down_size.append(nn.BatchNorm2d(output_kernel_size))
                down_size.append(nn.LeakyReLU(0.2, inplace=True))
            input_kernel_size = output_kernel_size
            output_kernel_size = output_kernel_size*2
            
        down_size.insert(0, kernel_up)
            
        self.main = torch.nn.ModuleList(down_size)
        self.nz = nz

    def forward(self, x):
        for layer in self.main:
            x = layer(x)
        return x.view(-1, self.nz, 1,1)

    
class LinDis(nn.Module):
    def __init__(self, nz, lin_num) : 
        super(LinDis, self).__init__()
        assert lin_num>=1, "linear layer must be larger than 1"
        linear_list = []
        for i in range(lin_num-1) : 
            linear_list.append(nn.Linear(nz, nz))
            linear_list.append(nn.BatchNorm1d(nz))
            linear_list.append(nn.LeakyReLU(0.2, inplace=True))
        linear_list.append(nn.Linear(nz,1))
        self.linear = torch.nn.ModuleList(linear_list)
    
    def forward(self, x) :
        for layer in self.linear : 
            x = layer(x)
        return x


        
    
class EncDis(nn.Module):
    def __init__(self, ngpu, nz=100, ndf=64, ngf=64, nc=3, ):
        super(EncDis, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            #nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Conv2d(ndf * 8, nz, 2, 1, 0, bias=False),
            nn.Tanh()
            # state size. 1x1x1
        )
        self.linear = nn.Sequential(
            nn.Linear(nz, nz),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(nz, nz),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(nz, nz),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(nz, nz),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(nz, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(nz, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(nz, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(nz, 1),
        )
        self.nz = nz

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu)).view(-1, self.nz)
            dis = nn.parallel.data_parallel(self.linear, output  , range(self.ngpu))
        else:
            output = self.main(input).view(-1,self.nz)
            dis = self.linear(output)

        return dis, output.view(-1,self.nz,1,1)