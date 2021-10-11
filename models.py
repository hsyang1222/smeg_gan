import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, ngpu, nz=100, ndf=64, ngf=64, nc=3, img_size=32):
        super(Generator, self).__init__()
        self.ngpu = ngpu
                
        conv = nn.Sequential(
            # input is Z, going into a convolution
            #nn.ConvTranspose2d(     nz, ngf * 8, 4, 1, 0, bias=False),
            nn.ConvTranspose2d(     nz, ngf * 8, 2, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 2 x 2
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 4 x 4
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 8 x 8
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 16 x 16
            nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
            # state size. (nc) x 32 x 32
        )
        
        up_size = []
        up_size.append(conv)
        cur_size = 32
        while cur_size < img_size : 
            up_size.append(nn.BatchNorm2d(3))
            up_size.append(torch.nn.ConvTranspose2d(3, 3, 2, 2))
            if cur_size*2 != img_size : up_size.append(nn.ReLU(True))
            cur_size *= 2
        self.main = torch.nn.ModuleList(up_size)

    def forward(self, x):
        for layer in self.main:
            x = layer(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, ngpu, nz=100, ndf=64, ngf=64, nc=3, img_size=32):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        
        conv = nn.Sequential(
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
            nn.Conv2d(ndf * 8, 1, 2, 1, 0, bias=False),
            # state size. 1x1x1
            nn.Sigmoid()
        )
        
        down_size = []
        cur_size = 32
        while cur_size < img_size : 
            down_size.append(torch.nn.Conv2d(3, 3, 2, 2))
            down_size.append(nn.BatchNorm2d(3))
            down_size.append(nn.ReLU(True))
            cur_size *= 2
        
        down_size.append(conv)
            
        self.main = torch.nn.ModuleList(down_size)


    def forward(self, x):
        for layer in self.main:
            x = layer(x)
        return x.view(-1, 1)

class Encoder(nn.Module):
    
    def __init__(self, ngpu, nz=100,ndf=64, ngf=64, nc=3, img_size=32):
        super(Encoder, self).__init__()
        self.ngpu = ngpu
        
        conv = nn.Sequential(
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
        )
        
        down_size = []
        cur_size = 32
        while cur_size < img_size : 
            down_size.append(torch.nn.Conv2d(3, 3, 2, 2))
            down_size.append(nn.BatchNorm2d(3))
            down_size.append(nn.ReLU(True))
            cur_size *= 2
        
        down_size.append(conv)
            
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