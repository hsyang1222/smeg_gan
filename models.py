import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, ngpu, nz=100, ndf=64, ngf=64, nc=3, img_size=32):
        super(Generator, self).__init__()

        self.init_size = img_size // 4
        self.l1 = nn.Sequential(nn.Linear(nz, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, nc, 3, stride=1, padding=1),
            nn.Tanh(),
        )
        self.nz = nz

    def forward(self, z):
        out = self.l1(z.view(-1,self.nz))
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

class Discriminator(nn.Module):
    def __init__(self, ngpu, nz=100, ndf=64, ngf=64, nc=3, img_size=32):
        super(Discriminator, self).__init__()
        
        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(nc, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = img_size // 2 ** 4
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity

class Encoder(nn.Module):
    
    def __init__(self, ngpu, nz=100,ndf=64, ngf=64, nc=3, img_size=32):
        super(Encoder, self).__init__()
        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(nc, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = img_size // 2 ** 4
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, nz))
        self.nz = nz

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity.view(-1,self.nz,1,1)

    
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