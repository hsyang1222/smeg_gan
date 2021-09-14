import generative_model_score
inception_model_score = generative_model_score.GenerativeModelScore()
import itertools
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.autograd import Variable
import os
import wandb
import argparse
import tqdm
from dataset import *
from train import *
from models import *
from utils import *
import hashlib

def main(args):
    global inception_model_score
    
    # load real images info or generate real images info
    #torch.cuda.set_device(device=args.device)
    device = args.device
    epochs = args.epochs
    batch_size = args.batch_size
    img_size = args.img_size
    save_image_interval = args.save_image_interval
    latent_dim = args.latent_dim
    project_name = args.project_name
    dataset = args.dataset
    basemodel = args.base_model
    AE_iter = args.AE_iter
    smeg_gan = args.smeg_gan
    nz = args.nz
    ngf = args.ngf
    ndf = args.ndf
    
    lr = args.lr
    n_iter = args.n_iter
    latent_layer = args.latent_layer

    image_shape = [3, img_size, img_size]
    
    wandb_name = dataset+','+basemodel+','+str(img_size)
    if args.run_test : wandb_name += ', test run'
    if args.smeg_gan : wandb_name +=', smeg'
    if args.wandb : 
        wandb.login()
        wandb.init(project=project_name, 
                   config=args,
                   name = wandb_name)
        config = wandb.config

    ngpu = 1
    '''
    customize
    '''
    netG = Generator(ngpu, nz, ndf, ngf, nc=3).to(device)
    netD = Discriminator(ngpu, nz, ndf, ngf, nc=3).to(device)
    
    if basemodel == 'wgan' : 
        netED = EncDis(ngpu, nz, ndf, ngf, nc=3).to(device)
    if smeg_gan : 
        netE = Encoder(ngpu, nz, ndf, ngf, nc=3).to(device)
    else : 
        netE = None
    if args.dis_step : 
        netD = LinDis(nz, args.dis_layer).to(device)

    ###########################################
    #####              Score              #####
    ###########################################
    inception_model_score.lazy_mode(True)
    

    '''
    dataset 채워주세요!
    customize
    '''
    if dataset == 'CelebA':
        train_loader = get_celebA_dataset(batch_size, img_size)
    elif dataset == 'FFHQ':
        train_loader, test_loader = get_ffhq_thumbnails(batch_size, img_size)
    elif dataset == 'mnist':
        train_loader = get_mnist_dataset(batch_size, img_size)
    elif dataset == 'mnist_fashion':
        train_loader = get_mnist_fashion_dataset(batch_size, img_size)
    elif dataset == 'emnist':
        train_loader = get_emnist_dataset(batch_size, img_size)
    elif dataset == 'LSUN_dining_room':
        #wget http://dl.yf.io/lsun/scenes/dining_room_train_lmdb.zip
        #unzip dining_room_train_lmdb.zip
        #located dining_room_train_lmdb folder in dataset directory
        train_loader = get_lsun_dataset(batch_size, img_size, classes='dining_room_train')
    elif dataset == 'LSUN_classroom':
        #wget http://dl.yf.io/lsun/scenes/classroom_train_lmdb.zip
        #unzip classroom_train_lmdb.zip
        #located classroom_train_lmdb folder in dataset directory
        train_loader = get_lsun_dataset(batch_size, img_size, classes='classroom_train')
    elif dataset == 'LSUN_conference':
        #wget http://dl.yf.io/lsun/scenes/conference_room_train_lmdb.zip
        #unzip conference_room_train_lmdb.zip
        #located conference_room_train_lmdb folder in dataset directory
        train_loader = get_lsun_dataset(batch_size, img_size, classes='conference_room_train')
    elif dataset == 'LSUN_churches':
        #wget http://dl.yf.io/lsun/scenes/church_outdoor_train_lmdb.zip
        #unzip church_outdoor_train_lmdb.zip
        #located church_outdoor_train_lmdb folder in dataset directory
        train_loader = get_lsun_dataset(batch_size, img_size, classes='church_outdoor_train')
    else:
        if dataset != 'cifar10' : 
            print("dataset is forced selected to cifar10")
        train_loader = get_cifar1_dataset(batch_size, img_size)
    
    
    real_images_info_file_name = hashlib.md5(str(train_loader.dataset).encode()).hexdigest()+'.pickle'
    if args.run_test : real_images_info_file_name += '.run_test' 
    
    os.makedirs('../../inception_model_info', exist_ok=True)
    if os.path.exists('../../inception_model_info/' + real_images_info_file_name) and not args.force_gen_real_info: 
        print("Using generated real image info.")
        print(train_loader.dataset)
        inception_model_score.load_real_images_info('../../inception_model_info/' + real_images_info_file_name)
        
    else : 
        inception_model_score.model_to(device)
        
        #put real image
        for each_batch in tqdm.tqdm(train_loader, desc='insert real dataset') : 
            X_train_batch = each_batch[0]
            inception_model_score.put_real(X_train_batch)
            if args.run_test : break

        #generate real images info
        inception_model_score.lazy_forward(batch_size=64, device=device, real_forward=True)
        inception_model_score.calculate_real_image_statistics()
        #save real images info for next experiments
        inception_model_score.save_real_images_info('../../inception_model_info/' + real_images_info_file_name)
        #offload inception_model
        inception_model_score.model_to('cpu')
    

    criterion = nn.BCELoss()
    mse = nn.MSELoss()

    real_label = 1
    fake_label = 0

    # setup optimizer
    

    loss_log = {}
        
    loss_ae = 0.
    if args.ae_end_conj or AE_iter > 0 :
        
        optimizerG = torch.optim.Adam(netG.parameters(), lr=args.lr, betas=(args.beta1, 0.999), weight_decay=args.lr )
        optimizerE = torch.optim.Adam(netE.parameters(), lr=args.lr, betas=(args.beta1, 0.999), weight_decay=args.lr )
        optimizerD = torch.optim.Adam(netD.parameters(), lr=args.lr, betas=(args.beta1, 0.999), weight_decay=args.lr )
        optimizerED = torch.optim.RMSprop(netED.parameters(), lr=args.lr)
        
        
        if args.ae_end_conj == False : 
            for epoch in range(AE_iter):
                for i, data in enumerate(tqdm.tqdm((train_loader), desc="AE[%d/%d]" % (epoch, AE_iter))):
                    real_cuda = data[0].to(device)
                    batch_size = real_cuda.size(0)
                    if basemodel in ['dcgan']:  
                        loss_ae = dcgan_update_autoencoder(netE, netG, optimizerE, optimizerG, mse, real_cuda, nz)
                    elif basemodel in ['wgan'] :
                        if args.dis_step : 
                            loss_ae = wgan_disstep_update_autoencoder(netE, netD, netG, optimizerE, optimizerD, optimizerG, real_cuda)
                        else : 
                            loss_ae = wgan_encdis_update_autoencoder(netED, netG, optimizerED, optimizerG, mse, criterion, real_cuda, nz)
                    if args.run_test : break
                 
                
                if args.dis_step : 
                    wandb_wgan_disstep_ae_only_sample(wandb, args, epoch, netG, netE, train_loader, device, loss_ae)
                else : 
                    wandb_update_wgan_ae_only_sample(wandb, args, epoch, netG, netED, train_loader, device, loss_ae)
                
                if args.run_test : break
                        
                        
        else : # args.ae_end_conj == True
            epoch = 0
            loss_mean_last = 9999.0
            while epoch < 1000 :
                loss_list = []
                for i, data in enumerate(tqdm.tqdm((train_loader), desc="AE[%d]" % epoch)):
                    real_cuda = data[0].to(device)
                    batch_size = real_cuda.size(0)
                    if basemodel in ['dcgan', 'wgan'] :  
                        loss_ae = dcgan_update_autoencoder(netE, netG, optimizerE, optimizerG, mse, real_cuda, nz)
                        loss_list.append(loss_ae)
                    
                    loss_mean = sum(loss_list)/len(loss_list) 
                #print("loss sum and len", sum(loss_list), len(loss_list))
                wandb_update_only_sample(wandb, args, epoch, netG, netE, train_loader,loss_mean_last-loss_mean, device)
                if loss_mean_last-loss_mean  <= args.ae_end_diffloss : 
                    print("success autoencoder (loss_diff=%f-%f=%f<=%f)" % \
                          (loss_mean_last, loss_mean, loss_mean_last-loss_mean, args.ae_end_diffloss))
                    break
                else : 
                    print("learn autoencoder (loss_diff=%f-%f=%f>%f)" % \
                          (loss_mean_last, loss_mean, loss_mean_last-loss_mean, args.ae_end_diffloss))
                loss_mean_last = loss_mean
                epoch +=1

                
            
    optimizerD = torch.optim.Adam(netD.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    optimizerG = torch.optim.Adam(netG.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    if netE is not None :
        optimizerE = torch.optim.Adam(netE.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

    if basemodel == 'wgan' : 
        optimizerD = torch.optim.RMSprop(netD.parameters(), lr=args.lr)
        optimizerG = torch.optim.RMSprop(netG.parameters(), lr=args.lr)
        if netE is not None :
            optimizerE = torch.optim.RMSprop(netE.parameters(), lr=args.lr)
            
            
            
            
            
            
    mul_alpha = torch.tensor([-9.0], requires_grad=True, device=device)
    optimizerM = torch.optim.SGD([mul_alpha], lr=0.001) 
    loss_alpha = (-1.,-1.)
    lossE = 0.
    if not smeg_gan : 
        sma = 1.

    if netE is not None : 
        real_cuda = next(iter(train_loader))[0].to(device)
        with torch.no_grad() :
            real_latent_4dim = netE(real_cuda).view(real_cuda.size(0), -1)
        fixed_noise = real_latent_4dim.detach().cpu()
    else : 
        fixed_noise = torch.rand(batch_size, nz)
    

    
    
    

    for i in range(0, epochs):
        batch_count = 0

        lossD_list = []
        lossG_list = []
        
        for image, label in tqdm.tqdm(train_loader, desc='Train[%d/%d]' %(i, epochs)):
            batch_count += 1
            real_cuda = image.to(device)
            
            if basemodel == 'dcgan' :  
                if smeg_gan : 
                    sma = torch.sigmoid(mul_alpha)
                    lossD = dcgan_smeg_update_discriminator(netD, netG, netE, optimizerD,\
                                                real_cuda, criterion, nz, sigmoid_mul_alpha=sma)
                    lossG = dcgan_smeg_update_generator(netD, netG, netE, optimizerG, real_cuda, criterion, nz, sma)
                    sma = torch.sigmoid(mul_alpha)
                    loss_alpha = dcgan_smeg_update_alpha(netD, netG, netE, optimizerM, real_cuda, criterion, nz, sma)
                    lossE = dcgna_smeg_update_encoder(netE, netG, real_cuda, optimizerE, mse, nz)
                else : 
                    lossD = dcgan_update_discriminator(netD, netG, netE, optimizerD,\
                                                    real_cuda, criterion, nz)
                    lossG = dcgan_update_generator(netD, netG, netE, optimizerG, real_cuda, criterion, nz)
            elif basemodel == 'wgan' : 
                if smeg_gan : 
                    if args.dis_step : 
                        sma = sma_gen(mul_alpha, set_min=args.add_z_min)
                        loss_dis = wgan_smeg_disstep_update_discriminator(netE, netG, netD, optimizerD, \
                                                                  real_cuda, nz, sma, args.clip_value)
                        if i % args.n_critic == 0 : 
                            loss_g = wgan_smeg_disstep_update_generator(netE, netG, netD, optimizerG, real_cuda, nz, sma)
                        sma = sma_gen(mul_alpha, set_min=args.add_z_min)
                        loss_alpha = wcgan_smeg_disstep_update_alpha(netE, netD, netG, optimizerM, real_cuda, nz, sma)
                    else : 
                        sma = sma_gen(mul_alpha, set_min=args.add_z_min)
                        loss_dis = wgan_smeg_update_discriminator(netED, netG, optimizerED, \
                                                                  real_cuda, criterion, nz, sma, args.clip_value)
                        if i % args.n_critic == 0 : 
                            loss_g = wgan_smeg_update_generator(netED, netG, optimizerG, real_cuda, criterion, nz, sma)
                        sma = sma_gen(mul_alpha, set_min=args.add_z_min)
                        loss_alpha = wcgan_smeg_update_alpha(netED, netG, optimizerM, real_cuda, criterion, nz, sma)
                else : 
                    lossD = wgan_update_discriminator(netD, netG, netE, optimizerD,\
                                                    real_cuda, criterion, nz, args.clip_value)
                    if i % args.n_critic == 0 : 
                        lossG = wgan_update_generator(netD, netG, netE, optimizerG, real_cuda, criterion, nz)


            '''
            elif basemodel == 'lsgan' :
                lsgan_smeg_update_discriminator()
                lsgan_smeg_update_generator()

            '''

            if args.run_test : break
                
        loss = {}
        loss.update(loss_dis)
        loss.update(loss_g)
        loss.update(loss_alpha)
        wandb_wgan_update(wandb, args, i, inception_model_score, netE, netG, train_loader, nz, device, sma, loss)

                

            
         
        
    #save_losses(epochs, loss_calculation_interval, r_losses, d_losses, g_losses)
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    #python main.py --device=cuda:0 --dataset=cifar10 --base_model=dcgan --smeg_gan=True

    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--img_size', type=int, default=32)
    parser.add_argument('--save_image_interval', type=int, default=5)
    parser.add_argument('--nz', type=int, default=100)
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--ndf', type=int, default=64)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--latent_layer', type=int, default=3)
    parser.add_argument('--latent_dim', type=int, default=10)
    parser.add_argument('--n_iter', type=int, default=3)
    parser.add_argument('--project_name', type=str, default='SMEG GAN')
    parser.add_argument('--dataset', type=str, default='', 
                        choices=['LSUN_dining_room', 'LSUN_classroom', 'LSUN_conference', 'LSUN_churches',
                                            'FFHQ', 'CelebA', 'cifar10', 'mnist', 'mnist_fashion', 'emnist'])
    parser.add_argument('--base_model', type=str, default='dcgan', choices=['dcgan', 'lsgan', 'wgan' ])
    parser.add_argument('--n_critic', type=int, default=5)
    parser.add_argument('--clip_value', type=float, default=0.01)
    parser.add_argument('--AE_iter', type=int, default=0)
    parser.add_argument('--smeg_gan', type=bool, default=False)
    parser.add_argument('--dis_step', type=bool, default=False)
    parser.add_argument('--dis_layer', type=int, default=8)

    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--run_test', type=bool, default=False)
    parser.add_argument('--wandb', type=bool, default=False)
    parser.add_argument('--sample_img_num', type=int, default=16)
    parser.add_argument('--force_gen_real_info', type=bool, default=False)
    parser.add_argument('--add_z_min', type=float, default=1e-2)
    parser.add_argument('--ae_end_conj', type=bool, default=False)
    parser.add_argument('--ae_end_diffloss', type=float, default=1e-4)
    

    args = parser.parse_args()

    main(args)
