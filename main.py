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
    if args.inf_gs : wandb_name += ', inf_gs'
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
        netD = Discriminator_notsig(ngpu, nz, ndf, ngf, nc=3).to(device)

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

        
        
        if args.ae_end_conj == False : 
            for epoch in range(1, AE_iter+1):
                for i, data in enumerate(tqdm.tqdm((train_loader), desc="AE[%d/%d]" % (epoch, AE_iter))):
                    real_cuda = data[0].to(device)
                    batch_size = real_cuda.size(0)                    
                    loss_ae = dcgan_update_autoencoder(netE, netG, optimizerE, optimizerG, mse, real_cuda, nz)
                    if args.run_test : break
                if args.run_test : break
                
            
    optimizerD = torch.optim.Adam(netD.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    optimizerG = torch.optim.Adam(netG.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    if netE is not None :
        optimizerE = torch.optim.Adam(netE.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

    if basemodel == 'wgan' : 
        optimizerD = torch.optim.RMSprop(netD.parameters(), lr=args.lr)
        optimizerG = torch.optim.RMSprop(netG.parameters(), lr=args.lr)
        if netE is not None :
            optimizerE = torch.optim.RMSprop(netE.parameters(), lr=args.lr)
            
    mul_alpha = torch.tensor([0.1], device=device)        

    if netE is not None : 
        real_cuda = next(iter(train_loader))[0].to(device)
        with torch.no_grad() :
            real_latent_4dim = netE(real_cuda).view(real_cuda.size(0), -1)
        fixed_noise = real_latent_4dim.detach().cpu()
    else : 
        fixed_noise = torch.rand(batch_size, nz)
    

    
    loss_e = {}
    loss_dis={}
    loss_g={}
    loss_alpha={}
    
    output_repaint_list = []
    output_mixed_list = []
    output_gsfake_list = []
    output_real_list = []
    smeg = args.smeg_gan
    
    i=1
    while i <= epochs:
        loss_log = {}
        sma = mul_alpha
        
        if smeg : 
            for image, label in tqdm.tqdm(train_loader, desc='Train(SMEG)[%d/%d]' %(i, epochs)):
                real_cuda = image.to(device)
                loss_d = wgan_smeg_v2_update_discriminator(netE, netG, netD, optimizerD, real_cuda, nz, sma, args.clip_value)
                output_real_list.append(loss_d['up d - output_real'].detach())
                output_gsfake_list.append(loss_d['up d - output_gaussian_fake'].detach())
                if i % args.n_critic == 0 : 
                    loss_g = wgan_smeg_v2_update_generator(netE, netG, netD, optimizerG, real_cuda, nz, sma, hyper_img_diff=1e-4)
                    loss_e = wgan_smeg_v2_update_encoder(netE, netG, netD, optimizerE, real_cuda)
                    output_repaint_list.append(loss_g['up g - output_repaint'].detach())
                    output_mixed_list.append(loss_g['up g - output_mixed'].detach())
                loss_log
                if args.run_test : break
            
            output_real = torch.stack(output_real_list).mean()
            output_repaint = torch.stack(output_repaint_list).mean()
            output_mixed = torch.stack(output_mixed_list).mean()
            output_gsfake = torch.stack(output_gsfake_list).mean()

            loss_m = wcgan_smeg_v2_update_alpha(output_real, output_repaint, output_mixed, output_gsfake, \
                                                sma, add_alpha=1e-1, per_close=args.per_close)
            smeg = not loss_m['up alpha - end_smeg']
        else : 
            for image, label in tqdm.tqdm(train_loader, desc='Train[%d/%d]' %(i, epochs)):
                real_cuda = image.to(device)
                loss_d = wgan_update_discriminator(netD, netG, netE, optimizerD, real_cuda, criterion, nz, clip_value=0.01)
                if i % args.n_critic == 0 : 
                    loss_g = wgan_update_generator(netD, netG, netE, optimizerG, real_cuda, criterion, nz)
                if args.run_test : break
            
        loss_log.update(loss_d)
        loss_log.update(loss_g)
        loss_log.update(loss_m)
        loss_log.update(loss_e)
        wandb_wgan_update(wandb, args, i, inception_model_score, netE, netG, netD, train_loader, nz, device, sma, loss_log)

        i+=1

        
               
            
         
        
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
    parser.add_argument('--project_name', type=str, default='smeg_gan_v2')
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
    parser.add_argument('--train_e', type=bool, default=False)
    parser.add_argument('--use_plain_alpha', type=bool, default=False)
    parser.add_argument('--alpha_conti', type=bool, default=False)
    parser.add_argument('--per_close', type=float, default=1e-2)
    parser.add_argument('--inf_gs', type=bool, default=True)

    args = parser.parse_args()

    main(args)
