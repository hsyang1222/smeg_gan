from matplotlib import pyplot as plt
import torch
import os
from torchvision.utils import save_image
from torch.autograd import Variable
import numpy as np
from PIL import Image
import tqdm
from train import *

def sample_fake(netG, netE, train_loader, nz, device, sigmoid_mul_alpha=1.) : 
    fake_image_list = []
    with torch.no_grad() : 
        for image, label in tqdm.tqdm(train_loader, desc='insert fake') :
            batch_size = image.size(0)
            real_cuda = image.to(device)
            insert_feature = gen_latent_feature(netE, real_cuda, batch_size, nz, sigmoid_mul_alpha)
            fake_image = netG(insert_feature)
            fake_image_list.append(fake_image.detach().cpu())
    fake_image_all = torch.cat(fake_image_list)
    return fake_image_all

def sample_fake_just(netG, netE, train_loader, nz, device, just, sigmoid_mul_alpha=1.) : 
    fake_image_list = []
    with torch.no_grad() : 
        for image, label in train_loader:
            batch_size = image.size(0)
            real_cuda = image.to(device)
            insert_feature = gen_latent_feature(netE, real_cuda, batch_size, nz, sigmoid_mul_alpha)
            fake_image = netG(insert_feature)
            fake_image_list.append(fake_image.detach().cpu())
            
            if train_loader.batch_size * len(fake_image_list) >= just : break
                
    fake_image_all = torch.cat(fake_image_list)
    return fake_image_all

def sample_repaint_just(netG, netE, train_loader, nz, device, just) : 
    fake_image_list = []
    with torch.no_grad() : 
        for image, label in train_loader:
            batch_size = image.size(0)
            real_cuda = image.to(device)
            insert_feature = netE(real_cuda)
            fake_image = netG(insert_feature)
            fake_image_list.append(fake_image.detach().cpu())
            
            if train_loader.batch_size * len(fake_image_list) >= just : break
                
    fake_image_all = torch.cat(fake_image_list)
    return fake_image_all

import torchvision.utils as vutils
def make_grid_img(fake_image, nrow=8) : 
    fake_image = fake_image * 0.5 + 0.5
    fake_np = vutils.make_grid(fake_image, nrow).permute(1,2,0).numpy()
    return fake_np 

def train_model_to(model_list, to) : 
    for model in model_list : 
        if model is not None : 
            model.to(to)

def gen_matric(inception_model_score, device):
    inception_model_score.model_to(device)
    inception_model_score.lazy_forward(fake_forward=True, device=device)
    inception_model_score.calculate_fake_image_statistics()
    matric = inception_model_score.calculate_generative_score()
    inception_model_score.model_to('cpu')
    return matric
    
def save_losses(epochs, save_calculation_interval, r_losses, d_losses, g_losses):
    X = range(1, epochs + 1, save_calculation_interval)
    fig = plt.figure(figsize=(30, 30))
    plt.subplot(3, 1, 1)
    plt.title("r_losses")
    plt.plot(X, r_losses, color="blue", linestyle="-", label="r_losses")
    plt.subplot(3, 1, 2)
    plt.title("g_losses")
    plt.plot(X, g_losses, color="purple", linestyle="-", label="g_losses")
    plt.subplot(3, 1, 3)
    plt.title("d_losses")
    plt.plot(X, d_losses, color="red", linestyle="-", label="d_losses")
    plt.savefig('aae_celebA/losses.png')
    plt.close()

def sma_gen(mul_alpha, set_min=0.1) : 
    sma = torch.sigmoid(mul_alpha) * (1-set_min) + set_min
    return sma
    
def save_scores_and_print(current_epoch, epochs, r_loss, d_loss, g_loss, precision, recall, fid, inception_score_real,
                          inception_score_fake, dataset, model_name):
    folder_name = 'logs/%s_%s' % (dataset, model_name)
    os.makedirs(folder_name, exist_ok=True)
    f = open("./%s/generative_scores.txt" % folder_name, "a")
    f.write("%d %f %f %f %f %f %f %f %f\n" % (current_epoch, r_loss, d_loss, g_loss, precision, recall, fid, inception_score_real, inception_score_fake))
    f.close()
    print("[Epoch %d/%d] [R loss: %f] [D loss: %f] [G loss: %f] [precision: %f] [recall: %f] [fid: %f] [inception_score_real: %f] [inception_score_fake: %f]"
          % (current_epoch, epochs, r_loss, d_loss, g_loss, precision, recall, fid, inception_score_real, inception_score_fake))


def save_images(n_row, epoch, latent_dim, model, dataset, model_name, device):
    folder_name = '%s_%s' % (dataset, model_name)
    os.makedirs('images/%s' % folder_name, exist_ok=True)
    """Saves a grid of generated digits"""
    # Sample noise
    z = torch.tensor(  np.random.normal(0, 1,(n_row ** 2, latent_dim) )).float().to(device)
    gen_imgs = model(z)
    image_name = "images/%s/%d_epoch.png" % (folder_name, epoch)
    save_image(gen_imgs.data, image_name, nrow=n_row, normalize=True)
    return Image.open(image_name)


def wandb_update_only_sample(wandb, args, i, netG, netE, train_loader, lossdiff, device):
    if args.wandb : 
        with torch.no_grad() : 
            images = next(iter(train_loader))[0]
            real_cuda = images.to(device)
            fake_sample = netG(netE(real_cuda)).cpu()
        matric = {}
        save_sample_img = fake_sample[:args.sample_img_num ** 2]
        grid_sample_img = make_grid_img(save_sample_img, nrow=args.sample_img_num)
        matric.update({'ae_step':i,
                  'AE_sample' : [wandb.Image(grid_sample_img, caption=str(i))],
                  'AE_lossdiff' : lossdiff,
                  })
        wandb.log(matric)   
        
        
def wandb_wgan_disstep_ae_only_sample(wandb, args, i, netG, netE, train_loader, device, loss_log) :
    if args.wandb :
        with torch.no_grad() :
            images = next(iter(train_loader))[0]
            real_cuda = images.to(device)
            repaint = netG(netE(real_cuda))
            fake_sample = repaint.cpu()
            matric = {}
            save_sample_img = fake_sample[:args.sample_img_num ** 2]
            grid_sample_img = make_grid_img(save_sample_img, nrow=args.sample_img_num)
            matric.update({'ae_step':i,
                      'AE_sample' : [wandb.Image(grid_sample_img, caption=str(i))]
                      })
            matric.update(loss_log)
            wandb.log(matric)    
            

def wandb_update_wgan_ae_only_sample(wandb, args, i, netG, netE, train_loader, device, loss_log):
    if args.wandb : 
        with torch.no_grad() : 
            images = next(iter(train_loader))[0]
            real_cuda = images.to(device)
            output, latent = netE(real_cuda)
            fake_sample = netG(latent).cpu()
        matric = {}
        save_sample_img = fake_sample[:args.sample_img_num ** 2]
        grid_sample_img = make_grid_img(save_sample_img, nrow=args.sample_img_num)
        matric.update({'ae_step':i,
                  'AE_sample' : [wandb.Image(grid_sample_img, caption=str(i))]
                  })
        matric.update(loss_log)
        wandb.log(matric)           

def sample_dissetp_wgan_fake(netE, netG, train_loader, nz, device, sma, just=-1, gaussian_only=False):
    fake_image_list = []
    disable = False if just==-1 else True
    sma = 1 if gaussian_only else sma
    with torch.no_grad() : 
        for image, label in tqdm.tqdm(train_loader, desc='insert fake', disable=disable) :
            batch_size = image.size(0)
            real_cuda = image.to(device)
            
            real_latent = netE(real_cuda)
            mixed_latent = (1-sma)*real_latent + (sma)*torch.randn(real_latent.shape, device=device)
            fake_img = netG(mixed_latent)

            fake_image = fake_img
            fake_image_list.append(fake_image.cpu())
            
            if just>0 and train_loader.batch_size * len(fake_image_list) >= just : break

    fake_image_all = torch.cat(fake_image_list)
    return fake_image_all
        
def sample_wgan_fake(netG, netED, train_loader, nz, device, sma=1.) : 
    fake_image_list = []
    with torch.no_grad() : 
        for image, label in tqdm.tqdm(train_loader, desc='insert fake') :
            batch_size = image.size(0)
            real_cuda = image.to(device)
            
            noise = torch.randn(batch_size, nz, 1, 1, device=real_cuda.device)    
            output, real_latent = netED(real_cuda)
            mixed_latent = (1-sma)*real_latent + (sma)*noise
            fake_mixed = netG(mixed_latent)
            
            fake_image = fake_mixed
            fake_image_list.append(fake_image.cpu())
    fake_image_all = torch.cat(fake_image_list)
    return fake_image_all

def sample_wgan_fake_just(netG, netED, train_loader, nz, device, just, sma=1.) : 
    fake_image_list = []
    with torch.no_grad() : 
        for image, label in train_loader:
            batch_size = image.size(0)
            real_cuda = image.to(device)
            
            noise = torch.randn(batch_size, nz, 1, 1, device=real_cuda.device)    
            output, real_latent = netED(real_cuda)
            mixed_latent = (1-sma)*real_latent + (sma)*noise
            fake_mixed = netG(mixed_latent)
            
            fake_image = fake_mixed
            fake_image_list.append(fake_image.detach().cpu())
            
            if train_loader.batch_size * len(fake_image_list) >= just : break
                
    fake_image_all = torch.cat(fake_image_list)
    return fake_image_all

def sample_wgan_repaint_just(netG, netED, train_loader, nz, device, just) : 
    fake_image_list = []
    with torch.no_grad() : 
        for image, label in train_loader:
            batch_size = image.size(0)
            real_cuda = image.to(device)
            output, insert_feature = netED(real_cuda)
            fake_image = netG(insert_feature)
            fake_image_list.append(fake_image.detach().cpu())
            
            if train_loader.batch_size * len(fake_image_list) >= just : break
                
    fake_image_all = torch.cat(fake_image_list)
    return fake_image_all



        
def wandb_wgan_update(wandb, args, i, inception_model_score, netE, netG, netD, train_loader, nz, device, sma, loss_log):
        
    if args.wandb : 
        if i % args.save_image_interval == 0 :
            inception_model_score.clear_fake()
            fake_sample = sample_dissetp_wgan_fake(netE, netG, train_loader, nz, device, sma, gaussian_only=args.inf_gs)
            inception_model_score.put_fake(fake_sample)

            train_model_to([netE, netG, netD], 'cpu')
            matric = gen_matric(inception_model_score, device)
            train_model_to([netE, netG, netD], device)

        else :
            just = args.sample_img_num ** 2
            fake_sample = sample_dissetp_wgan_fake(netE, netG, train_loader, nz, device, sma, just, gaussian_only=args.inf_gs)
            matric = {}

        save_sample_img = fake_sample[:args.sample_img_num ** 2]
        grid_sample_img = make_grid_img(save_sample_img, nrow=args.sample_img_num)
        matric.update({'Step':i,
                  'sample' : [wandb.Image(grid_sample_img, caption=str(i))],
                  })
        matric.update(loss_log)
        if args.smeg_gan : 
            repaint_sample = sample_dissetp_wgan_fake(netE, netG, train_loader, \
                                                      nz, device, 1.0, args.sample_img_num ** 2, gaussian_only=args.inf_gs)
            grid_repaint_img = make_grid_img(repaint_sample[:args.sample_img_num ** 2], nrow=args.sample_img_num)
            matric.update({'sample_repaint' : [wandb.Image(grid_repaint_img, caption=str(i))]})

        wandb.log(matric)


        
        
def wandb_update(wandb, args, i, inception_model_score, netG, netE, netD, train_loader, nz, device, sma, loss):
    lossD = loss['lossD']
    lossG = loss['lossG']
    loss_alpha = loss['loss_alpha']
    lossE = loss['loss_alpha']
    
    
    if args.wandb : 
        if i % args.save_image_interval == 0 :
            inception_model_score.clear_fake()
            fake_sample = sample_fake(netG, netE, train_loader, nz, device, sigmoid_mul_alpha=sma)
            inception_model_score.put_fake(fake_sample)

            train_model_to([netE, netG, netD], 'cpu')
            matric = gen_matric(inception_model_score, device)
            train_model_to([netE, netG, netD], device)

        else :
            fake_sample = sample_fake_just(netG, netE, train_loader, nz, device, args.sample_img_num ** 2, sigmoid_mul_alpha=sma)
            matric = {}

        save_sample_img = fake_sample[:args.sample_img_num ** 2]
        grid_sample_img = make_grid_img(save_sample_img, nrow=args.sample_img_num)
        matric.update({'Step':i,
                  'sample' : [wandb.Image(grid_sample_img, caption=str(i))],
                  'lossD_real' : lossD[0],
                  'lossD_fake' : lossD[1],
                  'lossD_E(x)' : lossD[2],
                  'lossD' : lossD[0] + lossD[1] + lossD[2],
                  'lossG' : lossG,
                  'loss_alpha_up' : loss_alpha[0],
                  'loss_alpha_down' : loss_alpha[1],
                  'lossE' : lossE,
                  })
        if args.smeg_gan : 
            matric.update({'sma':sma})
            repaint_sample = sample_repaint_just(netG, netE, train_loader, nz, device, args.sample_img_num ** 2)
            grid_repaint_img = make_grid_img(repaint_sample[:args.sample_img_num ** 2], nrow=args.sample_img_num)
            matric.update({'sample_repaint' : [wandb.Image(grid_repaint_img, caption=str(i))]})


        wandb.log(matric)