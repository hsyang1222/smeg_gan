from matplotlib import pyplot as plt
import torch
import os
from torchvision.utils import save_image
from torch.autograd import Variable
import numpy as np
from PIL import Image
import tqdm
from train import *
import seaborn as sns
import time

def timeparse(time_str) : 
    if time_str == '' :
        return -1
    splited = time_str.split(":")
    sec = int(splited[0]) * 3600 + int(splited[1]) * 60 + int(splited[2])
    return sec

def check_time_over(start, limit_sec):
    if limit_sec < 0 :
        return False
    return (time.time() - start) > limit_sec

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
        if isinstance(model,torch.nn.Module) :
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
            
            if isinstance(netE,torch.nn.Module) :
                real_latent = netE(real_cuda)
                mixed_latent = (1-sma)*real_latent + (sma)*torch.randn(real_latent.shape, device=device)
            else : 
                 mixed_latent = torch.randn(real_cuda.size(0), nz, 1, 1, device=device)
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


def make_feature_plt(z, M_z, E_x, title) : 
    fig1, ax1 = plt.subplots()
    sns.kdeplot(z, label='N(0,1)', ax=ax1)
    sns.kdeplot(E_x, label='E(x)', ax=ax1)
    sns.kdeplot(M_z, label='Mixed', ax=ax1)
    ax1.legend()
    ax1.set_title('feature ' + title)
    return fig1

def feature_plt_list(z, M_z, E_x,sma) : 
    plt.clf()
    plt_list = []
    assert z.size(1) == M_z.size(1) and M_z.size(1) == E_x.size(1)
    for i in range(z.size(1)) :
        plt_list.append(make_feature_plt(z[:,i].flatten(), M_z[:,i].flatten(), E_x[:,i].flatten(), "dim=%d, sma=%.4f"%(i,sma)))

    return plt_list

def feature_explore_plt_list(train_loader, netE, nz, sma, device) : 
    batch_size = train_loader.batch_size
    gaussian_z = torch.randn(batch_size, nz, 1, 1)
    real_cuda = next(iter(train_loader))[0].to(device)
    encoded_z = netE(real_cuda).detach().cpu()
    sma = float(sma)
    mixed_z =  (1-sma)*encoded_z + sma *gaussian_z
    
    plt_list = feature_plt_list(gaussian_z, mixed_z, encoded_z, sma)
    return plt_list
 
def sample_gaussian(netG, nz, device, size=16) : 
    with torch.no_grad() : 
        fake_latent = torch.randn(size, nz, 1, 1, device=device)
        fake_img = netG(fake_latent)
        
    return fake_img.cpu()

def sample_repaint(netG, netE, train_loader, nz, device, size=16) : 
    sample = [] 
    with torch.no_grad() : 
        for data, label in train_loader :
            real_cuda = data.to(device)
            latent = netE(real_cuda)
            fake_img = netG(latent.view(-1,nz,1,1))
            sample.append(fake_img)
            if len(sample) * train_loader.batch_size >= size : break
        
    sample_torch = torch.cat(sample)[:size]
    return sample_torch.cpu()
        
def sample_batch_gaussian(netG, nz, device, size=50000, batch_size=2048) : 
    sample = []
    with torch.no_grad() :
        while True :
            fake_latent = torch.randn(batch_size, nz, 1, 1, device=device)
            fake_img = netG(fake_latent).cpu()
            sample.append(fake_img)
            if len(sample) * batch_size >= size : break
    
    sample_torch = torch.cat(sample)[:size]
    return sample_torch


def wandb_update(wandb, args, i, inception_model_score, netE, netG, netD, train_loader, nz, device, sma, loss_log, force_metric=False):
        
    if args.wandb : 
        if (i % args.save_image_interval == 0) or force_metric:
            inception_model_score.clear_fake()
            fake_sample = sample_batch_gaussian(netG, nz, device, size=len(train_loader.dataset), batch_size=args.batch_size)
            inception_model_score.put_fake(fake_sample)

            train_model_to([netE, netG, netD], 'cpu')
            matric = gen_matric(inception_model_score, device)
            train_model_to([netE, netG, netD], device)

        else :
            matric = {}

        save_sample_img = sample_gaussian(netG, nz, device, size=args.sample_img_num**2)
        grid_sample_img = make_grid_img(save_sample_img, nrow=args.sample_img_num)
        matric.update({'Step':i,
                  'sample' : [wandb.Image(grid_sample_img, caption=str(i))],
                  })
        matric.update(loss_log)
        if args.smeg_gan : 
            '''
            repaint_sample = sample_repaint(netG, netE, train_loader, nz, device, size=args.sample_img_num**2)
            grid_repaint_img = make_grid_img(repaint_sample, nrow=args.sample_img_num)
            matric.update({'sample_repaint' : [wandb.Image(grid_repaint_img, caption=str(i))]})
            '''
            if (i % args.feature_kde_every ==0) or force_metric : 
                feature_kde = feature_explore_plt_list(train_loader, netE, nz, sma, device)
                matric.update({"feature_kde" : [wandb.Image(plt) for plt in feature_kde]})
            
        wandb.log(matric)


  