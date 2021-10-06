import torch
from torch.autograd import Variable
import numpy as np

def update_autoencoder(args, netE, netDld, netG, optimizerE, optimizerDld, optimizerG, real_cuda, nz) :
    if args.autoencoder in ['aae'] :
        loss_e = wgan_update_aae(args, netE, netDld, netG, optimizerE, optimizerDld, optimizerG, real_cuda, nz)
    elif args.autoencoder in ['autoencoder'] :
        loss_e = dcgan_update_autoencoder(netE, netG, optimizerE, optimizerG, real_cuda, nz)
    return loss_e
    
def wgan_update_aae(args, netE, netDld, netG, optimizerE, optimizerDld, optimizerG, real_cuda, nz) :
    mse = torch.nn.MSELoss()
    batch_size = real_cuda.size(0)
    device = args.device
    #update Dld
    gaussian_z = torch.randn(batch_size, nz, 1, 1, device=device)
    encoded_z = netE(real_cuda).detach()
    
    output_gaussian = netDld(gaussian_z).mean()
    output_encoded = netDld(encoded_z).mean()
    
    loss_dld = -output_gaussian + output_encoded
    optimizerDld.zero_grad()
    loss_dld.backward()
    optimizerDld.step()

    
    #update G
    repaint_img = netG(encoded_z)
    
    loss_g = mse(repaint_img, real_cuda)
    optimizerG.zero_grad()
    loss_g.backward()
    optimizerG.step()
    
    #update E
    encoded_z = netE(real_cuda)
    output_encoded = netDld(encoded_z).mean()
    repaint_img = netG(encoded_z)
    
    
    loss_e_repaint = mse(repaint_img, real_cuda)
    loss_e_dld = -output_encoded
    
    loss_e = loss_e_repaint+loss_e_dld
    optimizerE.zero_grad()
    loss_e.backward()
    optimizerE.step()
    
    loss_e = {   'up aae_d - output_gaussian':output_gaussian.item(),
                 'up aae_d - output_encoded':output_encoded.item(),
                 'up aae_g - repaint_loss':loss_g.item(),
                 'up aae_e - repaint_loss':loss_e_repaint.item(),
                 'up aae_e - output_encoded':output_encoded.item()}
    
    return loss_e
    
    
def update_discriminator(args, netE, netG, netD, optimizerD, real_cuda, sma) :
    nz = args.nz
    if args.base_model in ['dcgan'] : 
        loss_d=dcgan_update_discriminator(netD, netG, netE, optimizerD, real_cuda, nz)
    elif args.base_model in ['wgan'] :
        loss_d=wgan_update_discriminator(netD, netG, netE, optimizerD, real_cuda, nz, args.clip_value)
    elif args.base_model in ['lsgan']:
        loss_d=lsgan_update_discriminator(netD, netG, netE, optimizerD, real_cuda, nz)
    return loss_d
        
def update_generatator(args, netE, netG, netD, optimizerG, real_cuda, sma) : 
    nz = args.nz
    if args.base_model in ['dcgan'] : 
        loss_g=dcgan_update_generator(netD, netG, netE, optimizerG, real_cuda, nz)
    elif args.base_model in ['wgan'] :
        loss_g=wgan_update_generator(netD, netG, netE, optimizerG, real_cuda, nz)
    elif args.base_model in ['lsgan']:
        loss_g=lsgan_update_generator(netD, netG, netE, optimizerG, real_cuda, nz)
    return loss_g

def update_smeg_discriminator(args, netE, netG, netD, optimizerD, real_cuda, sma) :
    nz = args.nz
    if args.base_model in ['dcgan'] : 
        loss_d=dcgan_smeg_update_discriminator(netD, netG, netE, optimizerD, real_cuda, nz)
    elif args.base_model in ['wgan'] :
        loss_d=wgan_smeg_v2_update_discriminator(netE, netG, netD, optimizerD, real_cuda, nz, sma, args.clip_value)
    elif args.base_model in ['lsgan']:
        loss_d=lsgan_update_discriminator(netD, netG, netE, optimizerD, real_cuda, nz)
    return loss_d

def update_smeg_generatator(args, netE, netG, netD, optimizerG, real_cuda, sma) : 
    nz = args.nz
    if args.base_model in ['dcgan'] : 
        loss_g=dcgan_smeg_update_generator(netD, netG, netE, optimizerG, real_cuda, nz, sma, hyper_img_diff=args.hyper_img_diff)
    elif args.base_model in ['wgan'] :
        loss_g=wgan_smeg_v2_update_generator(netE, netG, netD, optimizerG, real_cuda, nz, sma, hyper_img_diff=args.hyper_img_diff)
    elif args.base_model in ['lsgan']:
        loss_g=lsgan_smeg_update_generator(netE, netG, netD, optimizerG, real_cuda, nz, sma, hyper_img_diff=args.hyper_img_diff)
    return loss_g

def lsgan_update_discriminator(netD, netG, netE, optimizerD, real_cuda, nz):
    batch_size = real_cuda.size(0)
    device = real_cuda.device

    real_label = torch.ones(batch_size, 1, device=device)
    fake_label = torch.zeros(batch_size, 1, device=device)
    
    # train with real
    output_real = netD(real_cuda)
    errD_real = 0.5 * torch.mean((output_real-real_label)**2) 
    
    
    # train with fake
    with torch.no_grad() : 
        fake_latent_insert = torch.randn(batch_size, nz, 1, 1, device=real_cuda.device)
        fake = netG(fake_latent_insert)
        
    output_fake = netD(fake)
    errD_fake = 0.5 * torch.mean((output_fake-fake_label)**2) 
    
    # optimize
    errD = errD_real + errD_fake
    optimizerD.zero_grad()
    errD.backward()
    optimizerD.step()
    
    
    loss_log={'up d - output_real': output_real.mean().item(),
         'up d - output_gaussian_fake':output_fake.mean().item(),
         'up d - loss':errD.item()}
    
    return loss_log

def lsgan_update_generator(netD, netG, netE, optimizerG, real_cuda, nz) :
    batch_size = real_cuda.size(0)
    device = real_cuda.device
    
    real_label = torch.ones(batch_size, 1, device=device)
    
    #train with fake
    fake_latent_insert = torch.randn(batch_size, nz, 1, 1, device=real_cuda.device)
    fake = netG(fake_latent_insert)
    output_fake = netD(fake)
    errG = 0.5 * torch.mean((output_fake - real_label)**2)
    
    optimizerG.zero_grad()
    errG.backward()
    optimizerG.step()
    
    loss_log={
        'up g - output_fake' : output_fake.mean().item()
        }
    
    return loss_log

def lsgan_smeg_update_generator(netE, netG, netD, optimizerG, real_cuda, nz, sma, hyper_img_diff=1e-4) :
    mse = torch.nn.MSELoss()
    batch_size = real_cuda.size(0)
    device= real_cuda.device
    
    real_label = torch.ones(batch_size, 1, device=device)
    
    with torch.no_grad() : 
        latent = netE(real_cuda)
        mixed_z = (1-sma)*latent + (sma)*torch.randn(latent.shape, device=device)
        
        
    repaint_img = netG(latent)    
    mixed_img = netG(mixed_z)
    
    output_repaint = netD(repaint_img)
    output_mixed = netD(mixed_img)
    
    loss_output_mixed = 0.5 * torch.mean((output_mixed - real_label)**2)
    
    mixed_repaint_diff = mse(mixed_img, repaint_img)
    repaint_error = mse(repaint_img, real_cuda)
    
    g_loss = loss_output_mixed + repaint_error - mixed_repaint_diff * hyper_img_diff
    optimizerG.zero_grad()
    g_loss.backward()
    optimizerG.step()

    loss_log={'up g - output_mixed':output_mixed.mean().item(),
              'up g - output_repaint':output_repaint.mean().item(),
              'up g - sma' : sma,
              'up g - mixed_repaint_diff' : mixed_repaint_diff.item(),
              'up g - repaint_error' : repaint_error.item(),
            }
    return loss_log


def wgan_smeg_v2_update_discriminator(netE, netG, netD, optimizerD, real_cuda, nz, sma, clip_value):
    batch_size = real_cuda.size(0)
    device= real_cuda.device
    
    with torch.no_grad() :
        gaussian_latent = torch.randn(batch_size, nz, 1, 1, device=device)
        fake_gaussian_img = netG(gaussian_latent)
    
    output_real = netD(real_cuda).mean()
    output_fake = netD(fake_gaussian_img).mean()
    
    dis_loss = -output_real+output_fake
    optimizerD.zero_grad()
    dis_loss.backward()
    optimizerD.step()
    
    for p in netD.parameters():
        p.data.clamp_(-clip_value, clip_value)
    
    loss_log={'up d - output_real':output_real.item(),
             'up d - output_gaussian_fake':output_fake.item(),
             'up d - loss':dis_loss.item(),
             'up d - sma' : sma}
    return loss_log

def wgan_smeg_v2_update_generator(netE, netG, netD, optimizerG, real_cuda, nz, sma, hyper_img_diff=1e-4):
    mse = torch.nn.MSELoss()
    batch_size = real_cuda.size(0)
    device= real_cuda.device
    
    with torch.no_grad() : 
        latent = netE(real_cuda)
        mixed_z = (1-sma)*latent + (sma)*torch.randn(latent.shape, device=device)
        
        
    repaint_img = netG(latent)    
    mixed_img = netG(mixed_z)
    
    output_repaint = netD(repaint_img).mean()
    output_mixed = netD(mixed_img).mean()
    
    mixed_repaint_diff = mse(mixed_img, repaint_img)
    repaint_error = mse(repaint_img, real_cuda)
    
    g_loss = -output_mixed + repaint_error - mixed_repaint_diff * hyper_img_diff
    optimizerG.zero_grad()
    g_loss.backward()
    optimizerG.step()

    loss_log={'up g - output_mixed':output_mixed.item(),
              'up g - output_repaint':output_repaint.item(),
              'up g - sma' : sma,
              'up g - mixed_repaint_diff' : mixed_repaint_diff.item(),
              'up g - repaint_error' : repaint_error.item(),
            }
    return loss_log

def semg_update_encoder(args, netE, netDld, netG, optimizerE, optimizerDld, real_cuda, nz):
    if args.autoencoder in ['aae'] :
        loss_e = update_encoder_disdl(netE, netDld, netG, optimizerE, optimizerDld, real_cuda, nz)
    else:
        loss_e = update_encoder(netE, netG, optimizerE, real_cuda)
    return loss_e
    
    
def update_encoder_disdl(netE, netDld, netG, optimizerE, optimizerDld, real_cuda, nz):
    mse = torch.nn.MSELoss()
    batch_size = real_cuda.size(0)
    #update Dld
    gaussian_z = torch.randn(batch_size, nz, 1, 1, device=real_cuda.device)
    encoded_z = netE(real_cuda).detach()
    
    output_gaussian = netDld(gaussian_z).mean()
    output_encoded = netDld(encoded_z).mean()
    
    loss_dld = -output_gaussian + output_encoded
    optimizerDld.zero_grad()
    loss_dld.backward()
    optimizerDld.step()
    
    #update E
    encoded_z = netE(real_cuda)
    output_encoded = netDld(encoded_z).mean()
    repaint_img = netG(encoded_z)
    
    loss_e_repaint = mse(repaint_img, real_cuda)
    loss_e_dld = -output_encoded
    
    loss_e = loss_e_repaint+loss_e_dld
    optimizerE.zero_grad()
    loss_e.backward()
    optimizerE.step()
    
    loss_e = {   'up edld - output_gaussian':output_gaussian.item(),
                 'up edld - output_encoded':output_encoded.item(),
                 'up edld - loss_dld' : loss_dld.item(),
                 'up edld - repaint_loss':loss_e_repaint.item(),
                 'up edld - output_encoded':output_encoded.item()}
    
    return loss_e

def update_encoder(netE, netG, optimizerE, real_cuda):
    mse = torch.nn.MSELoss()
    
    latent = netE(real_cuda)
    repaint_img = netG(latent)
    
    repaint_error = mse(repaint_img, real_cuda)
    
    r_loss = repaint_error
    optimizerE.zero_grad()
    r_loss.backward()
    optimizerE.step()
    
    loss_log = {'up e - repaint_error' : repaint_error.item()}
    
    return loss_log

def smeg_update_alpha(output_real, output_repaint, output_mixed, output_gsfake, sma, add_alpha=1e-2, per_close=1e-2) : 
            
    add_sma = 0
    end_smeg = 0
    diff_fake_real = output_real - output_gsfake
    close_standard = diff_fake_real * per_close
    if ((output_repaint - output_mixed) < close_standard) or per_close==-1:
            add_sma = add_alpha
            sma[0] += add_sma
    if sma>1 : 
        sma = 1
        end_smeg = 1
        
    loss_log={
        'up alpha - add_sma' : add_sma,
        'up alpha - sma' : sma,
        'up alpha - output_real' : output_real.item(),
        'up alpha - output_repaint' : output_repaint.item(),
        'up alpha - output_mixed' : output_mixed.item(),
        'up alpha - end_smeg' : end_smeg,
        'up alpha - output repaint-mixed' : (output_repaint - output_mixed).item(),
        'up alpha - diff_fake_real' : diff_fake_real.item(),
        'up alpha - close_standard' : close_standard,
    }
    
    return loss_log



def dcgan_update_autoencoder(netE, netG, optimizerE, optimizerG, real_cuda, nz) :
    mse = torch.nn.MSELoss()
    
    batch_size = real_cuda.size(0)
    latent_vector = netE(real_cuda)
    real_latent_4dim = latent_vector.view(batch_size,nz,1,1)
    repaint = netG(real_latent_4dim)

    mse_loss = mse(repaint, real_cuda)
    optimizerE.zero_grad()
    optimizerG.zero_grad()
    mse_loss.backward()
    optimizerE.step()
    optimizerG.step()
    return mse_loss.item()
    

def dcgan_smeg_frozen_z(netE, real_cuda, nz):
    batch_size = real_cuda.size(0)
    if netE is not None : 
        with torch.no_grad() : 
            latent_vector = netE(real_cuda)
            real_latent_4dim = latent_vector.view(batch_size,nz,1,1)
    else :
        real_latent_4dim = torch.randn(batch_size, nz, 1,1)
    return real_latent_4dim
    
    

def dcgna_smeg_update_encoder(netE, netG, real_cuda, optimizerE, mse, nz):
    batch_size = real_cuda.size(0)
    latent_vector = netE(real_cuda)
    real_latent_4dim = latent_vector.view(batch_size,nz,1,1)
    repaint = netG(real_latent_4dim)

    mse_loss = mse(repaint, real_cuda)
    optimizerE.zero_grad()
    mse_loss.backward()
    optimizerE.step()

    return mse_loss.item()

def gen_latent_feature(netE, real_cuda, batch_size, nz, sigmoid_mul_alpha) : 
    noise = torch.randn(batch_size, nz, 1, 1, device=real_cuda.device)
    if netE is not None and sigmoid_mul_alpha != 1.: 
        with torch.no_grad() :
            latent_vector = netE(real_cuda)
            real_latent_4dim = latent_vector.view(batch_size,nz,1,1)
            real_feature = real_latent_4dim.detach()
        fake_latent_insert  = (1-sigmoid_mul_alpha)*real_feature + sigmoid_mul_alpha*noise
    else : 
        fake_latent_insert = noise
    return fake_latent_insert

def dcgan_smeg_update_discriminator(netD, netG, netE, optimizerD, real_cuda, nz):
    batch_size = real_cuda.size(0)
    device= real_cuda.device
    
    criterion = torch.nn.BCELoss()
    
    with torch.no_grad() :
        gaussian_latent = torch.randn(batch_size, nz, 1, 1, device=device)
        fake_gaussian_img = netG(gaussian_latent)
    
    output_real = netD(real_cuda)
    output_fake = netD(fake_gaussian_img)
    
    label_ones = torch.ones(batch_size,1,device=device)
    label_zeros = torch.zeros(batch_size,1,device=device)
    
    dis_loss = criterion(output_real, label_ones)+criterion(output_fake, label_zeros)
    optimizerD.zero_grad()
    dis_loss.backward()
    optimizerD.step()
    
    loss_log={'up d - output_real':output_real.mean().item(),
             'up d - output_gaussian_fake':output_fake.mean().item(),
             'up d - loss':dis_loss.item()}
    return loss_log

def dcgan_smeg_update_generator(netD, netG, netE, optimizerG, real_cuda, nz, sma=1., hyper_img_diff=1e-4) : 
    mse = torch.nn.MSELoss()
    bce = torch.nn.BCELoss()
    batch_size = real_cuda.size(0)
    device= real_cuda.device
    
    with torch.no_grad() : 
        latent = netE(real_cuda)
        mixed_z = (1-sma)*latent + (sma)*torch.randn(latent.shape, device=device)
        
        
    repaint_img = netG(latent)    
    mixed_img = netG(mixed_z)
    
    output_repaint = netD(repaint_img)
    output_mixed = netD(mixed_img)
    
    mixed_repaint_diff = mse(mixed_img, repaint_img)
    repaint_error = mse(repaint_img, real_cuda)
    
    label_ones=torch.ones(batch_size, 1, device=device)
    
    g_loss = bce(output_mixed, label_ones) + repaint_error - mixed_repaint_diff * hyper_img_diff
    optimizerG.zero_grad()
    g_loss.backward()
    optimizerG.step()

    loss_log={'up g - output_mixed':output_mixed.mean().item(),
              'up g - output_repaint':output_repaint.mean().item(),
              'up g - sma' : sma,
              'up g - mixed_repaint_diff' : mixed_repaint_diff.item(),
              'up g - repaint_error' : repaint_error.item(),
              'up g - loss' : g_loss.item(),
              'up g - mean of E(x)' : latent.mean(dim=(0,2,3)).detach().cpu()
            }
    return loss_log


def dcgan_update_discriminator(netD, netG, netE, optimizerD, real_cuda, nz):
    batch_size = real_cuda.size(0)
    device= real_cuda.device
    
    criterion = torch.nn.BCELoss()
    
    with torch.no_grad() :
        gaussian_latent = torch.randn(batch_size, nz, 1, 1, device=device)
        fake_gaussian_img = netG(gaussian_latent)
        #print("gen img size : ", fake_gaussian_img.shape)
    
    output_real = netD(real_cuda)
    output_fake = netD(fake_gaussian_img)
    #print("dis size:", output_real.shape)
    
    label_ones = torch.ones(batch_size,1,device=device)
    label_zeros = torch.zeros(batch_size,1,device=device)
    
    dis_loss = criterion(output_real, label_ones)+criterion(output_fake, label_zeros)
    optimizerD.zero_grad()
    dis_loss.backward()
    optimizerD.step()
    
    loss_log={'up d - output_real':output_real.mean().item(),
             'up d - output_gaussian_fake':output_fake.mean().item(),
             'up d - loss':dis_loss.item()}
    return loss_log

def dcgan_update_generator(netD, netG, netE, optimizerG, real_cuda, nz) : 
    batch_size = real_cuda.size(0)
    device = real_cuda.device
    
    criterion = torch.nn.BCELoss()
    
    #train with fake
    fake_latent_insert = torch.randn(batch_size, nz, 1, 1, device=real_cuda.device)
    fake = netG(fake_latent_insert)
    
    label_ones = torch.ones(batch_size, 1, device=device)
    
    output_fake = netD(fake)
    errD_fake = criterion(output_fake, label_ones)
    errD = errD_fake
    
    optimizerG.zero_grad()
    errD.backward()
    optimizerG.step()
    
    loss_log={
        'up g - output_fake' : output_fake.mean().item(),
        'up g - loss' : errD.item(),
        }
    
    return loss_log

def wgan_update_discriminator(netD, netG, netE, optimizerD, real_cuda, nz, clip_value=0.01):
    batch_size = real_cuda.size(0)
    device = real_cuda.device

    # train with real
    output_real = netD(real_cuda)
    errD_real = -torch.mean(output_real)

    # train with fake
    fake_latent_insert = torch.randn(batch_size, nz, 1, 1, device=real_cuda.device)
    fake = netG(fake_latent_insert).detach()
    output_fake = netD(fake)
    errD_fake = torch.mean(output_fake)
    
    # optimize
    errD = errD_real + errD_fake
    optimizerD.zero_grad()
    errD.backward()
    optimizerD.step()
    
    # clip param
    for p in netD.parameters():
            p.data.clamp_(-clip_value, clip_value)
    
    loss_log={'up d - output_real': -errD_real.item(),
         'up d - output_gaussian_fake':errD_fake.item(),
         'up d - loss':errD}
    
    return loss_log

def wgan_update_generator(netD, netG, netE, optimizerG, real_cuda, nz) : 
    batch_size = real_cuda.size(0)
    device = real_cuda.device
    
    #train with fake
    fake_latent_insert = torch.randn(batch_size, nz, 1, 1, device=real_cuda.device)
    fake = netG(fake_latent_insert)
    output_fake = netD(fake)
    errD_fake = -torch.mean(output_fake)
    errD = errD_fake
    
    optimizerG.zero_grad()
    errD.backward()
    optimizerG.step()
    
    loss_log={
        'up g - output_fake' : -errD_fake.item()
        }
    
    return loss_log


def wgan_smeg_update_discriminator(netED, netG, optimizerED, real_cuda, criterion, nz, sma, clip_value=0.01):
    batch_size = real_cuda.size(0)
    device = real_cuda.device

    # train with real
    output_real, latent = netED(real_cuda)
    errD_real = torch.mean(output_real)

    # train with fake
    fake_latent_insert = (1-sma)*latent + (sma)*torch.randn(batch_size, nz, 1, 1,device=device)
    fake = netG(fake_latent_insert).detach()
    output_fake, latent = netED(real_cuda)
    errD_fake = torch.mean(output_fake)
    
    # train with E(x)
    reapint = netG(latent).detach()
    output_repaint, latent_repaint = netED(reapint)
    errD_repaint = torch.mean(output_repaint)
    
    errD = -errD_real + errD_fake
    
    # optimize
    
    optimizerED.zero_grad()
    errD.backward()
    optimizerED.step()
    
    # clip param
    for p in netED.parameters():
            p.data.clamp_(-clip_value, clip_value)
    
    #return errD.item()
    loss_log = {'dis_errD_real' : errD_real.item(),
            'dis_errD_fake' : errD_fake.item(),
            'dis_errD_repaint' : errD_repaint.item(),
            'dis_errD' : errD.item(),
           }

    return loss_log


def wgan_smeg_update_generator(netED, netG, optimizerG, real_cuda, criterion, nz, sma) : 
    batch_size = real_cuda.size(0)
    device = real_cuda.device
    
    #train with fake
    output_real, latent = netED(real_cuda)
    fake_latent_insert = (1-sma)*latent + (sma)*torch.randn(batch_size, nz, 1, 1, device=device)
    fake = netG(fake_latent_insert.detach())
    output_fake, latent_fake = netED(fake)
    errD_fake = -torch.mean(output_fake)
    errD = errD_fake
    
    optimizerG.zero_grad()
    errD.backward()
    optimizerG.step()
    
    loss_log = {'g_errD_fake' : errD_fake.item(),
       }
    
    return loss_log

def wcgan_smeg_update_alpha(netED, netG, optimizerM, real_cuda, criterion, nz, sma) : 
    batch_size = real_cuda.size(0)
    device = real_cuda.device
    
    noise = torch.randn(batch_size, nz, 1, 1, device=real_cuda.device)    
    #Generate(mixed)
    output, real_latent = netED(real_cuda)
    mixed_latent = (1-sma)*real_latent.detach() + (sma)*noise
    fake_mixed = netG(mixed_latent)
    #Generate(E(x))
    fake_Ex = netG(real_latent)
    #loss1
    loss1 = torch.mean(torch.abs(fake_mixed - fake_Ex))
    loss2 = torch.tensor([0.])
    '''
    #D(Generated(mixed))
    output_midex = netD(fake_mixed.detach())
    #D(Generate(E(x)))
    output_Ex = netD(fake_Ex.detach())
    loss2 = -torch.abs((1-sma)*output_Ex.mean() - sma*output_midex.mean())
    '''
    #check could maximize sma
    output_fake_miexed, latent = netED(fake_mixed)
    output_fake_repaint, latent = netED(real_cuda)
    
    output_fake_miexed = output_fake_miexed.mean()
    output_fake_repaint = output_fake_repaint.mean()
    
    loss_alpha = -loss1
    if output_fake_miexed>=output_fake_repaint : 
        optimizerM.zero_grad()
        loss_alpha.backward()
        optimizerM.step()
        
    loss_log = {'sma_loss' : loss_alpha.item(),
        'output_fake_miexed' : output_fake_miexed.item(),
        'output_fake_repaint' : output_fake_repaint.item(),
       } 

    return loss_log
    
def wcgna_smeg_update_encoder(netD, netG, netE, optimizerE, real_cuda, mse, nz, sma) : 
    batch_size = real_cuda.size(0)
    latent_vector = netE(real_cuda)
    real_latent_4dim = latent_vector.view(batch_size,nz,1,1)
    repaint = netG(real_latent_4dim)
    
    # update encoder only
    # fake image is better than repaint

    mse_loss = mse(repaint, real_cuda)
    
    with torch.no_grad() : 
        output_repaint = netD(repaint).mean()
        output_fake = netD(netG(gen_latent_feature(netE, real_cuda, batch_size, nz, sma))).mean()
    
    if output_fake >= output_repaint :
        optimizerE.zero_grad()
        mse_loss.backward()
        optimizerE.step()

    return mse_loss.item()