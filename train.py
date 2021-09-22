import torch
from torch.autograd import Variable
import numpy as np

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
    
    loss_log={'up d - output_real':output_real,
             'up d - output_gaussian_fake':output_fake,
             'up d - dis_loss in update d':dis_loss,
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

    loss_log={'up g - output_mixed':output_mixed,
              'up g - output_repaint':output_repaint,
              'up g - sma' : sma,
              'up g - mixed_repaint_diff' : mixed_repaint_diff,
              'up g - repaint_error' : repaint_error,
            }
    return loss_log

def wgan_smeg_v2_update_encoder(netE, netG, netD, optimizerE, real_cuda):
    mse = torch.nn.MSELoss()
    
    latent = netE(real_cuda)
    repaint_img = netG(latent)
    
    repaint_error = mse(repaint_img, real_cuda)
    
    r_loss = repaint_error
    optimizerE.zero_grad()
    r_loss.backward()
    optimizerE.step()
    
    loss_log = {'up e - repaint_error' : repaint_error}
    
    return loss_log

def wcgan_smeg_v2_update_alpha(output_real, output_repaint, output_mixed, output_gsfake, sma, add_alpha=1e-2, per_close=1e-2) : 
            
    add_sma = 0
    end_smeg = False
    diff_fake_real = output_real - output_gsfake
    close_standard = diff_fake_real * per_close
    if sma < 1 :
        if output_repaint - output_mixed < close_standard:
            add_sma = add_alpha
            sma[0] += add_sma
    else : 
        sma = 1
        end_smeg = True
        
    loss_log={
        'up alpha - add_sma' : add_sma,
        'up alpha - sma' : sma,
        'up alpha - output_repaint' : output_repaint,
        'up alpha - output_mixed' : output_mixed,
        'up alpha - end_smeg' : end_smeg,
        'up alpha - output repaint-mixed' : output_repaint - output_mixed,
    }
    
    return loss_log



def dcgan_update_autoencoder(netE, netG, optimizerE, optimizerG, mse, real_cuda, nz) :
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


def wgan_disstep_update_autoencoder(netE, netD, netG, optimizerE, optimizerD, optimizerG, real_cuda) :
    mse = torch.nn.MSELoss(reduction='none')
    
    batch_size = real_cuda.size(0)
    device= real_cuda.device
    
    latent = netE(real_cuda)
    repaint = netG(latent)
    repaint_error = mse(repaint, real_cuda)
    repaint_loss = repaint_error.mean()
    
    optimizerE.zero_grad()
    optimizerG.zero_grad()
    repaint_loss.backward()
    optimizerE.step()
    optimizerG.step()
    
    target_score = -repaint_error.detach().mean(dim=[1,2,3]).view(batch_size,-1)
    predict_repaint_score = netD(repaint.detach())
    predict_repaint_score_loss = mse(predict_repaint_score, target_score).mean()
    
    real_score = torch.zeros(batch_size,1, device=device)
    predict_real_score = netD(real_cuda.detach())
    predict_real_score_loss = mse(predict_real_score, real_score).mean()
    
    score_loss = predict_repaint_score_loss + predict_real_score_loss
    
    
    optimizerD.zero_grad()
    score_loss.backward()
    optimizerD.step()
    
    loss_log = {
        'up ae - repaint_loss' : repaint_loss,
        'up ae - score_loss' : score_loss,
        'up ae - pr_repaint_score' : predict_repaint_score.mean(),
        'up ae - actual repaint_score' : target_score.mean(),
        'up ae - pr_real_score' : predict_real_score.mean(),
        'up ae - pr_real_score_loss' : predict_repaint_score_loss,
    }
    
    return loss_log


def wcgan_smeg_disstep_update_encoder(netE, netD, netG, optimizerE, real_cuda, nz, sma) : 
    mse = torch.nn.MSELoss()
    
    batch_size = real_cuda.size(0)
    device= real_cuda.device
    
    latent = netE(real_cuda)
    repaint = netG(latent)
    repaint_error = mse(repaint, real_cuda)
    repaint_loss = repaint_error
    
    optimizerE.zero_grad()
    repaint_loss.backward()
    optimizerE.step()

    loss_log = {
        'up e - repaint_loss' : repaint_loss,
    }
    
    return loss_log    
    
    

    
def wgan_smeg_disstep_update_discriminator(netE, netG, netD, optimizerD, real_cuda, nz, sma, clip_value):
    batch_size = real_cuda.size(0)
    device= real_cuda.device
    
    latent = netE(real_cuda)
    
    output_real = netD(real_cuda).mean()
    
    with torch.no_grad() :
        output_repaint = netD(netG(latent)).mean()
        mixed_z = (1-sma)*latent + (sma)*torch.randn(latent.shape, device=device)
        fake_img = netG(mixed_z)
    output_fake = netD(fake_img).mean()
    
    dis_loss = -output_real+output_fake
    optimizerD.zero_grad()
    dis_loss.backward()
    optimizerD.step()
    
    for p in netD.parameters():
        p.data.clamp_(-clip_value, clip_value)
    
    loss_log={'up d - output_real':output_real,
             'up d - output_fake':output_fake,
              'up d - output_repaint' : output_repaint,
             'up d - dis_loss in update d':dis_loss,
             'up d - sma' : sma}
    return loss_log
    
def wgan_smeg_disstep_update_generator(netE, netG, netD, optimizerG, real_cuda, nz, sma):
    batch_size = real_cuda.size(0)
    device= real_cuda.device
    
    with torch.no_grad() : 
        latent = netE(real_cuda)
        mixed_z = (1-sma)*latent + (sma)*torch.randn(latent.shape, device=device)
        
    fake_img = netG(mixed_z)
    output_fake = netD(fake_img).mean()
    '''
    img_diff = torch.mean(torch.abs(fake_img - real_cuda))
    img_diff_loss = -img_diff
    + img_diff_loss * 1e-4
    '''
    g_loss = -output_fake 
    optimizerG.zero_grad()
    g_loss.backward()
    optimizerG.step()

    loss_log={'up g - output_fake':output_fake,
              'up g - sma' : sma
            }
    return loss_log


def wcgan_smeg_disstep_update_plain_alpha(train_loader, netE, netD, netG, optimizerM, real_cuda, nz, sma, device, add_alpha, conti=False) :
    output_real_sum = 0.
    output_repaint_sum = 0.
    output_fake_sum= 0.
    
    while True : 
        with torch.no_grad() : 
            for image, label in train_loader:
                real_cuda = image.to(device)

                latent = netE(real_cuda)
                mixed_z = (1-sma)*latent + (sma)*torch.randn(latent.shape, device=device)
                fake_img = netG(mixed_z)
                repaint_img = netG(latent)

                output_real = netD(real_cuda)
                output_repaint = netD(repaint_img)
                output_fake = netD(fake_img)

                output_real_sum += output_real.sum()
                output_repaint += output_repaint.sum()
                output_fake_sum += output_fake.sum()

        len_data = len(train_loader.dataset)
        output_real_mean = output_real_sum / len_data
        output_repaint_mean = output_repaint_sum / len_data
        output_fake_mean = output_fake_sum / len_data

        if sma > 1 :
            sma[0] = 1.
            break
        elif output_fake_mean >= output_repaint_mean: 
            sma[0] = sma[0] + add_alpha
            if not conti : break
        else :
            break

    loss_log={'up a - output_real':output_real_mean,
         'up a - output_fake':output_fake_mean,
          'up a - output_repaint' : output_repaint_mean,
         'up a - sma' : sma}
    
    return loss_log
    

def wcgan_smeg_disstep_update_alpha(netE, netD, netG, optimizerM, real_cuda, nz, sma):
    batch_size = real_cuda.size(0)
    device = real_cuda.device
     
    #Generate(mixed)
    with torch.no_grad() : 
        real_latent = netE(real_cuda)
        repaint_img = netG(real_latent)
    mixed_latent = (1-sma)*real_latent + (sma)*torch.randn(real_latent.shape, device=device)
    fake_img = netG(mixed_latent)
    
    img_diff = torch.mean(torch.abs(fake_img - repaint_img))
    img_diff_loss = -img_diff
    
    with torch.no_grad() : 
        output_repaint = netD(repaint_img).mean()
        output_fake = netD(netG(mixed_latent)).mean()
        
    if output_fake>=output_repaint : 
        optimizerM.zero_grad()
        img_diff_loss.backward()
        optimizerM.step()
        
    loss_log = {'up m - img_diff' : img_diff,
        'up m - output_fake' : output_fake.item(),
        'up m - output_repaint' : output_repaint.item(),
        'up m - update?' : int(output_fake>=output_repaint),
        'up m - sma' : sma,
       } 

    return loss_log



def wgan_encdis_update_autoencoder(netED, netG, optimizerED, optimizerG, mse, criterion, real_cuda, nz) : 
    batch_size = real_cuda.size(0)
    device= real_cuda.device
    output_real, latent_vector = netED(real_cuda)
    repaint = netG(latent_vector)
    
    #ed repaint loss
    edmse_loss = mse(repaint, real_cuda)
    
    #ed dis real loss
    eddis_real_loss = output_real.mean()
    
    #ed dis fake loss
    output_fake, latent_vector_repaint = netED(repaint)
    eddis_fake_loss = output_fake.mean()
    
    #optimize ed
    ed_loss = edmse_loss - eddis_real_loss + eddis_fake_loss
    optimizerED.zero_grad()
    ed_loss.backward()
    optimizerED.step()
    
    
    # g repaint loss
    repaint = netG(latent_vector.detach())
    gmse_loss = mse(repaint, real_cuda)
    
    # g dis fake loss
    output_fake, latent_vector_repaint = netED(repaint)
    gdis_fake_loss = output_fake.mean()
    
    #optimize g
    g_loss = gmse_loss - gdis_fake_loss
    optimizerG.zero_grad()
    g_loss.backward()
    optimizerG.step()
    
    
    loss_log = {'AE_ed_mse_loss' : edmse_loss.item(),
                'AE_ed_dis_real_loss' : eddis_real_loss.item(),
                'AE_ed_dis_fake_loss' : eddis_fake_loss.item(),
                'AE_g_dis_fake_loss' : gdis_fake_loss.item(),
                'AE_g_mse_loss' : gmse_loss.item()
               }
        
    
    return loss_log
    
    
    
    

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

def dcgan_smeg_update_discriminator(netD, netG, netE, optimizerD, real_cuda, criterion, nz, sigmoid_mul_alpha=1.):
    batch_size = real_cuda.size(0)
    device = real_cuda.device

    # train with real

    label_ones = torch.ones(batch_size, 1, device=device)
    output_real = netD(real_cuda)
    errD_real = criterion(output_real, label_ones)

    # train with fake
    fake_latent_insert = gen_latent_feature(netE, real_cuda, batch_size, nz, sigmoid_mul_alpha)
    fake = netG(fake_latent_insert).detach()

    label_zeros = torch.zeros(batch_size, 1, device=device)
    output_fake = netD(fake)
    errD_fake = criterion(output_fake, label_zeros)
    errD = errD_real + errD_fake
    optimizerD.zero_grad()
    errD.backward()
    optimizerD.step()
    return errD_real.item(), errD_fake.item()

def dcgan_smeg_update_generator(netD, netG, netE, optimizerG, real_cuda, criterion, nz, sigmoid_mul_alpha=1.) : 
    batch_size = real_cuda.size(0)
    device = real_cuda.device
    
    #train with fake
    
    label = torch.ones(batch_size, 1, device=device)
    
    fake_latent_insert = gen_latent_feature(netE, real_cuda, batch_size, nz, sigmoid_mul_alpha)
    fake = netG(fake_latent_insert)
    output_fake = netD(fake)
    
    errD_fake = criterion(output_fake, label)
    errD = errD_fake
    optimizerG.zero_grad()
    errD.backward()
    optimizerG.step()
    return errD.item()

def dcgan_smeg_update_alpha(netD, netG, netE, optimizerM, real_cuda, criterion, nz, sma) : 
    batch_size = real_cuda.size(0)
    device = real_cuda.device
    
    noise = torch.randn(batch_size, nz, 1, 1, device=real_cuda.device)    
    #Generate(mixed)
    real_latent = netE(real_cuda).detach() 
    mixed_latent = (1-sma)*real_latent + (sma)*noise
    fake_mixed = netG(mixed_latent)
    #Generate(E(x))
    fake_Ex = netG(real_latent)
    #loss1
    loss1 = -torch.mean(torch.abs(fake_mixed - fake_Ex))
    
    #D(Generated(mixed))
    output_midex = netD(fake_mixed.detach())
    #D(Generate(E(x)))
    output_Ex = netD(fake_Ex.detach())
    loss2 = -torch.abs((1-sma)*output_Ex.mean() - sma*output_midex.mean())
    
    loss_alpha = loss1 + loss2
    optimizerM.zero_grad()
    loss_alpha.backward()
    optimizerM.step()
    return loss1.item(), loss2.item()
    
    

def dcgan_update_discriminator(netD, netG, netE, optimizerD, real_cuda, criterion, nz):
    batch_size = real_cuda.size(0)
    device = real_cuda.device

    # train with real
    label_ones = torch.ones(batch_size, 1, device=device)
    output_real = netD(real_cuda)
    errD_real = criterion(output_real, label_ones)

    # train with fake
    fake_latent_insert = torch.randn(batch_size, nz, 1, 1, device=real_cuda.device)
    fake = netG(fake_latent_insert).detach()
    label_zeros = torch.zeros(batch_size, 1, device=device)
    output_fake = netD(fake)
    errD_fake = criterion(output_fake, label_zeros)
    
    # optimize
    errD = errD_real + errD_fake
    optimizerD.zero_grad()
    errD.backward()
    optimizerD.step()
    return errD_real.item(), errD_fake.item()

def dcgan_update_generator(netD, netG, netE, optimizerG, real_cuda, criterion, nz) : 
    batch_size = real_cuda.size(0)
    device = real_cuda.device
    
    #train with fake
    label = torch.ones(batch_size, 1, device=device)
    fake_latent_insert = torch.randn(batch_size, nz, 1, 1, device=real_cuda.device)
    fake = netG(fake_latent_insert)
    output_fake = netD(fake)
    errD_fake = criterion(output_fake, label)
    errD = errD_fake
    
    optimizerG.zero_grad()
    errD.backward()
    optimizerG.step()
    return errD.item()

def wgan_update_discriminator(netD, netG, netE, optimizerD, real_cuda, criterion, nz, clip_value=0.01):
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
    
    loss_log={'up d - output_real': -errD_real,
         'up d - output_fake':errD_fake,
         'up d - loss':errD}
    
    return loss_log

def wgan_update_generator(netD, netG, netE, optimizerG, real_cuda, criterion, nz) : 
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
        'up g - output_fake' : -errD_fake
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