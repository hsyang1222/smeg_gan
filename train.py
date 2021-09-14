import torch
from torch.autograd import Variable
import numpy as np

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
    predict_score = netD(latent.detach().view(batch_size,-1))
    predict_score_loss = mse(predict_score, target_score).mean()
    
    
    
    
    
    optimizerD.zero_grad()
    predict_score_loss.backward()
    optimizerD.step()
    
    loss_log = {
        'repaint_loss in ae' : repaint_loss,
        'predict_score_loss in ae' : predict_score_loss,
    }
    
    return loss_log
    
def wgan_smeg_disstep_update_discriminator(netE, netG, netD, optimizerD, real_cuda, nz, sma, clip_value) : 
    batch_size = real_cuda.size(0)
    device= real_cuda.device
    
    latent = netE(real_cuda)
    output_real = netD(latent.view(batch_size, -1)).mean()
    
    with torch.no_grad() :
        mixed_z = (1-sma)*latent + (sma)*torch.randn(latent.shape, device=device)
        fake_img = netG(mixed_z)
        latent_fake = netE(fake_img)
    output_fake = netD(latent_fake.view(batch_size, -1)).mean()
    
    dis_loss = -output_real+output_fake
    optimizerD.zero_grad()
    dis_loss.backward()
    optimizerD.step()
    
    for p in netD.parameters():
        p.data.clamp_(-clip_value, clip_value)
    
    loss_log={'output_real in update d':output_real,
             'output_fake in update d':output_fake,
             'dis_loss in update d':dis_loss,
             'sma in update d' : sma}
    return loss_log
    
def wgan_smeg_disstep_update_generator(netE, netG, netD, optimizerG, real_cuda, nz, sma) :
    batch_size = real_cuda.size(0)
    device= real_cuda.device
    
    with torch.no_grad() : 
        latent = netE(real_cuda)
        mixed_z = (1-sma)*latent + (sma)*torch.randn(latent.shape, device=device)
        
    fake_img = netG(mixed_z)
    latent_fake = netE(fake_img)
    output_fake = netD(latent_fake.view(batch_size, -1)).mean()
    
    img_diff = torch.mean(torch.abs(fake_img - real_cuda))
    img_diff_loss = -img_diff
    
    g_loss = -output_fake + img_diff_loss * 1e-4
    optimizerG.zero_grad()
    g_loss.backward()
    optimizerG.step()

    loss_log={'output_fake in update g':output_fake,
              'img diff in update g':img_diff,
              'sma in update g' : sma
            }
    return loss_log

def wcgan_smeg_disstep_update_alpha(netE, netD, netG, optimizerM, real_cuda, nz, sma) :
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
        output_real = netD(real_latent.view(batch_size, -1)).mean()
        output_fake = netD(mixed_latent.view(batch_size, -1)).mean()
        
    if output_fake>=output_real : 
        optimizerM.zero_grad()
        img_diff_loss.backward()
        optimizerM.step()
        
    loss_log = {'img_diff in alpha update' : img_diff,
        'output_fake in alpha update' : output_fake.item(),
        'output_real in alpha update' : output_real.item(),
        'update? in alpha update' :output_fake>=output_real,
        'sma in update alpha' : sma,
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
    
    #return errD.item()
    return errD_real.item(), errD_fake.item()

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
    return errD.item()


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