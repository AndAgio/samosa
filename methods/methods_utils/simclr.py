import time
import torch.optim
from .transform_layers import HorizontalFlipLayer
from .contrastive_loss import get_similarity_matrix, NT_xent
from .ccal_util import AverageMeter, normalize
import numpy as np

def semantic_train_epoch(args, epoch, model, criterion, optimizer, scheduler, loader,
          simclr_aug=None, linear=None, linear_optim=None):
    assert simclr_aug is not None
    assert args.sim_lambda == 1.0

    hflip = HorizontalFlipLayer().to(args.gpu)

    batch_time = AverageMeter()
    data_time = AverageMeter()

    losses = dict()
    losses['cls'] = AverageMeter()
    losses['sim'] = AverageMeter()

    check = time.time()
    #print('len(multi-datasets loader): ', len(loader))
    for n, (index, (images, labels)) in enumerate(loader):
        model.train()
        count = n #* args.n_gpus  # number of trained samples

        data_time.update(time.time() - check)
        check = time.time()

        ### SimCLR loss ###
        if args.dataset != 'imagenet':
            batch_size = images.size(0)
            images = images.to(args.gpu)
            images_pair = hflip(images.repeat(2, 1, 1, 1))  # 2B with hflip
        else:
            batch_size = images[0].size(0)
            images1, images2 = images[0].to(args.gpu), images[1].to(args.gpu)
            images_pair = torch.cat([images1, images2], dim=0)  # 2B

        labels = labels.to(args.gpu)

        images_pair = simclr_aug(images_pair)  # transform

        _, outputs_aux = model(images_pair, simclr=True, penultimate=True)

        simclr = normalize(outputs_aux['simclr'])  # normalize
        sim_matrix = get_similarity_matrix(simclr) #, multi_gpu=args.multi_gpu)
        loss_sim = NT_xent(sim_matrix, temperature=0.5) * args.sim_lambda

        ### total loss ###
        loss = loss_sim

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        scheduler.step(epoch - 1 + n / len(loader))
        lr = optimizer.param_groups[0]['lr']

        batch_time.update(time.time() - check)

        ### Post-processing stuffs ###
        simclr_norm = outputs_aux['simclr'].norm(dim=1).mean()

        ### Linear evaluation ###
        outputs_linear_eval = linear(outputs_aux['penultimate'].detach())
        # loss_linear = criterion(outputs_linear_eval, labels.repeat(2).type(torch.LongTensor).cuda())

        idx = np.where(labels.cpu().numpy() < args.known_class)[0]
        loss_linear = criterion(outputs_linear_eval[idx], labels[idx].type(torch.LongTensor).to(args.gpu)) #.repeat(2)

        linear_optim.zero_grad()
        loss_linear.backward()
        linear_optim.step()

        ### Log losses ###
        losses['cls'].update(0, batch_size)
        losses['sim'].update(loss_sim.item(), batch_size)

        if epoch%100 == 0 and count % 3000 == 0:
            print('[Epoch %3d; %3d] [Time %.3f] [Data %.3f] [LR %.5f]\n'
                 '[LossC %f] [LossSim %f]' %
                 (epoch, count, batch_time.value, data_time.value, lr,
                  losses['cls'].value, losses['sim'].value))
        check = time.time()