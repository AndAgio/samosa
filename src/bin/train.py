import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import time
import pickle
import math
import glob
# Import custom stuff
from src.datasets import Cutout, CIFAR10, CIFAR100, SVHN, FashionMNIST, ImageNet, TinyImageNet
from src.models import ResNet18, ResNet50, Inception, enable_running_stats, disable_running_stats
from src.models import VGG19, MobileNetV3Small, MobileNetV3Large, InceptionV3, WRN2810, ViT
from src.models import ResNet18NoBn, ResNet50NoBn, VGG19NoBn, WRN2810NoBn
from src.optimizers import SAM, SGD, Adam
from src.optimizers.schedulers import GradualWarmupScheduler, CosineAnnealingWarmupRestarts, CustomLRSchedule
from .runner import Runner


class Trainer(Runner):
    def __init__(self, settings):
        if settings.deactivate_bn:
            experiment_name='standard_{}_over_{}_with_{}_nobn_sched_{}'.format(settings.model, settings.dataset, settings.optimizer, settings.lr_sched)
        else:
            experiment_name='standard_{}_over_{}_with_{}_bn_sched_{}'.format(settings.model, settings.dataset, settings.optimizer, settings.lr_sched)
        super().__init__(settings=settings, experiment_name=experiment_name)
        self.gather_dataset()
        self.setup_model()
        self.setup_training()

    def gather_dataset(self):
        self.logger.print_it('Gathering dataset "{}". This may take a while...'.format(self.settings.dataset))
        # Image Preprocessing
        if self.settings.dataset in ['cifar10', 'cifar100']:
            normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                            std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
            # Setup train transforms
            train_transform = transforms.Compose([])
            if self.settings.data_augmentation:
                train_transform.transforms.append(transforms.RandomCrop(32, padding=4))
                train_transform.transforms.append(transforms.RandomHorizontalFlip())
            train_transform.transforms.append(transforms.ToTensor())
            train_transform.transforms.append(normalize)
            if self.settings.cutout_holes > 0:
                train_transform.transforms.append(
                    Cutout(n_holes=self.settings.cutout_holes, length=self.settings.cutout_length))
            # Setup test transforms
            test_transform = transforms.Compose([transforms.ToTensor(), normalize])
        elif self.settings.dataset == 'svhn':
            mean = [0.4377, 0.4438, 0.4728]
            std = [0.1980, 0.2010, 0.1970]
            train_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
            test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        elif self.settings.dataset == 'fmnist':
            mean = [0.2861]
            std = [0.3530]
            train_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
            test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        elif self.settings.dataset == 'imagenet':
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])
            train_transform = transforms.Compose([])
            train_transform.transforms.append(transforms.RandomResizedCrop(224))
            if self.settings.data_augmentation:
                train_transform.transforms.append(transforms.RandomHorizontalFlip())
            train_transform.transforms.append(transforms.ToTensor())
            train_transform.transforms.append(normalize)
            test_transform = transforms.Compose([transforms.Resize(256),
                                                    transforms.CenterCrop(224),
                                                    transforms.ToTensor(),
                                                    normalize])
        elif self.settings.dataset == 'tiny_imagenet':
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                             std=[0.229, 0.224, 0.225])
            # Setup train transforms for Tiny ImageNet
            train_transform = transforms.Compose([])
            if self.settings.data_augmentation:
                train_transform.transforms.append(transforms.RandomCrop(64, padding=4))
                train_transform.transforms.append(transforms.RandomHorizontalFlip())
            train_transform.transforms.append(transforms.ToTensor())
            train_transform.transforms.append(normalize)
            if self.settings.cutout_holes > 0:
                train_transform.transforms.append(
                    Cutout(n_holes=self.settings.cutout_holes, length=self.settings.cutout_length))
            # Setup test transforms for Tiny ImageNet
            test_transform = transforms.Compose([transforms.ToTensor(), normalize])
        else:
            raise ValueError('Dataset "{}" is not available!'.format(self.settings.dataset))

        # Load the appropriate train and test datasets
        if self.settings.dataset == 'cifar10':
            self.num_classes = 10
            self.im_size = (32,32)
            self.im_channels = 3
            root = os.path.join(self.settings.datasets_folder, 'cifar10')
            self.train_dataset = CIFAR10(root=root, train=True, transform=train_transform, download=True)
            self.test_dataset = CIFAR10(root=root, train=False, transform=test_transform, download=True)
        elif self.settings.dataset == 'cifar100':
            self.num_classes = 100
            self.im_size = (32,32)
            self.im_channels = 3
            root = os.path.join(self.settings.datasets_folder, 'cifar100')
            self.train_dataset = CIFAR100(root=root, train=True, transform=train_transform, download=True)
            self.test_dataset = CIFAR100(root=root, train=False, transform=test_transform, download=True)
        elif self.settings.dataset == 'svhn':
            self.im_channels = 3
            self.im_size = (32, 32)
            self.num_classes = 10
            root = os.path.join(self.settings.datasets_folder, 'svhn')
            self.train_dataset = SVHN(root=root, split='train', download=True, transform=train_transform)
            self.train_dataset.targets = self.train_dataset.labels
            self.test_dataset = SVHN(root=root, split='test', download=True, transform=test_transform)
            self.test_dataset.targets = self.test_dataset.labels
        elif self.settings.dataset == 'fmnist':
            self.im_channels = 1
            self.im_size = (28, 28)
            self.num_classes = 10
            root = os.path.join(self.settings.datasets_folder, 'fmnist')
            self.train_dataset = FashionMNIST(root=root, train=True, download=True, transform=train_transform)
            self.test_dataset = FashionMNIST(root=root, train=False, download=True, transform=test_transform)
        elif self.settings.dataset == 'imagenet':
            self.im_channels = 3
            self.im_size = (224, 224)
            self.num_classes = 1000
            root = os.path.join(self.settings.datasets_folder, 'imagenet')
            self.train_dataset = ImageNet(root=root, split='train', download=True, transform=train_transform)
            self.test_dataset = ImageNet(root=root, split='val', download=True, transform=test_transform)
        elif self.settings.dataset == 'tiny_imagenet':
            self.im_channels = 3
            self.im_size = (64, 64)
            self.num_classes = 200
            root = os.path.join(self.settings.datasets_folder, 'tiny_imagenet')
            self.train_dataset = TinyImageNet(root=root, train=True, transform=train_transform)
            self.test_dataset = TinyImageNet(root=root, train=False, transform=test_transform)
        else:
            raise ValueError('Dataset "{}" is not available!'.format(self.settings.dataset))

        # Get indices of examples that should be used for training
        self.train_indx = np.array(range(len(self.train_dataset)))

        # Get indices of examples that should be used for testing
        self.test_indx = np.array(range(len(self.test_dataset)))

        self.logger.print_it('Gathered dataset "{}":\tTraining samples = {} & Testing samples = {}'.format(self.settings.dataset,
                                                                                            len(self.train_dataset),
                                                                                            len(self.test_dataset)))

    def setup_model(self):
        self.logger.print_it('Setting up model "{}"...'.format(self.settings.model))
        # Setup model
        if self.settings.model == 'resnet18':
            if self.settings.deactivate_bn:
                model = ResNet18NoBn(channel=self.im_channels, num_classes=self.num_classes)
            else:
                model = ResNet18(channel=self.im_channels, num_classes=self.num_classes)
        elif self.settings.model == 'resnet50':
            if self.settings.deactivate_bn:
                model = ResNet50NoBn(channel=self.im_channels, num_classes=self.num_classes)
            else:
                model = ResNet50(channel=self.im_channels, num_classes=self.num_classes)
        elif self.settings.model == 'wideresnet':
            if self.settings.deactivate_bn:
                model = WRN2810NoBn(channel=self.im_channels, num_classes=self.num_classes, im_size=self.im_size)
            else:
                model = WRN2810(channel=self.im_channels, num_classes=self.num_classes, im_size=self.im_size)
        elif self.settings.model == 'vgg19':
            if self.settings.deactivate_bn:
                model = VGG19NoBn(channel=self.im_channels, num_classes=self.num_classes, im_size=self.im_size)
            else:
                model = VGG19(channel=self.im_channels, num_classes=self.num_classes, im_size=self.im_size)
        elif self.settings.model == 'mobile_small':
            model = MobileNetV3Small(channel=self.im_channels, num_classes=self.num_classes, im_size=self.im_size)
        elif self.settings.model == 'mobile_large':
            model = MobileNetV3Large(channel=self.im_channels, num_classes=self.num_classes, im_size=self.im_size)
        elif self.settings.model == 'inception':
            model = InceptionV3(channel=self.im_channels, num_classes=self.num_classes, im_size=self.im_size)
        elif self.settings.model == 'vit':
            model = ViT(pretrained=True, in_channels=self.im_channels, num_classes=self.num_classes, image_size=self.im_size)
        else:
            print('Specified model "{}" not recognized!'.format(self.settings.model))
        # Move model to device
        self.model = model.to(self.device)
        self.logger.print_it('Model setup done!')

    def setup_training(self):
        self.logger.print_it('Setting up training with "{}" optimizer...'.format(self.settings.optimizer))
        self.criterion = nn.CrossEntropyLoss(reduction='none').to(self.device)

        # Setup optimizer
        if self.settings.optimizer == 'adam':
            self.optimizer = Adam(self.model.parameters(),
                                            lr=self.settings.learning_rate)
        elif self.settings.optimizer == 'sgd':
            self.optimizer = SGD(self.model.parameters(), 
                                            lr=self.settings.learning_rate,
                                            momentum=0.9,
                                            nesterov=False)
        elif self.settings.optimizer == 'sam':
            self.optimizer = SAM(params=self.model.parameters(),
                                base_optimizer=SGD,
                                lr=self.settings.learning_rate,
                                momentum=0.9,
                                nesterov=False,
                                rho=0.05)
        else:
            raise ValueError('Specified optimizer "{}" not supported. Options are: adam and sgd and sam'.format(self.settings.optimizer))
        
        if self.settings.lr_sched == 'const':
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=1)
        elif self.settings.lr_sched == 'warmup_step':
            scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=math.ceil(self.settings.epochs/3), gamma=0.1)
            self.scheduler = GradualWarmupScheduler(self.optimizer, multiplier=1, total_epoch=math.ceil(self.settings.epochs/40), after_scheduler=scheduler)
            self.scheduler.step()
        elif self.settings.lr_sched == 'warmup_exp':
            scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.98)
            self.scheduler = GradualWarmupScheduler(self.optimizer, multiplier=1, total_epoch=math.ceil(self.settings.epochs/40), after_scheduler=scheduler)
            self.scheduler.step()
        elif self.settings.lr_sched == 'warmup_cosine':
            cycle_steps = math.ceil(self.settings.epochs/5)
            warmup_steps = math.ceil(cycle_steps/10)
            max_lr=self.settings.lr
            min_lr=max_lr/100
            self.scheduler = CosineAnnealingWarmupRestarts(self.optimizer, first_cycle_steps=cycle_steps, cycle_mult=1.0, max_lr=max_lr, min_lr=min_lr, warmup_steps=warmup_steps, gamma=0.5)
        elif self.settings.lr_sched == 'step':
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=math.ceil(self.settings.epochs/3), gamma=0.1)
        elif self.settings.lr_sched == 'exp':
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.98)
        elif self.settings.lr_sched == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.settings.epochs)
        elif self.settings.lr_sched == 'custom':
            self.scheduler = CustomLRSchedule(self.optimizer)
        else:
            raise ValueError('Learning rate scheduler "{}" not available!'.format(self.settings.lr_sched))

        self.logger.print_it('Training setup done!')

    def train(self, store_each_epoch=False):
        if self.settings.resume:
            self.load_last_resume_ckpt()
        else:
            # Initialize dictionary to save statistics for every example presentation
            self.best_acc = 0
            self.elapsed_time = 0
            self.logger.print_it('Starting training of "{}" for "{}" from scratch. This will take a while...'.format(self.settings.model,
                                                                                                                    self.settings.dataset))
            self.test_accs = []
            self.epoch = 1

        while(self.epoch <= self.settings.epochs):
            start_time = time.time()

            self.train_epoch()
            test_acc = self.test_epoch()
            self.test_accs.append(test_acc)

            epoch_time = time.time() - start_time
            self.elapsed_time += epoch_time
            h, m, s = Trainer.convert_to_hms(self.elapsed_time)
            self.logger.print_it('Elapsed time for epoch {}: {}:{:02d}:{:02d}'.format(self.epoch,h,m,s))

            # Update optimizer step
            if self.settings.optimizer != 'adam':
                self.scheduler.step()

            if store_each_epoch:
                self.save_model('epoch_{}.pt'.format(self.epoch))

            # Save checkpoint when best model
            if test_acc > self.best_acc:
                self.logger.print_it('New Best model at epoch {}: \t Top1-acc = {:.2f}'.format(self.epoch, test_acc*100))
                self.save_model('best.pt')
                self.best_acc = test_acc
            
            # Save model when last epoch
            if self.epoch == (self.settings.epochs):
                self.logger.print_it('Saving last model: \t Top1-acc = {:.2f}'.format(test_acc*100))
                self.save_model('last.pt')

            self.store_resume_ckpt()

            self.epoch += 1

        h, m, s = Trainer.convert_to_hms(self.elapsed_time)
        self.logger.print_it('Training of "{}" for "{}" completed in: {}:{:02d}:{:02d}'.format(self.settings.model, self.settings.dataset, h, m, s))
        # Compute metrics and store them in folder
        return self.test_accs

    def train_epoch(self):
        train_loss = 0.
        correct = 0.
        total = 0.

        self.model.train()

        # Get permutation to shuffle trainset
        trainset_permutation_inds = np.random.permutation(np.arange(len(self.train_dataset)))

        batch_size = self.settings.batch_size
        for batch_idx, batch_start_ind in enumerate(range(0, len(self.train_dataset), batch_size)):

            # Get trainset indices for batch
            batch_inds = trainset_permutation_inds[batch_start_ind: batch_start_ind + batch_size]

            # Get batch inputs and targets, transform them appropriately
            transformed_trainset = []
            targets = []
            for ind in batch_inds:
                transformed_trainset.append(self.train_dataset.__getitem__(ind)[0])
                targets.append(torch.tensor(self.train_dataset.__getitem__(ind)[1]))
            inputs = torch.stack(transformed_trainset)
            print(targets)
            targets = torch.stack(targets)
            # targets = torch.LongTensor(np.array(self.train_dataset.targets)[batch_inds].tolist())

            # Map to available device
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # Compute loss and predictions
            if self.settings.optimizer=='sam':
                # first forward-backward step
                enable_running_stats(self.model)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.mean().backward()
                self.optimizer.first_step(zero_grad=True)
                # second forward-backward step
                disable_running_stats(self.model)
                self.criterion(self.model(inputs), targets).mean().backward()
                self.optimizer.second_step(zero_grad=True)  
            else:
                # Forward propagation, compute loss, get predictions
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
            _, predicted = torch.max(outputs.data, 1)

            # Update loss, backward propagate, update optimizer
            loss = loss.mean()
            train_loss += loss.item()
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()
            if self.settings.optimizer != 'sam':
                loss.backward()
                self.optimizer.step()

            # Print message on console
            metrics = {'loss': train_loss / total,
                        'acc': correct.item() / total}
            self.print_message(epoch_index=self.epoch, total_epochs=self.settings.epochs,
                            index_batch=batch_idx+1, total_batches=len(self.train_dataset) // batch_size + 1,
                            metrics=metrics, mode='train')

        self.logger.set_logger_newline()

    def test_epoch(self):
        test_loss = 0.
        correct = 0.
        total = 0.
        test_batch_size = 32

        self.model.eval()

        for batch_idx, batch_start_ind in enumerate(range(0, len(self.test_dataset), test_batch_size)):

            # Get batch inputs and targets
            transformed_testset = []
            targets = []
            for ind in range(batch_start_ind, min(len(self.test_dataset), batch_start_ind + test_batch_size)):
                transformed_testset.append(self.test_dataset.__getitem__(ind)[0])
                targets.append(torch.tensor(self.test_dataset.__getitem__(ind)[1]))
            inputs = torch.stack(transformed_testset)
            targets = torch.stack(targets)
            # targets = torch.LongTensor(np.array(self.test_dataset.targets)[batch_start_ind:batch_start_ind + test_batch_size].tolist())

            # Map to available device
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            # Forward propagation, compute loss, get predictions
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss = loss.mean()
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

            # Print message on console
            metrics = {'loss': test_loss / total,
                        'acc': correct.item() / total}
            self.print_message(epoch_index=self.epoch, total_epochs=self.settings.epochs,
                            index_batch=batch_idx+1, total_batches=len(self.test_dataset) // test_batch_size + 1,
                            metrics=metrics, mode='test')
        self.logger.set_logger_newline()

        # Add test accuracy to dict
        acc = correct.item() / total
        return acc

    def print_message(self, epoch_index, total_epochs, index_batch, total_batches, metrics, mode='train'):
        message = '| EPOCH: {}/{} |'.format(epoch_index, total_epochs)
        bar_length = 10
        progress = float(index_batch) / float(total_batches)
        if progress >= 1.:
            progress = 1
        block = int(round(bar_length * progress))
        message += '[{}]'.format('=' * block + ' ' * (bar_length - block))
        message += '| {}: '.format(mode.upper())
        if metrics is not None:
            train_metrics_message = ''
            index = 0
            for metric_name, metric_value in metrics.items():
                train_metrics_message += '{}={:.5f}{} '.format(metric_name, metric_value,
                                                            ',' if index < len(metrics.keys()) - 1 else '')
                index += 1
            message += train_metrics_message
        message += '|'
        self.logger.print_it_same_line(message)

    def run(self):
        test_accs = self.train()
        pickles_folder = os.path.join(self.settings.out_folder, 'out_standard_train', 'pickles', self.experiment_name)
        os.makedirs(pickles_folder, exist_ok=True)
        pickle_file = os.path.join(pickles_folder, 'seed_{}.pkl'.format(self.settings.seed))
        with open(pickle_file, 'wb') as f:
            pickle.dump(test_accs, f)
        self.load_last_model()
        # self.model = self.last_model
        last_acc = self.test_epoch(epoch=-1)
        self.load_best_model()
        # self.model = self.best_model
        best_acc = self.test_epoch(epoch=-1)
        message = 'Best model performance = {}%\n' \
                'Last model performance = {}%\n'.format(best_acc*100, last_acc*100)
        txt_folder = os.path.join(self.settings.out_folder, 'out_standard_train', 'txts', self.experiment_name)
        os.makedirs(txt_folder, exist_ok=True)
        txt_file = os.path.join(txt_folder, 'seed_{}.txt'.format(self.settings.seed))
        with open(txt_file, 'x') as f:
            f.write(message)

    def store_resume_ckpt(self):
        state = {
                'epoch': self.epoch,
                'arch': self.settings.model,
                'best_acc': self.best_acc,
                'elapsed_time': self.elapsed_time,
                'test_accs': self.test_accs,
                'state_dict': self.model.state_dict(),
                'optimizer' : self.optimizer.state_dict(),
                'scheduler' : self.scheduler.state_dict(),
            }
        checkpoint_folder = os.path.join(self.settings.resume_ckpts_folder, self.experiment_name, 'seed_{}'.format(self.settings.seed))
        os.makedirs(checkpoint_folder, exist_ok=True)
        torch.save(state, os.path.join(checkpoint_folder, 'epoch={}.pth.tar'.format(self.epoch)))

    def load_last_resume_ckpt(self):
        checkpoint_folder = os.path.join(self.settings.resume_ckpts_folder, self.experiment_name, 'seed_{}'.format(self.settings.seed))
        checkpoints = glob.glob(os.path.join(checkpoint_folder, '*.pth.tar'))
        found_epochs = [int(check.split('epoch=')[-1].split('.pth.tar')[0]) for check in checkpoints]
        if found_epochs == []:
            self.logger.print_it('No resume checkpoint found! Are you sure you wanted to resume?')
            self.logger.print_it('Starting from scratch...')
            self.best_acc = 0
            self.elapsed_time = 0
            self.test_accs = []
            self.epoch = 1
            self.logger.print_it('Starting training of "{}" for "{}" from scratch. This will take a while...'.format(self.settings.model,
                                                                                                                    self.settings.dataset))
        else:
            last_epoch = max(found_epochs)
            checkpoint_to_load = os.path.join(self.settings.resume_ckpts_folder, self.experiment_name, 'seed_{}'.format(self.settings.seed), 'epoch={}.pth.tar'.format(last_epoch))
            # Read variables
            checkpoint = torch.load(checkpoint_to_load, map_location=self.device)
            self.epoch = checkpoint['epoch'] + 1
            self.best_acc = checkpoint['best_acc']
            self.elapsed_time = checkpoint['elapsed_time']
            self.test_accs = checkpoint['test_accs']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])