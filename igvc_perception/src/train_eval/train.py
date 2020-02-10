import argparse
import cv2
import numpy as np
import os
import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import transforms

from IGVCDataset import IGVCDataset

import models.model
import utils
from collections import defaultdict

from torch.utils.tensorboard import SummaryWriter
import torchvision

def train():
    tb = SummaryWriter()
    images, labels = next(iter(train_loader))
    grid = torchvision.utils.make_grid(images)
    tb.add_image('image', grid)
    tb.add_graph(model.cuda(), images.cuda())
    
    for epoch in range(1, args.epochs + 1):
        batch_metrics = defaultdict(list)
        batch_metrics = {'iters':[],'lrs':[],'train_losses':[],'val_losses':[],'val_accuracies':[]}
        model.train()
        # train loop
        for batch_idx, batch in enumerate(train_loader):
            # prepare data
            images = Variable(batch[0])
            targets = Variable(batch[1])

            if args.cuda:
                images, targets = images.cuda(), targets.cuda()

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            if args.vis and batch_idx % args.log_interval == 0 and images.shape[0] == 1:
                cv2.imshow('output: ', outputs.cpu().data.numpy()[0][0])
                cv2.imshow('target: ', targets.cpu().data.numpy()[0][0])
                cv2.waitKey(10)
            """
            # Learning rate decay.
            if epoch % args.step_interval == 0 and epoch != 1 and batch_idx == 0:
                if args.lr_decay != 1:
                    global lr, optimizer
                    lr *= args.lr_decay
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr
                    print('Learning rate decayed to %f.' % lr)
            """
            if batch_idx % args.log_interval == 0:
                val_loss, val_acc = evaluate('val', n_batches=80)
                train_loss = loss.item()
                batch_metrics['iters'].append(len(train_loader.dataset)*(epoch-1)+batch_idx)
                batch_metrics['lrs'].append(lr)
                batch_metrics['train_losses'].append(train_loss)
                batch_metrics['val_losses'].append(val_loss)
                batch_metrics['val_accuracies'].append(val_acc)

                examples_this_epoch = batch_idx * len(images)
                epoch_progress = 100. * batch_idx / len(train_loader)
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\t'
                    'Train Loss: {:.6f}\tVal Loss: {:.6f}\tVal Acc: {}'.format(
                    epoch, examples_this_epoch, len(train_loader.dataset),
                    epoch_progress, train_loss, val_loss, val_acc))

        print("epoch: {} total train_loss: {} total val_loss: {} total val_acc: {}".format(epoch,sum(batch_metrics['train_losses']), sum(batch_metrics['val_losses']), sum(batch_metrics['val_accuracies'])/len(batch_metrics['val_accuracies'])))

        if (epoch % args.save_interval == 0 and args.save_model):
            save_path = os.path.join(backup_dir, 'IGVCModel' + '_' + str(epoch) + '.pt')
            print('Saving model: %s' % save_path)
            torch.save(model.state_dict(), save_path)
       
        # Tensorboard Implementation
        tb.add_scalar('train loss', sum(batch_metrics['train_losses']), epoch)
        tb.add_scalar('val loss', sum(batch_metrics['val_losses']), epoch)
        tb.add_scalar('val_acc', sum(batch_metrics['val_accuracies'])/len(batch_metrics), epoch)

        for name, weight in model.named_parameters():
            tb.add_histogram(name, weight, epoch)
            tb.add_histogram("{}.grad".format(name), weight.grad, epoch)
        
        metrics_path = os.path.join(backup_dir, 'metrics.npy')
        np.save(metrics_path, batch_metrics)
    tb.close()

def evaluate(split, verbose=False, n_batches=None):
    '''
    Compute loss on val or test data.
    '''
    model.eval()
    loss = 0
    acc = 0
    correct = 0
    n_examples = 0
    if split == 'val':
        loader = val_loader
    elif split == 'test':
        loader = test_loader
    for batch_i, batch in enumerate(loader):
        data, target = batch
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        #data, target = Variable(data, volatile=True), Variable(target)
        data, target = Variable(data), Variable(target)
        output = model(data)
        loss += criterion(output, target).item()
        acc += (np.sum(output.cpu().data.numpy()[target.cpu().data.numpy()!=0] > 0.5) \
            + np.sum(output.cpu().data.numpy()[target.cpu().data.numpy()==0] < 0.5)) / float(args.im_size[1]*args.im_size[2])
        n_examples += output.size(0)

        if n_batches and (batch_i == n_batches-1):
            break

    loss /= n_examples
    acc /= n_examples
    return loss, acc

# 3. Added a run builder for hyperparameter tuning
# 4. Refactor this code similar to unet pytorch

if __name__ == '__main__':
    # Set print option
    np.set_printoptions(threshold=5)
    torch.set_printoptions(precision=10)

    # Get info from argparse class
    args = utils.get_args()
    
    # Check if cuda is available or not
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    # Set randam seed
    torch.manual_seed(args.seed)
    
    if args.cuda:
        print('Using cuda.')
        torch.cuda.manual_seed(args.seed)

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    
    transform = transforms.Compose([
                    transforms.ToTensor(),
            ])
    
    cfg_params = utils.read_data_cfg(args.cfgfile)
    
    train_txt = cfg_params['train']
    test_txt = cfg_params['test']
    backup_dir = cfg_params['backup']

    if args.load_model is not None:
        print('Loading model from %s.' % args.load_model)
        model = models.model.UNet(args.im_size, args.kernel_size)
        model.load_state_dict(torch.load(args.load_model))
    elif args.test:
        print('Missing model file for evaluating test set.')
        exit()
    else:
        model = models.model.UNet(args.im_size, args.kernel_size)

    # Datasets and dataloaders.
    if not args.test:
        train_dataset = IGVCDataset(train_txt, im_size=args.im_size, split='train', transform=transform, val_samples=args.val_samples)
        val_dataset = IGVCDataset(train_txt, im_size=args.im_size, split='val', transform=transform, val_samples=args.val_samples)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
        
        # Optmizer
        lr = args.lr
        print('Initial lr: %f.' % lr)
    else:
        test_dataset = IGVCDataset(test_txt, im_size=args.im_size, split='test', transform=transform)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True, **kwargs)

    # Declare loss function and optimizer
    criterion = F.binary_cross_entropy
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)

    if args.cuda:
        model.cuda()

    if args.test:
        print("Running evaluation on test set.")
        test_loss, test_acc = evaluate('test')
        print('Test loss: %f  Test accuracy: %f' % (test_loss, test_acc))
    else:
        # train the model
        train()