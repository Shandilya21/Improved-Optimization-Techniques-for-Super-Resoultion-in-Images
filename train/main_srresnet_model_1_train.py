# Adapted from Pytorch-SRRESNet
import argparse, os
import torch
import math, random
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from srresnet import _NetG, _NetD
from dataset import DatasetFromHdf5
from torchvision import models
import torch.utils.model_zoo as model_zoo

# Training settings
parser = argparse.ArgumentParser(description="PyTorch SRResNet")
parser.add_argument("--batchSize", type=int, default=16, help="training batch size")
parser.add_argument("--nEpochs", type=int, default=500, help="number of epochs to train for")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning Rate. Default=1e-4")
parser.add_argument("--step", type=int, default=200, help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=500")
parser.add_argument("--cuda", action="store_true", help="Use cuda?")
parser.add_argument("--resume", default="", type=str, help="Path to checkpoint (default: none)")
parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--threads", type=int, default=0, help="Number of threads for data loader to use, Default: 1")
parser.add_argument("--pretrained", default="", type=str, help="path to pretrained model (default: none)")
parser.add_argument("--vgg_loss", action="store_true", help="Use content loss?")
parser.add_argument("--gpus", default="0", type=str, help="gpu ids (default: 0)")

def main():

    global opt, model, netContent
    opt = parser.parse_args()
    print(opt)

    cuda = opt.cuda
    if cuda:
        print("=> use gpu id: '{}'".format(opt.gpus))
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
        if not torch.cuda.is_available():
                raise Exception("No GPU found or Wrong gpu id, please run without --cuda")

    opt.seed = random.randint(1, 10000)
    print("Random Seed: ", opt.seed)
    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)

    cudnn.benchmark = True

    print("===> Loading datasets")
    train_set = DatasetFromHdf5("/path/to/your/hdf5/data/like/rgb_srresnet_x4.h5")
    training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, \
        batch_size=opt.batchSize, shuffle=True)

    # if opt.vgg_loss:
    #     print('===> Loading VGG model')
    #     netVGG = models.vgg19()
    #     netVGG.load_state_dict(model_zoo.load_url('https://download.pytorch.org/models/vgg19-dcbb9e9d.pth'))
    #     class _content_model(nn.Module):
    #         def __init__(self):
    #             super(_content_model, self).__init__()
    #             self.feature = nn.Sequential(*list(netVGG.features.children())[:-1])
                
    #         def forward(self, x):
    #             out = self.feature(x)
    #             return out

    #     netContent = _content_model()

    print("===> Building model")
    model_G = _NetG()
    model_D = _NetD()
    criterion = nn.MSELoss()
    criterion_D = nn.MSELoss()

    print("===> Setting GPU")
    if cuda:
        model = model.cuda()
        criterion = criterion.cuda()
        if opt.vgg_loss:
            netContent = netContent.cuda() 

    # optionally resume from a checkpoint
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            opt.start_epoch = checkpoint["epoch"] + 1
            model.load_state_dict(checkpoint["model"].state_dict())
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    optionally copy weights from a checkpoint
    if opt.pretrained:
        if os.path.isfile(opt.pretrained):
            print("=> loading model '{}'".format(opt.pretrained))
            weights = torch.load(opt.pretrained)
            model.load_state_dict(weights['model'].state_dict())
        else:
            print("=> no model found at '{}'".format(opt.pretrained))

    print("===> Setting Optimizer")
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)

    print("===> Training")
    for epoch in range(opt.start_epoch, opt.nEpochs + 1):
        train(training_data_loader, optimizer, model, criterion, epoch)
        save_checkpoint(model, epoch)

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10"""
    lr = opt.lr * (0.1 ** (epoch // opt.step))
    return lr 

def train(training_data_loader, optimizer, model, criterion, epoch):

    lr = adjust_learning_rate(optimizer, epoch-1)
    
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    print("Epoch={}, lr={}".format(epoch, optimizer.param_groups[0]["lr"]))
    model.train()

    for iteration, batch in enumerate(training_data_loader, 1):

        input, target = Variable(batch[0]), Variable(batch[1], requires_grad=False)

        if opt.cuda:
            input = input.cuda()
            target = target.cuda()

        out_neg = model_G(input)
        
        out_0, out_1, out_2, out_3, out_4,out_5,out_6, out_7, out_8, out_9 = D(out_neg)


#         target_neg = target.clone()
        t_0, t_1, t_2, t_3, t_4, t_5, t_6, t_7, t_8, t_9 = model_D(target)

        loss_ = criterion(out_neg,target_neg)
        loss_0 = criterion(out_0, t_0)
        loss_1 = criterion(out_1, t_1)
        loss_2 = criterion(out_2, t_2)
        loss_3 = criterion(out_3, t_3)
        loss_4 = criterion(out_4, t_4)
        loss_5 = criterion(out_5, t_5)
        loss_6 = criterion(out_6, t_6)
        loss_7 = criterion(out_7, t_7)
        loss_8 = criterion(out_8, t_8)

        loss = loss_ + loss_0 + loss_1 + loss_2 + loss_3 + loss_4 + loss_5 + loss_6 + loss_7+ loss_8



        # if opt.vgg_loss:
        #     content_input = netContent(output)
        #     content_target = netContent(target)
        #     content_target = content_target.detach()
        #     content_loss = criterion(content_input, content_target)

        optimizer.zero_grad()

        # if opt.vgg_loss:
        #     netContent.zero_grad()
        #     content_loss.backward(retain_graph=True)

        loss.backward()

        optimizer.step()

        if iteration%100 == 0:
            if opt.vgg_loss:
                print("===> Epoch[{}]({}/{}): Loss: {:.5} Content_loss {:.5}".format(epoch, iteration, len(training_data_loader), loss.data[0]))
        else:
            print("===> Epoch[{}]({}/{}): Loss: {:.5}".format(epoch, iteration, len(training_data_loader), loss.data[0]))

def save_checkpoint(model, epoch):
    model_out_path = "checkpoint/" + "model_epoch_{}.pth".format(epoch)
    state = {"epoch": epoch ,"model": model}
    if not os.path.exists("checkpoint/"):
        os.makedirs("checkpoint/")

    torch.save(state, model_out_path)

    print("Checkpoint saved to {}".format(model_out_path))

if __name__ == "__main__":
    main()
