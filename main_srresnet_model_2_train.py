# Adapted from Pytorch-SRRESNet
import argparse, os
import torch
import math, random
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from srresnet1 import _NetG, _NetD
from dataset import DatasetFromHdf5
from torchvision import models
import torch.utils.model_zoo as model_zoo
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] ="5"

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True
set_session(tf.Session(config=config))



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
    train_set = DatasetFromHdf5("srresnet_x4.h5")
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
    model_G = _NetG() #changed
    model_D = _NetD() #changed
    criterion = nn.MSELoss()
    criterion_D = nn.CrossEntropy() #changed

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

    # optionally copy weights from a checkpoint
    if opt.pretrained:
        if os.path.isfile(opt.pretrained):
            print("=> loading model '{}'".format(opt.pretrained))
            weights = torch.load(opt.pretrained)
            model.load_state_dict(weights['model'].state_dict())
        else:
            print("=> no model found at '{}'".format(opt.pretrained))

    print("===> Setting Optimizer")
    optimizer_g = optim.Adam(model_G.parameters(), lr=opt.lr) #changed
    optimizer_d = optim.Adadelta(model_D.parameters()) #changed

    print("===> Training")
    for epoch in range(opt.start_epoch, opt.nEpochs + 1):
        train(training_data_loader, optimizer, model, criterion, epoch)
        save_checkpoint(model, epoch)

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10"""
    lr = opt.lr * (0.1 ** (epoch // opt.step))
    return lr 
c = 1
def train(training_data_loader, optimizer, model_G, model_D, criterion, epoch):

    lr = adjust_learning_rate(optimizer, epoch-1)
    
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    print("Epoch={}, lr={}".format(epoch, optimizer.param_groups[0]["lr"]))
    model_G.train()
    model_D.train()

    for iteration, batch in enumerate(training_data_loader, 1):

        input, target = Variable(batch[0]), Variable(batch[1], requires_grad=False)

        if opt.cuda:
            input = input.cuda()
            target = target.cuda()

        out_neg = model_G(input)
        
        out_0, out_1, out_2, out_3, out_4,out_5,out_6, out_7, out_8, out_9 = model_D(out_neg) #changed


#         target_neg = target.clone()
        t_0, t_1, t_2, t_3, t_4, t_5, t_6, t_7, t_8, t_9 = model_D(target) #changed
 
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

        l = loss.mean()
        l0 = loss_0.mean()
        l1 = loss_1.mean()
        l2 = loss_2.mean()
        l3 = loss_3.mean()
        l4 = loss_4.mean()
        l5 = loss_5.mean()
        l6 = loss_6.mean()
        l7 = loss_7.mean()
        l8 = loss_8.mean()

        fp = torch.exp()
        sum_fp = fp(l)+fp(l1)+fp(l2)+fp(l3)+fp(l4)+fp(l5)+fp(l6)+fp(l7)+fp(l8)
        l_0 = fp(l)/sum_fp
        l_1 = fp(l1)/sum_fp
        l_2 = fp(l2)/sum_fp
        l_3 = fp(l3)/sum_fp
        l_4 = fp(l4)/sum_fp
        l_5 = fp(l5)/sum_fp
        l_6 = fp(l6)/sum_fp
        l_7 = fp(l7)/sum_fp
        l_8 = fp(l8)/sum_fp



        loss_overall = (l* fp(l))/sum_fp \
            + (l1 * fp(l1))/sum_fp \
             + (l2 * fp(l2))/sum_fp \
              + (l3 * fp(l3))/sum_fp \
               + (l4 * fp(l4))/sum_fp \
                + (l5 * fp(f5))/sum_fp \
                 + (l6 * fp(l6))/sum_fp \
                 + (l7 * fp(l7))/sum_fp  \
                  + (l8 * fp(l8))/sum_fp  #changed
        




        # if opt.vgg_loss:
        #     content_input = netContent(output)
        #     content_target = netContent(target)
        #     content_target = content_target.detach()
        #     content_loss = criterion(content_input, content_target)

        optimizer.zero_grad()

        # if opt.vgg_loss:
        #     netContent.zero_grad()
        #     content_loss.backward(retain_graph=True)

        loss_overall.backward()

        optimizer.step()

        if iteration%100 == 0:
            if opt.vgg_loss:
                print("===> Epoch[{}]({}/{}): Loss: {:.5} Content_loss {:.5}".format(epoch, iteration, len(training_data_loader), loss_overall.data[0]))
        else:
            print("===> Epoch[{}]({}/{}): Loss: {:.5}".format(epoch, iteration, len(training_data_loader), loss_overall.data[0]))
     #coverage
     c_1 = 0.9 * (c) + 0.1 * (l_0)
     c_2 = 0.9 * (c) + 0.1 * (l_1)
     c_3 = 0.9 * (c) + 0.1 * (l_2)
     c_4 = 0.9 * (c) + 0.1 * (l_3)
     c_5 = 0.9 * (c) + 0.1 * (l_4)
     c_6 = 0.9 * (c) + 0.1 * (l_5)
     c_7 = 0.9 * (c) + 0.1 * (l_6)
     c_8 = 0.9 * (c) + 0.1 * (l_7)
     c_9 = 0.9 * (c) + 0.1 * (l_8)
    
    c+ = 1
def save_checkpoint(model, epoch):
    model_out_path = "checkpoint/" + "model_epoch_{}.pth".format(epoch)
    state = {"epoch": epoch ,"model": model}
    if not os.path.exists("checkpoint/"):
        os.makedirs("checkpoint/")

    torch.save(state, model_out_path)

    print("Checkpoint saved to {}".format(model_out_path))

if __name__ == "__main__":
    main()
