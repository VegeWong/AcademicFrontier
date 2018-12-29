import torch
import torchvision

class Adversary(object):
    def __init__(self, net, method:String, **kargs):
        super(Adversary, self).__init__()

        self.net = net
        self.method = getattr(self, method)
        
    def generate(self, x, y, attack:Boolean ,**kargs):
        x.requires_grad = True
        self.net.zero_grad()
        
        return self.method(x, y, kargs)


    def fgsm(self, x, y, **kargs):
        pred = self.net(x)
        xm = x + kargs['eps']* 

    def ifgsm