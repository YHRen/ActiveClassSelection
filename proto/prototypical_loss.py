import torch
import torch.nn as nn

class Prototypical(nn.Module):
    def __init__(self, base_model, num_class, num_features, lmd=0.01, proto_per_cls = 1 ):
        super(Prototypical, self).__init__()
        self.base_model    = base_model
        self.num_features  = num_features
        self.num_class     = num_class
        self.proto_per_cls = proto_per_cls
        self.prototypes    = nn.Parameter(torch.zeros( num_class, proto_per_cls, num_features, requires_grad=True))
        self.proto_loss    = Prototype_Loss(lmd)

    def forward(self, x):
        bsz      = x.shape[0]
        ft       = self.base_model(x)
        self.ft  = ft
        dst      = self.distance(ft)
        dst, idx2= dst.max(dim=2)
        idx1     = dst.argmax(dim=1)
        pred     = dst.sigmoid()
        proto    = self.prototypes[idx1, idx2[:, idx1]]
        return pred, dst, ft, proto
            

    def distance(self, feat):
        """ 
            feat[ures] has dimension [ b_sz x ft_sz ]
            prot[otypes]  has dimension [ cls_sz x ft_sz ]
            with ppc > 1, prototypes has dimension [ (cls_sz*ppc) x ft_sz ]
            output has dim of [b_sz x cls_sz]
        """
        bsz = feat.shape[0]
        pt  = self.prototypes.view(-1, self.num_features)
        f2  = feat.pow(2).sum(dim=1).unsqueeze(1)
        c2  = pt.pow(2).sum(dim=1)
        cf  = feat @ pt.t()
        rst = f2 - 2 * cf + c2
        rst = rst.view(bsz, self.num_class, -1)
        return -rst

    def get_loss(self):
        return self.proto_loss


class Prototype_Loss(nn.Module) :
    def __init__(self, lmd=0.01):
        super(Prototype_Loss, self).__init__()
        self.ces = nn.CrossEntropyLoss()
        self.lmd = lmd 
        self.l1  = 0.0 
        self.l2  = 0.0 
        
    def forward(self, input, target):
        """ 
            input = (prediction, distance, features, selected_prototypes)
        """
        _, dst, ft, proto = input
        l1 = self.ces(dst, target)
        l2 = torch.sum((ft - proto)**2) / 2 / ft.size()[0] / ft.size()[1]
        self.l1 = 0.1*self.l1 + 0.9*l1.item()
        self.l2 = 0.1*self.l2 + 0.9*l2.item()
        return l1 + self.lmd * l2

