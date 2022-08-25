import torch
from torch.nn.modules import Module
from torch.nn import _reduction as _Reduction
#from torch._jit_internal import weak_module, weak_script_method
import torch.nn as nn
import torch.nn.functional as F

class _Loss(Module):
    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(_Loss, self).__init__()
        if size_average is not None or reduce is not None:
            self.reduction = _Reduction.legacy_get_string(size_average, reduce)
        else:
            self.reduction = reduction


class CrossEntropyLoss2d(nn.Module):
	def __init__(self, weight=None):
		super().__init__()
		self.loss = nn.NLLLoss2d(weight)

	def forward(self, outputs, targets):
        	# torch version >0.2 F.log_softmax(input, dim=?)
        	# dim (int): A dimension along which log_softmax will be computed.
		try:
			return self.loss(F.log_softmax(outputs, dim=1), targets)  # if torch version >=0.3
		except TypeError as t:
			return self.loss(F.log_softmax(outputs), targets)       # else

class CrossEntropyLoss2d_eNet(nn.Module):
	def __init__(self, weight=None):
		super().__init__()
		self.loss = nn.NLLLoss2d(weight)

	def forward(self, outputs, targets, outputs2):
        	#torch version >0.2 F.log_softmax(input, dim=?)
        	#dim (int): A dimension along which log_softmax will be computed.
		try:
			return self.loss(F.log_softmax(outputs, dim=1), targets) + 0.1*self.loss(F.log_softmax(outputs,dim=1), outputs2) # if torch version >=0.3
		except TypeError as t:
			return self.loss(F.log_softmax(outputs), targets) + 0.1*self.loss(F.log_softmax(outputs), outputs2)       #else



#@weak_module
class BCEWithLogitsLoss_eNet(_Loss):
	__constants__ = ['weight', 'pos_weight', 'reduction']

	def __init__(self, weight=None, size_average=None, reduce=None, reduction='mean', pos_weight=None):
		super(BCEWithLogitsLoss_eNet, self).__init__(size_average, reduce, reduction)
		self.register_buffer('weight', weight)
		self.register_buffer('pos_weight', pos_weight)

	#@weak_script_method
	def forward(self, input, target, input2):
		return F.binary_cross_entropy_with_logits(input, target, self.weight, pos_weight= self.pos_weight, reduction= self.reduction)# - 0.1*F.binary_cross_entropy_with_logits(input, input2, self.weight, pos_weight=self.pos_weight, reduction=self.reduction)


# iou loss

class IOULoss(nn.Module):
	def __init__(self) -> None:
		super().__init__()
	def forward(self, mask1, mask2):
		# iou
		# intersection = torch.sum(mask1*mask2)
		# area1 = torch.sum(mask1)
		# area2 = torch.sum(mask2)
		# union = (area1 + area2) - intersection
		# ret =  intersection / union
		# print("intersection: {}, union: {}".format(intersection, union))
		# return ret

		# (A * ~B)/(B + 1)
		intersection = torch.sum(mask1 * mask2)
		area1 = torch.sum(mask1)
		area2 = torch.sum(mask2)
		# union = (area1 + area2) - intersection
		ret = 1 - float(intersection+1) / float((area1+area2 - intersection + 1))
		# print("intersection: {}, union: {}".format(intersection, area))
		return ret
        
		# torch version >0.2 F.log_softmax(input, dim=?)
        # dim (int): A dimension along which log_softmax will be computed.
		# return 1 - ((sum(sum(outputs * targets))+1) / ((sum(sum(outputs+targets))-sum(sum(targets*outputs))+1)))
    	# intersection = torch.sum(mask1*mask2, dim=(0, 1))
        # area1 = torch.sum(mask1, dim=(0, 1))
        # area2 = torch.sum(mask2, dim=(0, 1))
        # union = (area1 + area2) - intersection
        # ret = intersection / union
        # return ret


class FLoss(nn.Module):
	def __init__(self) -> None:
		super().__init__()
	def forward(self, labels, targets, beta):
		intersecation = torch.sum(labels * targets)
		area1 = torch.sum(labels)
		area2 = torch.sum(targets)
		rec = intersecation * 0.9 / area2
		p = intersecation * 1. / area1
		f1 = ((beta * beta +1) * rec * p) / (beta * beta * p +rec)
		# sourceTensor.clone().detach().requires_grad_(True)
		return torch.sigmoid((1-f1)*(1-f1)).detach().requires_grad_(True)