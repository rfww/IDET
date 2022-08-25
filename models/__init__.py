from .fcn import FCN8, FCN16, FCN32
from .erfnet import ERFNet
from .utils import *
from .eNet import eNet
from .idet import feature_extractor
from .DSIFN import DSIFN
net_dic = {'dsifn':DSIFN, 'idet': feature_extractor}
def get_model(args):
    Net = net_dic[args.model]
    model = Net(args.num_classes)
    # model.apply(weights_init)
    return model
