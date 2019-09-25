from __future__ import absolute_import    #使用绝对引用，这样当自定的模块和python自带的模块重名时不会混淆

import torch.nn as nn
import torch.nn.functional as F


__all__ = ['SiamFC']    #在默认情况下，如果使用“from 模块名 import *”这样的语句来导入模块，程序会导入该模块中所有函数
                        #但在一些场景中，我们并不希望每个成员都被暴露出来供外界使用，此时可借助于模块的 __all__ 变量，
                        #将变量的值设置成一个列表，只有该列表中的成员才会被暴露出来。


class SiamFC(nn.Module):

    def __init__(self, out_scale=0.001):
        super(SiamFC, self).__init__()
        self.out_scale = out_scale
    
    def forward(self, z, x):
        return self._fast_xcorr(z, x) * self.out_scale  #xcorr表示互相关，定义一个前向传播函数。输出z和x互相关、再乘上out_scale的值
    
    def _fast_xcorr(self, z, x):   #定义互相关函数
        # fast cross correlation
        nz = z.size(0)
        nx, c, h, w = x.size()
        x = x.view(-1, nz * c, h, w)
        out = F.conv2d(x, z, groups=nz)     #卷积操作，输入为x，卷积核为z，group=2表示把输入分为两组做卷积
        out = out.view(nx, -1, out.size(-2), out.size(-1))
        return out
