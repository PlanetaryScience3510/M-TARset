import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Softmax

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def INF(B, H, W,device):
    '''
    生成(B*W,H,H)大小的对角线为inf的三维矩阵
    Parameters
    ----------
    B: batch
    H: height
    W: width
    '''
    return -torch.diag(torch.tensor(float("inf"), device=device).repeat(H), 0).unsqueeze(0).repeat(B * W, 1, 1)


class CC_module(nn.Module):

    def __init__(self, in_dim):
        '''
        Parameters
        ----------
        in_dim : int
            channels of input
        '''
        super(CC_module, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = Softmax(dim=3)
        self.INF = INF
        self.gamma = nn.Parameter(torch.zeros(1))
        #self.device = device



    def forward(self, x):
        m_batchsize, _, height, width = x.size()

        proj_query = self.query_conv(x)  # size = (b,c2,h,w), c1 = in_dim, c2 = c1 // 8

        # size = (b*w, h, c2)
        proj_query_H = proj_query.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height).permute(0, 2,
                                                                                                                 1)

        # size = (b*h, w, c2)
        proj_query_W = proj_query.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width).permute(0, 2,
                                                                                                                 1)

        proj_key = self.key_conv(x)  # size = (b,c2,h,w)

        proj_key_H = proj_key.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_key_W = proj_key.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1,
                                                                    width)  # size = (b*w,c2,h)

        proj_value = self.value_conv(x)  # size = (b,c1,h,w)
        proj_value_H = proj_value.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1,
                                                                        height)  # size = (b*w,c1,h)
        proj_value_W = proj_value.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1,
                                                                        width)  # size = (b*h,c1,w)

        # size = (b*w, h,h) ,其中[:,i,j]表示Q所有W的第Hi行的所有通道值与K上所有W的第Hj列的所有通道值的向量乘积
        energy_H = torch.bmm(proj_query_H, proj_key_H)

        # size = (b,h,w,h) #这里为什么加 INF并没有理解
        energy_H = (energy_H + self.INF(m_batchsize, height, width, x.device)).view(m_batchsize, width, height, height).permute(0,
                                                                                                                      2,
                                                                                                                      1,
                                                                                                                      3)

        # size = (b*h,w,w),其中[:,i,j]表示Q所有H的第Wi行的所有通道值与K上所有H的第Wj列的所有通道值的向量乘积
        energy_W = torch.bmm(proj_query_W, proj_key_W)
        energy_W = energy_W.view(m_batchsize, height, width, width)  # size = (b,h,w,w)

        concate = self.softmax(torch.cat([energy_H, energy_W], 3))  # size = (b,h,w,h+w) #softmax归一化
        # concate = concate * (concate>torch.mean(concate,dim=3,keepdim=True)).float()
        att_H = concate[:, :, :, 0:height].permute(0, 2, 1, 3).contiguous().view(m_batchsize * width, height,
                                                                                 height)  # size = (b*w,h,h)
        # print(concate)
        # print(att_H)
        att_W = concate[:, :, :, height:height + width].contiguous().view(m_batchsize * height, width,
                                                                          width)  # size = (b*h,w,w)

        # size = (b*w,c1,h) #[:,i,j]表示V所有W的第Ci行通道上的所有H 与att_H的所有W的第Hj列的h权重的乘积
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1))
        out_H = out_H.view(m_batchsize, width, -1, height).permute(0, 2, 3, 1)  # size = (b,c1,h,w)

        # size = (b*h,c1,w) #[:,i,j]表示V所有H的第Ci行通道上的所有W 与att_W的所有H的第Wj列的W权重的乘积
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1))
        out_W = out_W.view(m_batchsize, height, -1, width).permute(0, 2, 1, 3)  # size = (b,c1,h,w)
        # print(out_H.size(),out_W.size())

        return self.gamma * (out_H + out_W) + x


# if __name__ == '__main__':
#     # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#     device = torch.device('cuda:0' if torch.cuda.device_count() > 1 else 'cpu')
# model = CC_module(8, device)
# x = torch.randn(4, 8, 20, 20).to(device)
# out = model(x).cuda()
# print(out.shape)
# # ————————————————
# 版权声明：本文为CSDN博主「fangzuliang」的原创文章，遵循CC
# 4.0
# BY - SA版权协议，转载请附上原文出处链接及本声明。
# 原文链接：https: // blog.csdn.net / weixin_43718675 / article / details / 106760382