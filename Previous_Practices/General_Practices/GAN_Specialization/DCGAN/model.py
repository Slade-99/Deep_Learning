## DCGAN


import torch
import torch.nn as nn



class Discriminator(nn.Module):
    def __init__(self, img_channels , filters):
        super().__init__()
        ## Input tensors  = 64x64x3
        self.disc = nn.Sequential(
            nn.Conv2d(img_channels,filters, kernel_size=4,stride=2,padding=1
            ),
            nn.LeakyReLU(0.2),
            self.block(filters,filters*2 , 4 , 2 ,1 ),  ## Halves spatial dim
            self.block(filters*2,filters*4 , 4 , 2 ,1 ),  ## 8x8
            self.block(filters*4,filters*8 , 4 , 2 ,1 ), ## 4x4
            nn.Conv2d(filters*8,1,kernel_size = 4, stride=2, padding =0), ## 1x1
            nn.Sigmoid(),

        )



    def block(self, in_channels , out_channels , kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(

                      in_channels,
                      out_channels,
                      kernel_size,
                      stride,
                      padding,
                      bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self,x):
        return self.disc(x)





class Generator(nn.Module):

    def __init__(self, noise_dim , img_channels , filters):

        super().__init__()
        ## Input: noise_dim x 1 x 1
        self.gen = nn.Sequential(

            self.block(noise_dim, filters*16 , 4, 1, 0 ), # 4x4
            self.block(filters*16, filters*8 , 4, 2, 1 ), # 8x8
            self.block(filters*8, filters*4 , 4, 2, 1 ), # 16x16
            self.block(filters*4, filters*2 , 4, 2, 1 ), # 32x32
            nn.ConvTranspose2d(
                filters*2, img_channels , 4 , 2 , 1
            ),
            nn.Tanh(),

        )


    def block(self,in_channels,out_channels,kernel_size,stride,padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),


        )


    def forward(self,x):
        return self.gen(x)





def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d,nn.ConvTranspose2d,nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0 , 0.02)



def test():

    N, in_channels ,H , W = 32 , 3 , 64 , 64
    noise_dim = 100

    ## Discriminator
    x = torch.randn((N,in_channels,H,W))
    disc = Discriminator(in_channels,8)
    initialize_weights(disc)
    assert disc(x).shape == (N,1,1,1)

    ## Generator
    z = torch.randn((N,noise_dim,1,1))
    gen = Generator(noise_dim,in_channels,8)
    assert gen(z).shape == (N,3,64,64)
    print("Architecture Works Successfully")




test()

