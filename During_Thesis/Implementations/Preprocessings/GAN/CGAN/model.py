import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, channels_img, features_d , num_classes , img_size ):
        
        super(Discriminator, self).__init__()
        self.img_size = img_size
        self.disc = nn.Sequential(
            # input: N x channels_img x 256 x 256
            nn.Conv2d(channels_img+1, features_d, kernel_size=4, stride=2, padding=1),  # 128x128
            nn.LeakyReLU(0.2),
            self._block(features_d, features_d * 2, 4, 2, 1),  # 64x64
            self._block(features_d * 2, features_d * 4, 4, 2, 1),  # 32x32
            self._block(features_d * 4, features_d * 8, 4, 2, 1),  # 16x16
            self._block(features_d * 8, features_d * 16, 4, 2, 1),  # 8x8
            self._block(features_d * 16, features_d * 32, 4, 2, 1),  # 4x4
            nn.Conv2d(features_d * 32, 1, kernel_size=4, stride=2, padding=0),  # 1x1
            
        )
        self.embed = nn.Embedding(num_classes, img_size*img_size)

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.InstanceNorm2d(out_channels , affine=True),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x , labels):
        embedding = self.embed(labels).view(labels.shape[0],1, self.img_size , self.img_size)
        x = torch.cat([x,embedding] , dim=1)
        return self.disc(x)


class Generator(nn.Module):
    def __init__(self, channels_noise, channels_img, features_g  , num_classes , img_size , embed_size):
        super(Generator, self).__init__()
        self.img_size = img_size
        self.num_classes = num_classes
        self.embed_size = embed_size
        self.net = nn.Sequential(
            # Input: N x channels_noise x 1 x 1
            self._block(channels_noise + embed_size, features_g * 16, 4, 1, 0),  # 4x4
            self._block(features_g * 16, features_g * 8, 4, 2, 1),  # 8x8
            self._block(features_g * 8, features_g * 4, 4, 2, 1),  # 16x16
            self._block(features_g * 4, features_g * 2, 4, 2, 1),  # 32x32
            self._block(features_g * 2, features_g, 4, 2, 1),  # 64x64
            nn.ConvTranspose2d(features_g, channels_img, kernel_size=4, stride=2, padding=1),  # 128x128
            nn.ConvTranspose2d(channels_img, channels_img, kernel_size=4, stride=2, padding=1),  # 256x256
            nn.Tanh(),
        )
        
        self.embed = nn.Embedding( num_classes , embed_size  )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x , labels):
        embedding = self.embed(labels).unsqueeze(2).unsqueeze(3)
        x = torch.cat([x,embedding] , dim=1)
        return self.net(x)


def initialize_weights(model):
    # Initializes weights according to the DCGAN paper
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)


def test():
    N, in_channels, H, W = 8, 3, 64, 64
    noise_dim = 100
    x = torch.randn((N, in_channels, H, W))
    disc = Discriminator(in_channels, 8)
    assert disc(x).shape == (N, 1, 1, 1), "Discriminator test failed"
    gen = Generator(noise_dim, in_channels, 8)
    z = torch.randn((N, noise_dim, 1, 1))
    assert gen(z).shape == (N, in_channels, H, W), "Generator test failed"
    print("Success, tests passed!")


if __name__ == "__main__":
    test()