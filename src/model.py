import torch
import torch.nn as nn
from torchvision import models, transforms


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and hasattr(m, 'weight'):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1 and hasattr(m, 'weight'):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class UNetGenerator(nn.Module):
    def __init__(self, in_channels=1, out_channels=3, features=64): # Defaults, but will be set by config
        super().__init__()
        # Encoder
        self.enc1 = self.conv_block(in_channels, features, bn=False, act_fn=nn.LeakyReLU(0.2, inplace=True))
        self.enc2 = self.conv_block(features, features * 2, act_fn=nn.LeakyReLU(0.2, inplace=True))
        self.enc3 = self.conv_block(features * 2, features * 4, act_fn=nn.LeakyReLU(0.2, inplace=True))
        self.enc4 = self.conv_block(features * 4, features * 8, act_fn=nn.LeakyReLU(0.2, inplace=True))
        self.enc5 = self.conv_block(features * 8, features * 8, act_fn=nn.LeakyReLU(0.2, inplace=True))
        self.enc6 = self.conv_block(features * 8, features * 8, act_fn=nn.LeakyReLU(0.2, inplace=True))
        self.enc7 = self.conv_block(features * 8, features * 8, act_fn=nn.LeakyReLU(0.2, inplace=True))
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features * 8, features * 8, 4, 2, 1, bias=False), # bias=False if no BN
            nn.LeakyReLU(0.2, inplace=True)
        )
        # Decoder
        self.dec1 = self.deconv_block(features * 8, features * 8, drop=True) # Input: features * 8 (from bottleneck)
        self.dec2 = self.deconv_block(features * 8 * 2, features * 8, drop=True) # Input: features * 8 (dec1) + features * 8 (e7)
        self.dec3 = self.deconv_block(features * 8 * 2, features * 8, drop=True) # Input: features * 8 (dec2) + features * 8 (e6)
        self.dec4 = self.deconv_block(features * 8 * 2, features * 8)            # Input: features * 8 (dec3) + features * 8 (e5)
        self.dec5 = self.deconv_block(features * 8 * 2, features * 4)            # Input: features * 8 (dec4) + features * 8 (e4)
        self.dec6 = self.deconv_block(features * 4 * 2, features * 2)            # Input: features * 4 (dec5) + features * 4 (e3)
        self.dec7 = self.deconv_block(features * 2 * 2, features)                # Input: features * 2 (dec6) + features * 2 (e2)
        
        self.final_conv = nn.Sequential(
            nn.ConvTranspose2d(features * 2, out_channels, 4, 2, 1), # Input: features (dec7) + features (e1)
            nn.Tanh()
        )

    def conv_block(self, ic, oc, bn=True, act_fn=nn.ReLU(True), ks=4, s=2, p=1):
        layers = [nn.Conv2d(ic, oc, ks, s, p, bias=not bn)]
        if bn: layers.append(nn.BatchNorm2d(oc))
        layers.append(act_fn)
        return nn.Sequential(*layers)

    def deconv_block(self, ic, oc, bn=True, drop=False, act_fn=nn.ReLU(True), ks=4, s=2, p=1):
        layers = [nn.ConvTranspose2d(ic, oc, ks, s, p, bias=not bn)]
        if bn: layers.append(nn.BatchNorm2d(oc))
        if drop: layers.append(nn.Dropout(0.5))
        layers.append(act_fn)
        return nn.Sequential(*layers)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        e6 = self.enc6(e5)
        e7 = self.enc7(e6)
        b = self.bottleneck(e7)
        d1 = torch.cat([self.dec1(b), e7], 1)
        d2 = torch.cat([self.dec2(d1), e6], 1)
        d3 = torch.cat([self.dec3(d2), e5], 1)
        d4 = torch.cat([self.dec4(d3), e4], 1)
        d5 = torch.cat([self.dec5(d4), e3], 1)
        d6 = torch.cat([self.dec6(d5), e2], 1)
        d7 = torch.cat([self.dec7(d6), e1], 1)
        return self.final_conv(d7)

class Discriminator(nn.Module):
    def __init__(self, in_channels_x=1, in_channels_y=3, features=64): # Defaults, but will be set by config
        super().__init__()
        total_in_channels = in_channels_x + in_channels_y
        self.model = nn.Sequential(
            nn.Conv2d(total_in_channels, features, 4, 2, 1, bias=False), # PatchGAN first layer, no BN
            nn.LeakyReLU(0.2, inplace=True),
            self.disc_block(features, features*2),     # Output: features*2 x 64x64
            self.disc_block(features*2, features*4),   # Output: features*4 x 32x32
            self.disc_block(features*4, features*8, s=1), # Stride 1 for last conv before output (PatchGAN)
                                                        # Output: features*8 x 31x31 (if ks=4,s=1,p=1) -> check padding for PatchGAN output size
            nn.Conv2d(features*8, 1, 4, 1, 1, bias=False) # Output: 1 x 30x30 (PatchGAN output)
            # Sigmoid is applied via BCEWithLogitsLoss
        )

    def disc_block(self, ic, oc, bn=True, ks=4, s=2, p=1):
        layers = [nn.Conv2d(ic, oc, ks, s, p, bias=not bn)]
        if bn: layers.append(nn.BatchNorm2d(oc))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x, y):
        xy = torch.cat([x, y], dim=1)
        return self.model(xy)

class VGGContentLoss(nn.Module):
    def __init__(self, feature_layer_index=16, device="cpu"): # Default feature_layer_index, device from config
        super().__init__()
        try:
            vgg_weights = models.VGG16_Weights.IMAGENET1K_V1
        except AttributeError: # Older torchvision
            vgg_weights = True 
            print("Warning: torchvision.models.VGG16_Weights.IMAGENET1K_V1 not found. Using default pretrained=True for VGG16.")
            
        self.vgg = models.vgg16(weights=vgg_weights).features[:feature_layer_index + 1].to(device)
        # VGG uses specific normalization. Input images to VGGContentLoss are expected to be in [-1, 1] range.
        # We first convert them to [0,1] and then apply VGG's normalization.
        self.transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        for param in self.vgg.parameters():
            param.requires_grad = False
        self.device = device

    def forward(self, x):
        if x.shape[1] == 1:  # If grayscale (e.g. S1 image)
            x = x.repeat(1, 3, 1, 1) # Convert to 3 channels for VGG
        
        # Denormalize from model's output range [-1, 1] to [0, 1] for VGG input
        x_denorm = (x + 1) / 2.0
        x_transformed = self.transform(x_denorm.to(self.device)) # Ensure tensor is on correct device
        return self.vgg(x_transformed)
