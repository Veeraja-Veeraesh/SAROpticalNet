import torch
import torch.nn as nn
from torchvision import models, transforms

# Constants (can be moved to config.yaml later)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHANNELS_IMG_GRAY = 1
CHANNELS_IMG_COLOR = 3
VGG_CONTENT_LAYERS_INDICES = {'block3_conv3': 16, 'block4_conv3': 23} # Example
CHOSEN_VGG_LAYER_INDEX = VGG_CONTENT_LAYERS_INDICES.get('block3_conv3', 16) # Default if not found

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and hasattr(m, 'weight'):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1 and hasattr(m, 'weight'):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class UNetGenerator(nn.Module):
    def __init__(self, in_channels=CHANNELS_IMG_GRAY, out_channels=CHANNELS_IMG_COLOR, features=64):
        super().__init__()
        self.enc1 = self.conv_block(in_channels, features, bn=False, act_fn=nn.LeakyReLU(0.2, inplace=True))
        self.enc2 = self.conv_block(features, features * 2, act_fn=nn.LeakyReLU(0.2, inplace=True))
        self.enc3 = self.conv_block(features * 2, features * 4, act_fn=nn.LeakyReLU(0.2, inplace=True))
        self.enc4 = self.conv_block(features * 4, features * 8, act_fn=nn.LeakyReLU(0.2, inplace=True))
        self.enc5 = self.conv_block(features * 8, features * 8, act_fn=nn.LeakyReLU(0.2, inplace=True))
        self.enc6 = self.conv_block(features * 8, features * 8, act_fn=nn.LeakyReLU(0.2, inplace=True))
        self.enc7 = self.conv_block(features * 8, features * 8, act_fn=nn.LeakyReLU(0.2, inplace=True))
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features * 8, features * 8, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True)
        )
        self.dec1 = self.deconv_block(features * 8, features * 8, drop=True)
        self.dec2 = self.deconv_block(features * 8 * 2, features * 8, drop=True)
        self.dec3 = self.deconv_block(features * 8 * 2, features * 8, drop=True)
        self.dec4 = self.deconv_block(features * 8 * 2, features * 8)
        self.dec5 = self.deconv_block(features * 8 * 2, features * 4)
        self.dec6 = self.deconv_block(features * 4 * 2, features * 2)
        self.dec7 = self.deconv_block(features * 2 * 2, features)
        self.final_conv = nn.Sequential(
            nn.ConvTranspose2d(features * 2, out_channels, 4, 2, 1), nn.Tanh()
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
        e1 = self.enc1(x); e2 = self.enc2(e1); e3 = self.enc3(e2); e4 = self.enc4(e3)
        e5 = self.enc5(e4); e6 = self.enc6(e5); e7 = self.enc7(e6)
        b = self.bottleneck(e7)
        d1 = torch.cat([self.dec1(b), e7], 1); d2 = torch.cat([self.dec2(d1), e6], 1)
        d3 = torch.cat([self.dec3(d2), e5], 1); d4 = torch.cat([self.dec4(d3), e4], 1)
        d5 = torch.cat([self.dec5(d4), e3], 1); d6 = torch.cat([self.dec6(d5), e2], 1)
        d7 = torch.cat([self.dec7(d6), e1], 1)
        return self.final_conv(d7)

class Discriminator(nn.Module):
    def __init__(self, in_channels_x=CHANNELS_IMG_GRAY, in_channels_y=CHANNELS_IMG_COLOR, features=64):
        super().__init__()
        total_in_channels = in_channels_x + in_channels_y
        self.model = nn.Sequential(
            nn.Conv2d(total_in_channels, features, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True),
            self.disc_block(features, features*2),
            self.disc_block(features*2, features*4),
            self.disc_block(features*4, features*8, s=1),
            nn.Conv2d(features*8, 1, 4, 1, 1)
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
    def __init__(self, feature_layer_index=CHOSEN_VGG_LAYER_INDEX, device=DEVICE):
        super().__init__()
        # Ensure VGG16_Weights.IMAGENET1K_V1 is available or adjust if needed
        try:
            vgg_weights = models.VGG16_Weights.IMAGENET1K_V1
        except AttributeError: # Older torchvision
            vgg_weights = True # This will load pretrained weights
            print("Warning: torchvision.models.VGG16_Weights.IMAGENET1K_V1 not found. Using default pretrained=True.")


        self.vgg = models.vgg16(weights=vgg_weights).features[:feature_layer_index + 1].to(device)
        self.transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, x):
        if x.shape[1] == 1:  # If grayscale
            x = x.repeat(1, 3, 1, 1) # Convert to 3 channels for VGG
        # Denormalize from [-1, 1] to [0, 1] before VGG normalization
        x_denorm = (x + 1) / 2.0
        x_transformed = self.transform(x_denorm)
        return self.vgg(x_transformed)