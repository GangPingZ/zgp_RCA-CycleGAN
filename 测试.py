import torch
from models.networks import get_norm_layer
import torch.nn as nn
import numpy as np
from skimage import color
import functools
torch.cuda.manual_seed(0)
torch.manual_seed(0)


if __name__ == "__main__":
    def lab2rgb(L, AB):
        """Convert an Lab tensor image to a RGB numpy output
        Parameters:
            L  (1-channel tensor array): L channel images (range: [-1, 1], torch tensor array)
            AB (2-channel tensor array):  ab channel images (range: [-1, 1], torch tensor array)

        Returns:
            rgb (RGB numpy image): rgb output images  (range: [0, 255], numpy array)
        """
        AB2 = AB * 110.0
        L2 = (L + 1.0) * 50.0
        print(AB2.shape, L2.shape)
        Lab = torch.cat([L2, AB2], dim=1)
        Lab = Lab[0].data.cpu().float().numpy()
        Lab = np.transpose(Lab.astype(np.float64), (1, 2, 0))
        rgb = color.lab2rgb(Lab) * 255
        return rgb
    a1 = torch.randint(low=-128, high=127, size=(1, 2, 256, 256), dtype=torch.float32)
    a2 = torch.randint(low=0, high=100, size=(1, 1, 256, 256), dtype=torch.float32)
    c = torch.cat((a1, a2), dim=1)
    print(c[:, 0:2, :, :].shape, torch.unsqueeze(c[:, 2, :, :], dim=0)==a2)
    # print(a1.shape, a2.shape, c.shape, e.shape)
    # norm_layer = get_norm_layer(norm_type='instance')
    # model = [nn.ReflectionPad2d(3), nn.Conv2d(3, 64, kernel_size=7, padding=0, bias=False),
    #         norm_layer(64, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)]
    # b = nn.Sequential(*model)(c)
    # print(b)
    # rgb = lab2rgb(torch.unsqueeze(c[:, 2, :, :], dim=0), c[:, 0:2, :, :])
    # print(b, rgb)

