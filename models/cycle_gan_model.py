import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from torch.autograd import Variable
import random
from torchvision import transforms
from scipy import stats
import cv2
import numpy as np
from skimage import color  # used for lab2rgb
from . import vgg


class CycleGANModel(BaseModel):
    """
    This class implements the CycleGAN model, for learning image-to-image translation without paired data.

    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').

    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For CycleGAN, in addition to GAN losses, we introduce lambda_A, lambda_B, and lambda_identity for the following losses.
        A (source domain), B (target domain).
        Generators: G_A: A -> B; G_B: B -> A.
        Discriminators: D_A: G_A(A) vs. B; D_B: G_B(B) vs. A.
        Forward cycle loss:  lambda_A * ||G_B(G_A(A)) - A|| (Eqn. (2) in the paper)
        Backward cycle loss: lambda_B * ||G_A(G_B(B)) - B|| (Eqn. (2) in the paper)
        Identity loss (optional): lambda_identity * (||G_A(B) - B|| * lambda_B + ||G_B(A) - A|| * lambda_A) (Sec 5.2 "Photo generation from paintings" in the paper)
        Dropout is not used in the original CycleGAN paper.
        """
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:  # lambda_A(B) default=10.0, lambda_identity default=0.5
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=1.0, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')

        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call<BaseModel.get_current_losses>
        self.loss_names = ['G_A', 'G_B', 'D_A', 'D_B', 'cycle_A', 'cycle_B', 'idt_A', 'idt_B', 'DC', 'LAB', 'FE_A', 'FE_B']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['real_A', 'fake_B', 'rec_A']
        visual_names_B = ['real_B', 'fake_A', 'rec_B']
        #if self.isTrain and self.opt.lambda_identity > 0.0:  # if identity loss is used, we also visualize
        #    visual_names_A.append('idt_B')
        #    visual_names_B.append('idt_A')

        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G_R', 'G_L', 'G_B', 'D_A', 'D_B',]
        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'G_B']

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.vgg = vgg.Vgg19(requires_grad=False).to(self.device)
        self.netG_R = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, 'unet_128_A', opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_L = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, 'unet_128_B', opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, 'unet_128_B', opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # define discriminators
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert(opt.input_nc == opt.output_nc)
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            #z 1.用来计算对抗损失 DA DB
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            #z 2.用来计算循环一致损失 'cycle_A', 'cycle_B'
            self.criterionCycle = torch.nn.L1Loss()
            #z 3.用来计算对比损失  'FE'
            self.criterionVgg = networks.VGGLoss1(self.device, vgg=self.vgg, normalize=False)
            #z 4.用来计算身份映射损失 'idt_A', 'idt_B'
            self.criterionIdt = torch.nn.L1Loss()
            #z 5.用来计算红通道损失 'DC'
            self.criterionDC = self.DCLoss
            # 6.用来计算Lab损失 'LAB'
            self.criterionLAB = self.LABLoss
            self.criterionLBA = self.LBALoss

            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_R.parameters(), self.netG_L.parameters(), self.netG_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(
                itertools.chain(self.netD_A.parameters(), self.netD_B.parameters(),
                                ),
                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def randomCrop(self, img):
        # 设置裁剪后的长,宽
        # width = random.randint(128, img.shape[3])
        # height = random.randint(128, img.shape[2])
        width = 128
        height = 128

        # 设置裁剪的起点
        x = random.randint(0, img.shape[3] - width)
        y = random.randint(0, img.shape[2] - height)

        img = img[:, :, y:y + height, x:x + width]
        # print("img", img.shape)
        return img

    def DCLoss(self, img, patch_size=35):
        """
        calculating dark channel of image, the image shape is of N*C*W*H
        """
        rev = torch.clone(img)
        rev[:, 0, :, :] = 1 - rev[:, 0, :, :] #红通道反转
        red = rev.detach()
        maxpool = torch.nn.MaxPool3d((3, patch_size, patch_size), stride=1, padding=(0, patch_size // 2, patch_size // 2))
        dc = maxpool(0 - red[:, None, :, :, :])


        target = Variable(torch.FloatTensor(dc.shape).zero_().cuda(self.device))

        loss = torch.nn.L1Loss(reduction='mean')(-dc, target)

        return loss

    def LABLoss(self, img):
        """
        calculating dark channel of image, the image shape is of N*C*W*H
        """

        image = img.cpu().detach().numpy().squeeze()
        image = image.transpose([1, 2, 0])
        image = np.array((image * 255), dtype=np.uint8)
        img_LAB = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        img_LAB = np.array(img_LAB, dtype=np.float64)

        img_a = img_LAB[:, :, 1]
        img_b = img_LAB[:, :, 2]

        # chroma = np.sqrt(np.square(img_a) + np.square(img_b))
        # sigma_c = np.std(chroma)
        a = img_a.flatten()
        b = img_b.flatten()

        za = stats.mode(a)[0][0]
        zb = stats.mode(b)[0][0]

        # l1 = 0.0
        # l2 = 0.0
        target = 127.5
        # if za < target:
        #     l1 = target - za
        # if zb < target:
        #     l2 = target - zb

        sub1 = (target - za)
        if sub1 < 0:
            sub1 = 0
        sub2 = (target - zb)
        if sub2 < 0:
            sub2 = 0
        loss = ((2/(1 + np.exp(-0.1 * sub1)) - 1)**2 + (2/(1 + np.exp(-0.1 * sub2))-1)**2) * 0.5
        #loss = abs(sub1) + abs(sub2)

        # loss = 1 / sigma_c

        return torch.tensor(loss, device=self.device, requires_grad=True)


    def LBALoss(self, img):
        """
        calculating dark channel of image, the image shape is of N*C*W*H
        """

        image = img.cpu().detach().numpy().squeeze()
        image = image.transpose([1, 2, 0])
        image = np.array((image * 255), dtype=np.uint8)
        # image = np.array(image, dtype=np.uint8)  #
        img_LAB = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        img_LAB = np.array(img_LAB, dtype=np.float64)

        img_a = img_LAB[:, :, 1]
        img_b = img_LAB[:, :, 2]

        # chroma = np.sqrt(np.square(img_a) + np.square(img_b))
        # sigma_c = np.std(chroma)
        a = img_a.flatten()
        b = img_b.flatten()

        za = stats.mode(a)[0][0]

        # l1 = 0.0
        # l2 = 0.0
        target = 127.5
        # if za < target:
        #     l1 = target - za
        # if zb < target:
        #     l2 = target - zb

        sub1 = (target - za)
        if sub1 > 0:
            sub1 = 0

        loss = abs(2 / (1 + np.exp(-0.1 * sub1)) - 1)
        # loss = abs(sub1) + abs(sub2)

        # loss = 1 / sigma_c

        return torch.tensor(loss, device=self.device, requires_grad=True)

    # 原
    # def lab2rgb(self, L, AB):
    #     """Convert an Lab tensor image to a RGB numpy output
    #     Parameters:
    #         L  (1-channel tensor array): L channel images (range: [-1, 1], torch tensor array)
    #         AB (2-channel tensor array):  ab channel images (range: [-1, 1], torch tensor array)
    #
    #     Returns:
    #         rgb (RGB numpy image): rgb output images  (range: [0, 255], numpy array)
    #     """
    #     AB2 = AB * 110.0
    #     # L2 = to(L + 1.0) * 50.0
    #     L2 = torch.unsqueeze((L + 1.0) * 50.0, dim=0)
    #     Lab = torch.cat([L2, AB2], dim=1)
    #     Lab = Lab[0].data.cpu().float().numpy()
    #     Lab = np.transpose(Lab.astype(np.float64), (1, 2, 0))
    #     rgb = color.lab2rgb(Lab) * 255
    #     return rgb
    def lab2rgb(self, img):
        """Convert an Lab tensor image to a RGB numpy output
        Parameters:
            L  (1-channel tensor array): L channel images (range: [-1, 1], torch tensor array)
            AB (2-channel tensor array):  ab channel images (range: [-1, 1], torch tensor array)

        Returns:
            rgb (RGB numpy image): rgb output images  (range: [0, 255], numpy array)
        """
        AB2 = img[:, 1:3, :, :] * 110.0
        # L2 = to(L + 1.0) * 50.0
        L2 = torch.unsqueeze((img[:, 0, :, :] + 1.0) * 50.0, dim=0)
        Lab = torch.cat([L2, AB2], dim=1)
        Lab = Lab[0].data.cpu().numpy()
        Lab = np.transpose(Lab.astype(np.float64), (1, 2, 0))
        rgb = color.lab2rgb(Lab) * 255.0
        rgb = torch.unsqueeze(torch.tensor(np.transpose(Lab.astype(np.float64), (2, 0, 1)), device=self.device, dtype=torch.float), dim=0)
        return rgb

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG_R(self.real_A) + self.lab2rgb(self.netG_L(self.real_A))  # G_A(A)
        self.rec_A = self.netG_B(self.fake_B)   # G_B(G_A(A))
        self.fake_A = self.netG_B(self.real_B)  # G_B(B)
        self.rec_B = self.netG_R(self.fake_A) + self.lab2rgb(self.netG_L(self.fake_A))   # G_A(G_B(B))

    def tensor_to_PIL(self, tensor):
        unloader = transforms.ToPILImage()
        image = tensor.cpu().float().clone()
        image = image.squeeze(0)
        image = unloader(image)
        return image

    def PIL_to_tensor(self, pil):
        transform = transforms.Compose(
            [transforms.ToTensor(),  # 函数接受PIL Image或numpy.ndarray，将其先由HWC转置为CHW格式，再转为float后每个像素除以255.
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        return transform(pil)

    def unnormalize(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor*255

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """

        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

    # def backward_local_D_A(self):
    #     """Calculate GAN loss for discriminator D_A"""
    #     self.loss_local_D_A = 0
    #     for i in range(2):
    #         croped_real_B = self.randomCrop(self.real_B)
    #         croped_fake_B = self.randomCrop(self.fake_B)
    #         self.loss_local_D_A += self.backward_D_basic(self.netlocal_D_A, croped_real_B, croped_fake_B)
    #     # self.loss_local_D_A /= 2

    # def backward_local_D_B(self):
    #     """Calculate GAN loss for discriminator D_B"""
    #     self.loss_local_D_B = 0
    #     for i in range(2):
    #         croped_real_A = self.randomCrop(self.real_A)
    #         croped_fake_A = self.randomCrop(self.fake_A)
    #         self.loss_local_D_B += self.backward_D_basic(self.netlocal_D_B, croped_real_A, croped_fake_A)
    #     # self.loss_local_D_B /= 2

    def backward_A2B(self):
        """计算生成器 G_A 的 loss"""
        # self.backward_local_G_A()

        lambda_idt = -1 #self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B

        if lambda_idt > 0:
            self.idt_A = self.netG_R(self.real_B) + self.lab2rgb(self.netG_L(self.real_B))
            # self.idt_AA = self.netG_A(self.idt_A)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt #* 0.5
        else:
            self.loss_idt_A = 0

        # # ----------- 局部判别器损失 -----------
        # self.croped_fake_B = self.randomCrop(self.fake_B)
        # self.loss_local_G_A = 0
        # for i in range(1):
        #     croped_fake_B = self.randomCrop(self.fake_B)
        #     self.loss_local_G_A += self.criterionGAN(self.netlocal_D_A(croped_fake_B), True)
        # # self.loss_local_G_A /= 4
        self.loss_FE_A = self.criterionVgg(self.rec_A, self.real_A)
        self.loss_DC = self.criterionDC((self.fake_B + 1) / 2) * 1 #10
        self.loss_LAB = self.criterionLAB((self.fake_B + 1) / 2) * 1 #4
        # self.loss_TV_A = self.TVLoss(self.fake_B) * 1e-5
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        if not self.netG_L.requires_grad_():
            self.loss_A2B = self.loss_G_A + self.loss_cycle_A + self.loss_DC + self.loss_FE_A + self.loss_idt_A
        else:
            self.loss_A2B = self.loss_G_A + self.loss_cycle_A + self.loss_LAB + self.loss_FE_A + self.loss_idt_A #+ self.loss_idt_A+ self.loss_LAB  #+ self.loss_vggA
        self.loss_A2B.backward()


    def backward_B2A(self):
        """计算生成器 G_B 的 loss"""
        # self.backward_local_G_B()

        lambda_idt = -1 #self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B

        if lambda_idt > 0:
            self.idt_B = self.netG_B(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt #* 0.5
        else:
            self.loss_idt_B = 0

        # # ----------- 局部判别器损失 -----------
        # self.croped_fake_A = self.randomCrop(self.fake_A)
        # self.loss_local_G_B = 0
        # for i in range(1):
        #     croped_fake_A = self.randomCrop(self.fake_A)
        #     self.loss_local_G_B += self.criterionGAN(self.netlocal_D_B(croped_fake_A), True)
        # self.loss_local_G_B /= 4

        self.loss_FE_B = self.criterionVgg(self.rec_B, self.real_B) #3
        # self.loss_TV_B = self.TVLoss(self.fake_A) * 0.01
        # self.loss_LBA = self.criterionLAB((self.fake_A + 1) / 2)  #6
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B

        self.loss_B2A = self.loss_G_B + self.loss_cycle_B + self.loss_FE_B + self.loss_idt_B #+ self.loss_idt_B+ self.loss_LBA  #+ self.loss_vggB
        self.loss_B2A.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad([self.netD_A, self.netD_B, self.netG_L], False)  # Ds require no gradients when optimizing Gs
        # self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_A2B()             # calculate gradients for G_A and G_B
        self.backward_B2A()
        self.optimizer_G.step()       # update G_A and G_B's weights

        self.forward()
        self.set_requires_grad([self.netD_A, self.netD_B, self.netG_R],
                               False)  # Ds require no gradients when optimizing Gs
        # self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_A2B()  # calculate gradients for G_A and G_B
        self.backward_B2A()
        self.optimizer_G.step()  # update G_A and G_B's weights

        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        # self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D_A()      # calculate gradients for D_A
        self.backward_D_B()      # calculate graidents for D_B
        self.optimizer_D.step()  # update D_A and D_B's weights
