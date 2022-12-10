from dataset import CustomDataset
from VGG import VGG
from model_our import Generator, Discriminator

from torch.utils.data import DataLoader
from torchvision import transforms, utils
from torchvision.models import vgg19
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
import argparse
import os
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description="colorization trainer")
parser.add_argument("--batch_size", default=16, type=int, help="batch size")
parser.add_argument("--iters", default=600000, type=int, help="maximum iterations")

log_dir = './logs_our/'
checkpoint_dir = './train_our/checkpoint/'
sample_dir = './train_our/sample/'

if not os.path.exists(log_dir):
    os.makedirs(log_dir)
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)


writer = SummaryWriter(log_dir)

def sample_data(sketch_dir, reference_dir, batch_size):

    s_transform = transforms.Compose(
        [   
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,))
        ]
    )
    
    r_transform = transforms.Compose(
        [   
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ]
    )

    dataset = CustomDataset(sketch_dir=sketch_dir, reference_dir=reference_dir, s_transform=s_transform, r_transform=r_transform)
    loader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=8, drop_last=True)
    loader = iter(loader)

    while True:
        try:
            yield next(loader)

        except StopIteration:
            loader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=8, drop_last=True)
            loader = iter(loader)
            yield next(loader)


def gram(x):
    b, c, h, w = x.shape
    x_tmp = x.reshape((b, c, (h * w)))
    # x_tmp = x.view(b, c, h*w)
    gram = torch.matmul(x_tmp, x_tmp.permute(0, 2, 1))
    return gram / (c * h * w)


def style_loss(fake, style):
    gram_loss = nn.L1Loss()(gram(fake), gram(style))
    return gram_loss


vgg_model = vgg19(pretrained=True)
vgg_model.to(device)

adv_loss = nn.MSELoss()
l1_loss = nn.L1Loss()


def train(opt, gen, dis, optim_g, optim_d):
    gen.train()
    dis.train()
    dataset = iter(sample_data(sketch_dir="./kaggle_data/sketch/", reference_dir="./kaggle_data/reference/", batch_size=opt.batch_size))
    with tqdm(range(opt.iters)) as pbar:
        for i in pbar:
            sketch, reference_tps, reference = next(dataset)
            # sketch = torch.max(sketch, dim=1, keepdim=True)[0]
            sketch = sketch.to(device)
            reference_tps = reference_tps.to(device)
            reference = reference.to(device)

            # 训练生成器
            fake_img_gen = gen(sketch, reference_tps)
            fake_output = dis(torch.cat([fake_img_gen, sketch], dim=1))
            g_adv_loss = adv_loss(fake_output, torch.ones_like(fake_output))
            g_l1_loss = l1_loss(fake_img_gen, reference) * 30
            g_vgg_loss = torch.tensor(0.).to(device)
            g_style_loss = torch.tensor(0.).to(device)
            # rates = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
            rates = [1.0, 1.0, 1.0, 1.0, 1.0]
            fake_features = VGG(vgg_model, fake_img_gen)
            real_features = VGG(vgg_model, reference)

            for j in range(len(fake_features)):
                a, b = fake_features[j], real_features[j]

                g_vgg_loss += rates[j] * l1_loss(a, b)
                g_style_loss += rates[j] * style_loss(a, b)

            g_vgg_loss *= 0.01
            g_style_loss *= 50

            g_loss = g_l1_loss + g_adv_loss + g_style_loss + g_vgg_loss
            writer.add_scalar("g_l1_loss", g_l1_loss.item(), global_step=i)
            writer.add_scalar("g_adv_loss", g_adv_loss.item(), global_step=i)
            writer.add_scalar("g_style_loss", g_style_loss.item(), global_step=i)
            writer.add_scalar("g_vgg_loss", g_vgg_loss.item(), global_step=i)
            writer.add_scalar("g_loss", g_loss.item(), global_step=i)


            optim_g.zero_grad()
            g_loss.backward()
            optim_g.step()
            # print("g_lr: ", optim_g.param_groups[0]["lr"])

            # 训练判别器
            fake_output = dis(torch.cat([fake_img_gen.detach(), sketch], dim=1))
            real_output = dis(torch.cat([reference, sketch], dim=1))
            d_fake_loss = adv_loss(fake_output, torch.zeros_like(fake_output))
            d_real_loss = adv_loss(real_output, torch.ones_like(real_output))
            d_loss = d_real_loss + d_fake_loss
            writer.add_scalar("d_real_loss", d_real_loss.item(), global_step=i)
            writer.add_scalar("d_fake_loss", d_fake_loss.item(), global_step=i)
            writer.add_scalar("d_loss", d_loss.item(), global_step=i)

            optim_d.zero_grad()
            d_loss.backward()
            optim_d.step()
            # print("d_lr: ", optim_d.param_groups[0]["lr"])

            pbar.set_description(f"g_loss: {g_loss.item():.5f}; d_loss: {d_loss.item():.5f}")

            if i % 100 == 0:
                utils.save_image(
                    fake_img_gen.cpu().data,
                    sample_dir + f"gen_{str(i+1)}.png",
                    normalize=True,
                    nrow=4,
                    range=(-1, 1),
                )
                utils.save_image(
                    sketch.cpu().data,
                    sample_dir + f"sketch_{str(i+1)}.png",
                    normalize=True,
                    nrow=4,
                    range=(-1, 1),
                )
                utils.save_image(
                    reference_tps.cpu().data,
                    sample_dir + f"reference_tps_{str(i+1)}.png",
                    normalize=True,
                    nrow=4,
                    range=(-1, 1),
                )
                utils.save_image(
                    reference.cpu().data,
                    sample_dir + f"reference_{str(i+1)}.png",
                    normalize=True,
                    nrow=4,
                    range=(-1, 1),
                )

            if i % 5000 == 0:
                torch.save({
                    "gen": gen.state_dict(),
                    "optim_g": optim_g.state_dict(),
                    "dis": dis.state_dict(),
                    "optim_d": optim_d.state_dict(),
                    "iter_num": i,
                    }, checkpoint_dir + f"ckpt_{str(i+1)}.pth")


if __name__ == "__main__":
    opt = parser.parse_args()
    generator = Generator(sketch_channels=1, reference_channels=3)
    generator.to(device)
    discriminator = Discriminator(ndf=16, in_channels=4)
    discriminator.to(device)

    optimizer_G = torch.optim.Adam(params=generator.parameters(), lr=1e-4, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(params=discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))
  
    train(opt, gen=generator, dis=discriminator, optim_g=optimizer_G, optim_d=optimizer_D)
