import os
from PIL import Image
import torch
from torchvision import transforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from prettytable import PrettyTable


def count_params(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

def pull_data(path):
    pixels = []
    labels = []
    subdirs = os.listdir(path)
    for subdir in subdirs:
        print(f"Loading {subdir}...")
        dir = os.path.join(path, subdir)
        baybayin_char = os.listdir(dir)
        for char_img in baybayin_char:
            char_img_path = os.path.join(dir, char_img)
            img = Image.open(char_img_path)
            img = img.convert("L")
            img = transforms.ToTensor()(img)
            img = transforms.Resize((28, 28), antialias=True)(img)
            img = 1 - img
            img = img.view(-1, 28*28)
            pixels.append(img)
            labels.append(str(subdir))
        break

    pixels = torch.stack(pixels) ## Make a tensor
    return pixels, labels

def show(tensor, ch, size=(28, 28), num=16):
    data = tensor.detach().cpu().view(-1, ch, *size)
    data = data * 255.0
    grid = make_grid(data[:num], nrow=4).permute(1, 2, 0)
    plt.imshow(np.uint8(grid.numpy()))
    plt.show()

def gen_noise(number, z_dim, device):
    return torch.randn(number, z_dim, device=device)

def calc_gen_loss(loss_fn, gen, disc, num, z_dim, device):
    noise = gen_noise(num, z_dim, device)
    fake = gen(noise)
    pred = disc(fake)
    real_targets = torch.ones_like(pred)
    loss = loss_fn(pred, real_targets)
    return loss

def calc_disc_loss(loss_fn, gen, disc, real, num, z_dim, device):
    noise = gen_noise(num, z_dim, device)
    fake = gen(noise)
    pred_fake = disc(fake.detach())
    fake_targets = torch.zeros_like(pred_fake)
    loss_fake = loss_fn(pred_fake, fake_targets)
    
    pred_real = disc(real)
    real_targets = torch.ones_like(pred_real)
    loss_real = loss_fn(pred_real, real_targets)

    loss = torch.mean(loss_fake + loss_real)
    return loss

def engine(num_epochs, dataloader, gen, gen_opt, gen_sch, z_dim, disc, disc_opt, disc_sch, loss_fn, info_step, device):
    cur_step = 0
    mean_disc_loss = 0
    mean_gen_loss = 0
    loop = tqdm(range(num_epochs), leave=True)
    for epoch in loop:
        for real, _ in dataloader:
            ## Discriminator
            cur_batchsize = len(real)
            real = real.view(cur_batchsize, -1).to(device)
            disc_loss = calc_disc_loss(loss_fn, gen, disc, real, cur_batchsize, z_dim, device)
            disc_opt.zero_grad()
            disc_loss.backward(retain_graph=True)
            disc_opt.step()

            ## Generator
            gen_loss = calc_gen_loss(loss_fn, gen, disc, cur_batchsize, z_dim, device)
            gen_opt.zero_grad()
            gen_loss.backward()
            gen_opt.step()

            mean_gen_loss += gen_loss.data.item() / info_step
            mean_disc_loss += disc_loss.data.item() / info_step

            if (cur_step % info_step == 0) and (cur_step > 0):
                loop.set_description(f"Epoch [{epoch}/{num_epochs}]")
                loop.set_postfix(gen_loss=mean_gen_loss, disc_loss=mean_disc_loss,
                                 gen_lr=gen_sch.get_last_lr()[0], disc_lr=disc_sch.get_last_lr()[0])
                noise = gen_noise(cur_batchsize, z_dim, device)
                fake = gen(noise)
                show(fake, 1)
                show(real, 1)
                mean_gen_loss = 0
                mean_disc_loss = 0
            cur_step += 1
        gen_sch.step()
        disc_sch.step()

def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.normal_(m.weight, 0.0, 0.25)
        torch.nn.init.constant_(m.bias, 0)
    if isinstance(m, torch.nn.BatchNorm1d):
        torch.nn.init.normal_(m.weight, 0.0, 0.25)
        torch.nn.init.constant_(m.bias, 0)