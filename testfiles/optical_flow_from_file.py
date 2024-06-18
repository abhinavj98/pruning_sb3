

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from pruning_sb3.pruning_gym.optical_flow import OpticalFlow
import matplotlib.pyplot as plt


import numpy as np
import cv2
import torchvision.transforms.functional as F
import torch
from PIL import Image
from torchvision.utils import flow_to_image

def plot(imgs, **imshow_kwargs):
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0])
    _, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]
            img = F.to_pil_image(img.to("cpu"))
            ax.imshow(np.asarray(img), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    # plt.savefig("C:\\Users\\abhin\\OneDrive\\Pictures\\Screenshots\\of_7.png")
    # plt.tight_layout()
    plt.show()
def load_image(imfile):
    img = np.array(Image.open(imfile))
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None]/255.0

of = OpticalFlow()
image_folder = "C:\\Users\\abhin\\OneDrive\\Desktop\\jum1_images"
images = sorted([os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith(('.png', '.jpg', '.jpeg'))])
for i in range((len(images) - 1)//30):
    img_prev = load_image(images[i*30])
    img = load_image(images[(i+1)*30])
    # img = torch.tensor(img).unsqueeze(0)
    # img_prev = torch.tensor(img_prev).unsqueeze(0)
    # print(img.shape, img_prev.shape)
    flow = of.calculate_optical_flow(img, img_prev)

    flow_imgs = flow_to_image(flow)

    # The images have been mapped into [-1, 1] but for plotting we want them in [0, 1]
    # img1_batch = [img.squeeze(0)]
    # img2_batch = [img_prev.squeeze(0)]
    print(flow_imgs.shape, img.shape)
    #resize img to be the same size as flow_imgs
    img = F.resize(img, flow_imgs.shape[2:])
    # grid = [[img1, img_prev, flow_img] for (img1, img_prev, flow_img) in zip(img1_batch, img2_batch, flow_imgs)]
    save_img = torch.cat((img, flow_imgs.cpu()), dim=3)
    save_img = save_img[0].permute(1, 2, 0).cpu().detach().numpy()
    # save_img = (save_img - save_img.min()) / (save_img.max() - save_img.min())
    save_img = (save_img * 255).astype(np.uint8)
    save_img = Image.fromarray(save_img)
    import time
    save_img.save('real_world_of/save_img_{}.tiff'.format(time.time()))
    # plot(grid)