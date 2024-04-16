from torchvision.models.optical_flow import raft_small, raft_large
from torchvision.models.optical_flow import Raft_Small_Weights
import torchvision.transforms.functional as F
import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
plt.rcParams["savefig.bbox"] = "tight"
import tempfile
from pathlib import Path
from urllib.request import urlretrieve


video_url = "https://download.pytorch.org/tutorial/pexelscom_pavel_danilyuk_basketball_hd.mp4"
video_path = Path(tempfile.mkdtemp()) / "basketball.mp4"
_ = urlretrieve(video_url, video_path)

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
            try:
                img = F.to_pil_image(img.to("cpu"))
            except:
                img = F.to_pil_image(img)
            ax.imshow(np.asarray(img), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    #Label the axes
    axs[0, 0].set_ylabel("Input")
    axs[0, 1].set_ylabel("Predicted Flow")
    axs[0, 2].set_ylabel("Predicted Flow CV")

    plt.tight_layout()
    plt.show()

method = cv2.optflow.calcOpticalFlowDenseRLOF

from torchvision.io import read_video
frames, _, _ = read_video(str(video_path), output_format="TCHW")

img1_batch = torch.stack([frames[100], frames[150]])
img2_batch = torch.stack([frames[102], frames[152]])

plot([img1_batch[0], img1_batch[1], img2_batch[0], img2_batch[1]])
weights = Raft_Small_Weights.DEFAULT
transforms = weights.transforms()


# If you can, run this example on a GPU, it will be a lot faster.
device = "cuda" if torch.cuda.is_available() else "cpu"
def preprocess(img1_batch, img2_batch):
    img1_batch = F.resize(img1_batch, size=[520, 960], antialias=False)
    img2_batch = F.resize(img2_batch, size=[520, 960], antialias=False)
    return transforms(img1_batch, img2_batch)


img1_batch, img2_batch = preprocess(img1_batch, img2_batch)
model = raft_small(weights=Raft_Small_Weights.DEFAULT, progress=False).to(device)
# model = raft_large(weights="DEFAULT", progress=False).to(device)
model = model.eval()

with torch.no_grad():
    list_of_flows = model(img1_batch.to(device), img2_batch.to(device), num_flow_updates=12)
print(f"type = {type(list_of_flows)}")
print(f"length = {len(list_of_flows)} = number of iterations of the model")


predicted_flows = list_of_flows[-1]
predicted_flows[:, 0, :, :] /= 520
predicted_flows[:, 1, :, :] /= 960
print(f"dtype = {predicted_flows.dtype}")
print(f"shape = {predicted_flows.shape} = (N, 2, H, W)")
print(f"min = {predicted_flows.min()}, max = {predicted_flows.max()}")
#conver frames to numpy
print(f"frames[100] = {frames[100].shape}")
#resize frames
frame_a = cv2.cvtColor(cv2.resize(np.moveaxis(np.array(frames[100]), 0, 2), (960, 520)), cv2.COLOR_RGB2BGR)
frame_b = cv2.cvtColor(cv2.resize(np.moveaxis(np.array(frames[105]), 0, 2), (960, 520)), cv2.COLOR_RGB2BGR)
frame_c = cv2.cvtColor(cv2.resize(np.moveaxis(np.array(frames[150]), 0, 2), (960, 520)), cv2.COLOR_RGB2BGR)
frame_d = cv2.cvtColor(cv2.resize(np.moveaxis(np.array(frames[155]), 0, 2), (960, 520)), cv2.COLOR_RGB2BGR)

flow_1 = method(frame_a, frame_b, None)
flow_2 = method(frame_c, frame_d, None)
from torchvision.utils import flow_to_image

# flow_imgs = flow_to_image(predicted_flows)
# flow_imgs_cv = flow_to_image(torch.from_numpy(flow_1))
# flow_imgs_cv2 = flow_to_image(torch.from_numpy(flow_2))
#Append flow_1 and flow_2 to predicted_flows
flow_1 = np.moveaxis(flow_1, -1, 0)
#scale
flow_1[0] /= 520
flow_1[1] /= 960
flow_2 = np.moveaxis(flow_2, -1, 0)
print(f"flow_1 = {flow_1.shape}", f"flow_2 = {flow_2.shape}", f"predicted_flows = {predicted_flows.shape}")
predicted_flows = torch.cat([predicted_flows.cpu(), torch.from_numpy(flow_1).unsqueeze(0), torch.from_numpy(flow_2).unsqueeze(0)], dim=0)
print(predicted_flows.shape)
predicted_flows_img = flow_to_image(predicted_flows)
predicted_flows_dl = predicted_flows_img[:2]
predicted_flows_cv = predicted_flows_img[2:]
# The images have been mapped into [-1, 1] but for plotting we want them in [0, 1]
img1_batch = [(img1 + 1) / 2 for img1 in img1_batch]
for a,b,c in zip(img1_batch, predicted_flows_dl, predicted_flows_cv):
    print(f"a = {a.shape}, b = {b.shape}, c = {c.shape}")
grid = [[img1, flow_img, flow_img_cv] for (img1, flow_img, flow_img_cv) in zip(img1_batch, predicted_flows_dl, predicted_flows_cv)]
# plot(grid)
grid = [predicted_flows[1][0], predicted_flows[1][1], flow_2[0], flow_2[1]]
#normalize grid by sum of squares
# grid = [i/np.sqrt((i**2).sum()) for i in grid]
#print statistics of the flow
print(f"min = {predicted_flows[0][0].min()}, max = {predicted_flows[0][0].max()}")
print(f"min = {flow_1[0].min()}, max = {flow_1[0].max()}")
plot(grid, cmap = 'gray')