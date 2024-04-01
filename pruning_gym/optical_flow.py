import torch as th
from torchvision.models.optical_flow import Raft_Small_Weights, raft_small
from torchvision.transforms import functional as F
import numpy as np

class OpticalFlow:
    def __init__(self, size = (224, 224), subprocess = False, shared_var = (None, None), num_envs = 1):
        self.device = "cuda" if th.cuda.is_available() else "cpu"
        weights = Raft_Small_Weights.DEFAULT
        self.transforms = weights.transforms()
        model = raft_small(weights=Raft_Small_Weights.DEFAULT, progress=False).to(self.device)
        self.model = model.eval()
        self.size = size
        self.subprocess = subprocess
        self.shared_queue, self.shared_dict = shared_var
        print("raft model loaded")
        self.num_envs = num_envs
        if self.subprocess:
            self._run_subprocess()
    def _run_subprocess(self):
        while True:
            # rgb, previous_rgb, pid = self.shared_queue.get()
            #while queue is not empty

            #make batch of all the elements in the queue
            rgb_array = []
            previous_rgb_array = []
            pid_list = []

            while len(pid_list) < self.num_envs:
                #make an array of all the elements in the queue
                rgb, previous_rgb, pid, name = self.shared_queue.get()
                rgb_array.append(rgb)
                previous_rgb_array.append(previous_rgb)
                pid_list.append(pid)
                # print(name)
                # Different queues for record/eval/test
                if "test" in name or "eval" in name or "record" in name:
                    break
                #print("of len", len(pid_list))
            optical_flow = self.calculate_optical_flow(rgb_array, previous_rgb_array)

            for i, pid in enumerate(pid_list):
                self.shared_dict[pid] = optical_flow[i]

    def _preprocess(self, img1, img2):

        img1 = F.resize(img1, size=self.size, antialias=False)
        img2 = F.resize(img2, size=self.size, antialias=False)
        return self.transforms(img1, img2)

    def calculate_optical_flow(self, current_rgb, previous_rgb):
        #convert to np array
        current_rgb = np.array(current_rgb)
        previous_rgb = np.array(previous_rgb)
        if len(current_rgb.shape)==3:
            current_rgb = np.expand_dims(current_rgb, 0)
            previous_rgb = np.expand_dims(previous_rgb, 0)
        current_rgb, previous_rgb = self._preprocess(th.tensor(current_rgb).permute(0, 3, 1, 2),
                                                     th.tensor(previous_rgb).permute(0, 3, 1, 2))
        with th.no_grad():
            list_of_flows = self.model(current_rgb.to(self.device), previous_rgb.to(self.device))
        predicted_flows = list_of_flows[-1]
        predicted_flows[:, 0, :, :] /= self.size[0]
        predicted_flows[:, 1, :, :] /= self.size[1]

        # from torchvision.utils import flow_to_image
        # print(predicted_flows.shape, predicted_flows.max(), predicted_flows.min())
        #
        # flow_img = flow_to_image(predicted_flows)
        # print(flow_img.shapee, flow_img.max(), flow_img.min())
        return predicted_flows.cpu().numpy()
