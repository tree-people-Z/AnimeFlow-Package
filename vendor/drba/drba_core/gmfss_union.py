# for high quality output
from drba_core.model_gmfss_union.GMFSS import Model
from drba_core.rife_426_heavy.IFNet_HDv3 import IFNet
from drba_core.utils.tools import *
import torch
import os


class GMFSS_UNION:
    def __init__(self, weights='weights/train_log_gmfss_union',
                 scale=1.0, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        self.model = Model()
        self.model.load_model(weights, -1)
        self.model.device(device)
        self.model.eval()
        self.ifnet = IFNet().to(device).eval()
        self.ifnet.load_state_dict(
            convert(torch.load(os.path.join(weights, "rife.pkl"), map_location='cpu')),
            strict=False)
        self.scale = scale
        self.scale_list = [16 / self.scale, 8 / self.scale, 4 / self.scale, 2 / self.scale, 1 / self.scale]
        self.pad_size = 128

    @torch.inference_mode()
    @torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu")
    def inference_ts(self, I0, I1, ts):
        reuse = self.model.reuse(I0, I1, self.scale)
        output = []
        for t in ts:
            if t == 0:
                output.append(I0)
            elif t == 1:
                output.append(I1)
            else:
                I0s = F.interpolate(I0, scale_factor=0.5, mode='bilinear', align_corners=False)
                I1s = F.interpolate(I1, scale_factor=0.5, mode='bilinear', align_corners=False)
                rife = self.ifnet(torch.cat((I0s, I1s), 1), timestep=t, scale_list=self.scale_list)[0]
                output.append(
                    self.model.inference(I0, I1, reuse, timestep0=t, timestep1=1 - t, rife=rife)
                )

        return output

