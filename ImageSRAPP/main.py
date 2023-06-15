# import argparse
import cv2
# import glob
import numpy as np
import os
import torch
from network_swinir import SwinIR as net


model_path = 'model.pth'
window_size = 8
scale = 4
model = net(upscale=scale, in_chans=3, img_size=48, window_size=8,
                    img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                    mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv')
model.load_state_dict(torch.load(model_path)['params'], strict=False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.eval()
model = model.to(device)

img_path = 'LR/babyx2.png'
(imgname, imgext) = os.path.splitext(os.path.basename(img_path))
img_gt = cv2.imread(img_path, cv2.IMREAD_COLOR).astype(np.float32) / 255.

img_lq = cv2.imread(f'{img_path}', cv2.IMREAD_COLOR).astype(np.float32) / 255.

img_lq = np.transpose(img_lq if img_lq.shape[2] == 1 else img_lq[:, :, [2, 1, 0]], (2, 0, 1))  
img_lq = torch.from_numpy(img_lq).float().unsqueeze(0).to(device)  

with torch.no_grad():
        _, _, h_old, w_old = img_lq.size()
        h_pad = (h_old // window_size + 1) * window_size - h_old
        w_pad = (w_old // window_size + 1) * window_size - w_old
        img_lq = torch.cat([img_lq, torch.flip(img_lq, [2])], 2)[:, :, :h_old + h_pad, :]
        img_lq = torch.cat([img_lq, torch.flip(img_lq, [3])], 3)[:, :, :, :w_old + w_pad]
        output = model(img_lq)
        output = output[..., :h_old*scale, :w_old*scale]

save_dir = 'LR'
output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
if output.ndim == 3:
    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0)) 
output = (output * 255.0).round().astype(np.uint8)  
cv2.imwrite(f'{save_dir}/{imgname}_X4_SwinIR5.png', output)