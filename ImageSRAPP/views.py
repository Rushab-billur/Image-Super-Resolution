import cv2
import numpy as np
import os
import torch
from ImageSRAPP.network_swinir import SwinIR as net
from django.shortcuts import render,redirect
from ImageSRAPP import models
from django.contrib import messages as msg

# Create your views here.

def home(request):
    return render(request, 'index.html')

def second_page(request):
    return render(request, 'second.html')

def second_page_Form(request):
    if request.method == 'POST':
        image_inst = models.LR_Image_table()
        image_inst2 = models.SR_Image_table_sr()
        if len(request.FILES) != 0:
            image_inst.lr_image = request.FILES['lr_img']
            image_inst.save()
            cwd = os.getcwd()
            files = os.listdir(cwd)
            print("Files in %r: %s" % (cwd, files))
            model_path = 'ImageSRAPP/trained_model/model.pth'
            print('model_path : ', model_path)
            window_size = 8
            scale = 4
            model = net(upscale=scale, in_chans=3, img_size=48, window_size=8,
                                img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                                mlp_ratio=2)
            model.load_state_dict(torch.load(model_path)['params'], strict=False)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.eval()
            model = model.to(device)

            img_path = image_inst.lr_image
            print('image_inst.lr_image : ', image_inst.lr_image)
            print('img_path : ', img_path)
            print('img_path.name : ',img_path.name )
            (imgname, imgext) = os.path.splitext(os.path.basename(img_path.name))
            img_gt = cv2.imread(f'{img_path}', cv2.IMREAD_COLOR).astype(np.float32) / 255.

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

            save_dir = 'static/SR'
            output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
            if output.ndim == 3:
                output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0)) 
            output = (output * 255.0).round().astype(np.uint8)  
            cv2.imwrite(f'{save_dir}/{imgname}_X4_SwinIR5.png', output)
            image_inst2.sr_image = f'{save_dir}/{imgname}_X4_SwinIR5.png'
            image_inst2.save()
        else:
            msg.warning(request, "Please insert the Image!")
            return redirect(second_page)
    return redirect(result)

def result(request):
     lr_inst = models.LR_Image_table.objects.all()
     sr_inst = models.SR_Image_table_sr.objects.all()

     lr_len = (len(lr_inst))
     sr_len = (len(sr_inst))
     lr_new_inst = models.LR_Image_table.objects.filter(id=lr_len)
     sr_new_inst = models.SR_Image_table_sr.objects.filter(id=sr_len)
     return render(request, 'result.html', {'lr_image' : lr_new_inst, 'sr_image' :sr_new_inst})