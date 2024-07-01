import glob
import os
from models.network_swin2sr import Swin2SR as net
import torch
import cv2
import numpy as np



MODEL_PATH = 'super_resolution/SR backend/model_zoo/Swin2SR_CompressedSR_X4_48.pth'
TASK = 'compressed_sr'
TRAINING_PATCH_SIZE = 48
SCALE = 4
INPUTS = 'super_resolution/media/images/inputs'
testpath = os.path.join(INPUTS, 'baki_pose.jpg')
OUTPUTS = 'super_resolution/media/images/outputs/'
window_size = 8


# TODO: define parameters method (mb user)
# TODO: rewrite this shit

def define_model(task, training_patch_size, model_path):
    model = net(upscale=4, in_chans=3, img_size=48, window_size=8,
                img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                mlp_ratio=2, upsampler='pixelshuffle_aux', resi_connection='1conv')
    param_key_g = 'params'  


    pretrained_model = torch.load(model_path)
    model.load_state_dict(pretrained_model, strict=True)

    return model



def get_image_pair(task, scale, folder_lq, save_img_only, path):
    (imgname, imgext) = os.path.splitext(os.path.basename(path))
    if task in ['classical_sr', 'lightweight_sr']:
        if save_img_only:
            img_gt = None
            img_lq = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.            
        else:
            img_gt = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
            img_lq = cv2.imread(f'{folder_lq}/{imgname}x{scale}{imgext}', cv2.IMREAD_COLOR).astype(
                np.float32) / 255.
            
            
    return imgname, img_lq, img_gt


def test(img_lq, model, window_size, tile=1):
    if tile is None:
        # test the image as a whole
        output = model(img_lq)
    


# print(get_image_pair(TASK, SCALE, INPUTS, True, testpath))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = define_model(TASK, TRAINING_PATCH_SIZE, MODEL_PATH)
model.eval()
model = model.to(device)
print(model)


# for idx, path in enumerate(sorted(glob.glob(os.path.join(INPUTS, '*')))):
#     # read image
#     imgname, img_lq, img_gt = get_image_pair(TASK, SCALE, INPUTS, True, path)  # image to HWC-BGR, float32
#     img_lq = np.transpose(img_lq if img_lq.shape[2] == 1 else img_lq[:, :, [2, 1, 0]], (2, 0, 1))  # HCW-BGR to CHW-RGB
#     img_lq = torch.from_numpy(img_lq).float().unsqueeze(0).to(device)  # CHW-RGB to NCHW-RGB

#     # inference
#     with torch.no_grad():
#         # pad input image to be a multiple of window_size
#         _, _, h_old, w_old = img_lq.size()
#         h_pad = (h_old // window_size + 1) * window_size - h_old
#         w_pad = (w_old // window_size + 1) * window_size - w_old
#         img_lq = torch.cat([img_lq, torch.flip(img_lq, [2])], 2)[:, :, :h_old + h_pad, :]
#         img_lq = torch.cat([img_lq, torch.flip(img_lq, [3])], 3)[:, :, :, :w_old + w_pad]
#         print(img_lq)
#         output = test(img_lq, model, window_size)
        

#     # save image
#     output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
#     if output.ndim == 3:
#         output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HCW-BGR
#     output = (output * 255.0).round().astype(np.uint8)  # float32 to uint8
#     cv2.imwrite(f'{OUTPUTS}/{imgname}_Swin2SR.png', output)



