from dlib_alignment import dlib_detect_face, face_recover
import easydict
import argparse
import utils

def get_FaceSR_opt():
  args = easydict.EasyDict({
    "gpu_ids": None,
    "batch_size": 32,
    "Ir_G": 1e-4,
    "weight_decay_G": 0,
    "beta1_G": 0.9,
    "beta2_G": 0.99,
    "Ir_D": 1e-4,
    "weight_decay_D": 0,
    "beta1_D": 0.9,
    "beta2_D": 0.99,
    "Ir_scheme": 'MultiStepLR',
    "niter": 100000,
    "warmup_iter": -1,
    "lr_steps": [50000],
    "lr_gamma": 0.5,
    "pixel_criterion": 'l1',
    "pixel_weight": 1e-2,
    "feature_criterion": 'l1',
    "feature_weight": 1,
    "gan_type": 'ragan',
    "gan_weight": 5e-3,
    "D_update_ratio": 1,
    "D_init_iters": 0,

    "print_freq": 100,
    "val_freq": 1000,
    "save_freq": 10000,
    "crop_size": 0.85,
    "lr_size": 128,
    "hr_size": 512,

    # network G
    "which_model_G": 'RRDBNet',
    "G_in_nc": 3,
    "out_nc": 3,
    "G_nf": 64,
    "nb": 16,

    # network D
    "which_model_D": 'discriminator_vgg_128',
    "D_in_nc": 3,
    "D_nf": 64,

    # data dir
    "pretrain_model_G": '90000_G.pth',
    "pretrain_model_D": None
  })
  '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_ids', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr_G', type=float, default=1e-4)
    parser.add_argument('--weight_decay_G', type=float, default=0)
    parser.add_argument('--beta1_G', type=float, default=0.9)
    parser.add_argument('--beta2_G', type=float, defauloutputt=0.99)
    parser.add_argument('--lr_D', type=float, default=1e-4)
    parser.add_argument('--weight_decay_D', type=float, default=0)
    parser.add_argument('--beta1_D', type=float, default=0.9)
    parser.add_argument('--beta2_D', type=float, default=0.99)
    parser.add_argument('--lr_scheme', type=str, default='MultiStepLR')
    parser.add_argument('--niter', type=int, default=100000)
    parser.add_argument('--warmup_iter', type=int, default=-1)
    parser.add_argument('--lr_steps', type=list, default=[50000])
    parser.add_argument('--lr_gamma', type=float, default=0.5)
    parser.add_argument('--pixel_criterion', type=str, default='l1')
    parser.add_argument('--pixel_weight', type=float, default=1e-2)
    parser.add_argument('--feature_criterion', type=str, default='l1')
    parser.add_argument('--feature_weight', type=float, default=1)
    parser.add_argument('--gan_type', type=str, default='ragan')
    parser.add_argument('--gan_weight', type=float, default=5e-3)
    parser.add_argument('--D_update_ratio', type=int, default=1)
    parser.add_argument('--D_init_iters', type=int, default=0)

    parser.add_argument('--print_freq', type=int, default=100)
    parser.add_argument('--val_freq', type=int, default=1000)
    parser.add_argument('--save_freq', type=int, default=10000)
    parser.add_argument('--crop_size', type=float, default=0.85)
    parser.add_argument('--lr_size', type=int, default=128)
    parser.add_argument('--hr_size', type=int, default=512)

    # network G
    parser.add_argument('--which_model_G', type=str, default='RRDBNet')
    parser.add_argument('--G_in_nc', type=int, default=3)
    parser.add_argument('--out_nc', type=int, default=3)
    parser.add_argument('--G_nf', type=int, default=64)
    parser.add_argument('--nb', type=int, default=16)

    # network D
    parser.add_argument('--which_model_D', type=str, default='discriminator_vgg_128')
    parser.add_argument('--D_in_nc', type=int, default=3)
    parser.add_argument('--D_nf', type=int, default=64)

    # data dir
    parser.add_argument('--pretrain_model_G', type=str, default='90000_G.pth')
    parser.add_argument('--pretrain_model_D', type=str, default=None)

    args = parser.parse_args(args[])
    '''
  return args

def sr_forward(img, padding=0.5, moving=0.1):
    img_aligned, M = dlib_detect_face(img, padding=padding, image_size=(128, 128), moving=moving)
    input_img = torch.unsqueeze(_transform(Image.fromarray(img_aligned)), 0)
    sr_model.var_L = input_img.to(sr_model.device)
    sr_model.test()
    output_img = sr_model.fake_H.squeeze(0).cpu().numpy()
    output_img = np.clip((np.transpose(output_img, (1, 2, 0)) / 2.0 + 0.5) * 255.0, 0, 255).astype(np.uint8)
    rec_img = face_recover(output_img, M * 4, img)
    return output_img, rec_img