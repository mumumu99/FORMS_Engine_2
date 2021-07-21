import numpy as np
import cv2

def vis_parsing_maps(im, parsing_anno, stride, save_im=False, save_path='vis_results/parsing_map_on_im.jpg'):
    im = np.array(im)
    vis_im = im.copy().astype(np.uint8)
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    mask2 = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3))

    for pi in range(1, 11):
        index = np.where(vis_parsing_anno == pi)
        mask2[index[0], index[1]] = [255,255,255]

    mask2 = cv2.erode(mask2, (81, 81), 1)
    mask2 = cv2.GaussianBlur(mask2, (41, 41), 0)
    #mask2 = cv2.resize(mask2, dsize=(0, 0), fx=1/2, fy=1/2, interpolation=cv2.INTER_LINEAR)
    #cv2_imshow(mask2)
    #print(mask2.shape)

    return mask2/255