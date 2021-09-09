import numpy as np
import cv2
from torchvision.utils import save_image
import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

show_progress = True # print progress while processing individual images

def print_progress(i, total):
    i += 1
    if show_progress:
        print("\r\t\t\t\r" + str(i) + "/" + str(total) + " done.", end="")
        if i == total:
            print()


def get_neighbour_imgs(img, array_path, knn_idxs, k, text=None):
    img = np.stack((img, img, img))

    # border around query img
    border_size = 5
    img[:2, :, :border_size] = 0
    img[:2, :, -border_size:] = 0
    img[:2, :border_size, :] = 0
    img[:2, -border_size:, :] = 0
    img[2, :, :border_size] = 255
    img[2, :, -border_size:] = 255
    img[2, :border_size, :] = 255
    img[2, -border_size:, :] = 255

    out = [img]
    for idx in knn_idxs:
        img = find_img(array_path, idx)
        img = np.stack((img, img, img))
        out.append(img)

    if text is not None:
        annotate_img(out[0][2], text)
    return out

def make_image_grid(imgs, n_per_row):
    for i in range(len(imgs)):
        imgs[i] = torch.tensor(imgs[i])
    imgs = make_grid(imgs, nrow=n_per_row)
    imgs = imgs.numpy()
    return imgs

def annotate_img(img, text):
	font = cv2.FONT_HERSHEY_SIMPLEX
	top_left_corner = (10, 30)
	font_scale = 1
	font_color = (255, 255, 255)
	line_type = 2

	cv2.putText(
        img, text, top_left_corner, font, font_scale, font_color, line_type
    )

def find_img(array_path, idx):
    HEADER_LEN = 128
    IMAGE_SIZE = 256 * 256
    FLOAT_SIZE = 4
    off = HEADER_LEN + IMAGE_SIZE * FLOAT_SIZE * idx
    image = np.memmap(array_path, dtype="float32", offset=off,
                      mode="r", shape=(256,256))
    image = (image * 255).astype('uint8').copy()
    return image