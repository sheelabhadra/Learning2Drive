"""
Test a trained vae
"""
import argparse
import os

import cv2
import numpy as np
from stable_baselines.common import set_global_seeds

from vae.controller import VAEController

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--folder', help='Log folder', type=str, default='logs/recorded_data/')
parser.add_argument('-vae', '--vae-path', help='Path to saved VAE', type=str, default='')
parser.add_argument('--n-samples', help='Max number of samples', type=int, default=20)
parser.add_argument('--seed', help='Random generator seed', type=int, default=0)
args = parser.parse_args()

set_global_seeds(args.seed)

if not args.folder.endswith('/'):
    args.folder += '/'

vae = VAEController()
vae.load(args.vae_path)

images = [im for im in os.listdir(args.folder) if im.endswith('.png')]
images = np.array(images)
n_samples = len(images)


for i in range(args.n_samples):
    # Load test image
    image_idx = np.random.randint(n_samples)
    image_path = args.folder + images[image_idx]
    image = cv2.imread(image_path)

    encoded = vae.encode_from_raw_image(image)
    reconstructed_image = vae.decode(encoded)[0]
    # Plot reconstruction
    cv2.imshow("Original", image)
    cv2.imshow("Reconstruction", reconstructed_image)
    cv2.waitKey(0)
