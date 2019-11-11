# Original code from https://github.com/araffin/robotics-rl-srl
# Authors: Antonin Raffin, René Traoré, Ashley Hill
import argparse

import cv2
import numpy as np

from vae.controller import VAEController


def create_figure_and_sliders(name, state_dim):
    """
    Creating a window for the latent space visualization,
    and another one for the sliders to control it.

    :param name: name of model (str)
    :param state_dim: (int)
    :return:
    """
    # opencv gui setup
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, 500, 500)
    cv2.namedWindow('slider for ' + name)
    # add a slider for each component of the latent space
    for i in range(state_dim):
        # the sliders MUST be between 0 and max, so we placed max at 100, and start at 50
        # So that when we substract 50 and divide 10 we get [-5,5] for each component
        cv2.createTrackbar(str(i), 'slider for ' + name, 50, 100, (lambda a: None))


def main():
    parser = argparse.ArgumentParser(description="latent space enjoy")
    parser.add_argument('--log-dir', default='', type=str, help='directory to load model')
    parser.add_argument('-vae', '--vae-path', help='Path to saved VAE', type=str, default='')

    args = parser.parse_args()

    vae = VAEController()
    vae.load(args.vae_path)

    fig_name = "Decoder for the VAE"

    # TODO: load data to infer bounds
    bound_min = -10
    bound_max = 10

    create_figure_and_sliders(fig_name, vae.z_size)

    should_exit = False
    while not should_exit:
        # stop if escape is pressed
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

        state = []
        for i in range(vae.z_size):
            state.append(cv2.getTrackbarPos(str(i), 'slider for ' + fig_name))
        # Rescale the values to fit the bounds of the representation
        state = (np.array(state) / 100) * (bound_max - bound_min) + bound_min

        reconstructed_image = vae.decode(state[None])[0]

        # stop if user closed a window
        if (cv2.getWindowProperty(fig_name, 0) < 0) or (cv2.getWindowProperty('slider for ' + fig_name, 0) < 0):
            should_exit = True
            break
        cv2.imshow(fig_name, reconstructed_image)

    # gracefully close
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
