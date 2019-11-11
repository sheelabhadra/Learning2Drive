# Original code from https://github.com/araffin/robotics-rl-srl
# Authors: Antonin Raffin, René Traoré, Ashley Hill
import queue
import time
from multiprocessing import Queue, Process

import cv2
import numpy as np
from joblib import Parallel, delayed

from config import IMAGE_WIDTH, IMAGE_HEIGHT, ROI


def preprocess_input(x, mode="rl"):
    """
    Normalize input.

    :param x: (np.ndarray) (RGB image with values between [0, 255])
    :param mode: (str) One of "image_net", "tf" or "rl".
        - rl: divide by 255 only (rescale to [0, 1])
        - image_net: will zero-center each color channel with
            respect to the ImageNet dataset,
            with scaling.
            cf http://pytorch.org/docs/master/torchvision/models.html
        - tf: will scale pixels between -1 and 1,
            sample-wise.
    :return: (np.ndarray)
    """
    assert x.shape[-1] == 3, "Color channel must be at the end of the tensor {}".format(x.shape)
    # RL mode: divide only by 255
    x /= 255.

    if mode == "tf":
        x -= 0.5
        x *= 2.
    elif mode == "image_net":
        # Zero-center by mean pixel
        x[..., 0] -= 0.485
        x[..., 1] -= 0.456
        x[..., 2] -= 0.406
        # Scaling
        x[..., 0] /= 0.229
        x[..., 1] /= 0.224
        x[..., 2] /= 0.225
    elif mode == "rl":
        pass
    else:
        raise ValueError("Unknown mode for preprocessing")
    return x


def denormalize(x, mode="rl"):
    """
    De normalize data (transform input to [0, 1])

    :param x: (np.ndarray)
    :param mode: (str) One of "image_net", "tf", "rl".
    :return: (np.ndarray)
    """

    if mode == "tf":
        x /= 2.
        x += 0.5
    elif mode == "image_net":
        # Scaling
        x[..., 0] *= 0.229
        x[..., 1] *= 0.224
        x[..., 2] *= 0.225
        # Undo Zero-center
        x[..., 0] += 0.485
        x[..., 1] += 0.456
        x[..., 2] += 0.406
    elif mode == "rl":
        pass
    else:
        raise ValueError("Unknown mode for denormalize")
    # Clip to fix numeric imprecision (1e-09 = 0)
    return (255 * np.clip(x, 0, 1)).astype(np.uint8)


def preprocess_image(image, convert_to_rgb=False):
    """
    Crop, resize and normalize image.
    Optionnally it also converts the image from BGR to RGB.

    :param image: (np.ndarray) image (BGR or RGB)
    :param convert_to_rgb: (bool) whether the conversion to rgb is needed or not
    :return: (np.ndarray)
    """
    # Crop
    # Region of interest
    r = ROI
    image = image[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
    # Resize
    im = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_AREA)
    # Convert BGR to RGB
    if convert_to_rgb:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    # Normalize
    im = preprocess_input(im.astype(np.float32), mode="rl")

    return im


class DataLoader(object):
    def __init__(self, minibatchlist, images_path, n_workers=1, folder='logs/recorded_data/',
                 infinite_loop=True, max_queue_len=4, is_training=False):
        """
        A Custom dataloader to preprocessing images and feed them to the network.

        :param minibatchlist: ([np.array]) list of observations indices (grouped per minibatch)
        :param images_path: (np.array) Array of path to images
        :param n_workers: (int) number of preprocessing worker (load and preprocess each image)
        :param folder: (str)
        :param infinite_loop: (bool) whether to have an iterator that can be resetted, set to False, it
        :param max_queue_len: (int) Max number of minibatches that can be preprocessed at the same time
        :param is_training: (bool)
        """
        super(DataLoader, self).__init__()
        self.n_workers = n_workers
        self.infinite_loop = infinite_loop
        self.n_minibatches = len(minibatchlist)
        self.minibatchlist = minibatchlist
        self.images_path = images_path
        self.shuffle = is_training
        self.folder = folder
        self.queue = Queue(max_queue_len)
        self.process = None
        self.start_process()

    @staticmethod
    def create_minibatch_list(n_samples, batch_size):
        """
        Create list of minibatches.

        :param n_samples: (int)
        :param batch_size: (int)
        :return: ([np.array])
        """
        minibatchlist = []
        for i in range(n_samples // batch_size + 1):
            start_idx = i * batch_size
            end_idx = min(n_samples, (i + 1) * batch_size)
            minibatchlist.append(np.arange(start_idx, end_idx))
        return minibatchlist

    def start_process(self):
        """Start preprocessing process"""
        self.process = Process(target=self._run)
        # Make it a deamon, so it will be deleted at the same time
        # of the main process
        self.process.daemon = True
        self.process.start()

    def _run(self):
        start = True
        with Parallel(n_jobs=self.n_workers, batch_size="auto", backend="threading") as parallel:
            while start or self.infinite_loop:
                start = False

                if self.shuffle:
                    indices = np.random.permutation(self.n_minibatches).astype(np.int64)
                else:
                    indices = np.arange(len(self.minibatchlist), dtype=np.int64)

                for minibatch_idx in indices:

                    images = self.images_path[self.minibatchlist[minibatch_idx]]

                    if self.n_workers <= 1:
                        batch = [self._make_batch_element(self.folder, image_path)
                                 for image_path in images]

                    else:
                        batch = parallel(delayed(self._make_batch_element)(self.folder, image_path)
                                         for image_path in images)

                    batch = np.concatenate(batch, axis=0)

                    if self.shuffle:
                        self.queue.put((minibatch_idx, batch))
                    else:
                        self.queue.put(batch)

                    # Free memory
                    del batch

                self.queue.put(None)

    @classmethod
    def _make_batch_element(cls, folder, image_path):
        """
        :param image_path: (str) path to an image (without the 'data/' prefix)
        :return: (np.ndarray)
        """
        image_path = folder + image_path

        im = cv2.imread(image_path)
        if im is None:
            raise ValueError("tried to load {}.jpg, but it was not found".format(image_path))

        im = preprocess_image(im)

        im = im.reshape((1,) + im.shape)
        return im

    def __len__(self):
        return self.n_minibatches

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            try:
                val = self.queue.get_nowait()
                break
            except queue.Empty:
                time.sleep(0.001)
                continue
        if val is None:
            raise StopIteration
        return val

    def __del__(self):
        if self.process is not None:
            self.process.terminate()
