# Learning to Drive in CARLA

Learning to drive in CARLA v0.8.2 using reinforcement learning. The code has been tested with Python 3.5 and Tensorflow 1.14.0.

## Quick Start
0. Install CARLA v0.8.2.
1. Install dependencies (cf requirements.txt).
2. (Optional) Download pre-trained VAE [here](https://drive.google.com/file/d/1cVnXp389UynmcYe1lECdZ3iG0UMj0led/view?usp=sharing).
3. Train a control policy for 10000 steps using Soft Actor-Critic (SAC).

```
python train.py --algo sac -vae <path-to-vae> -n 10000
```

4. Enjoy the trained agent for 2000 steps.

```
python enjoy.py --algo sac -vae <path-to-vae> -model <path-to-trained-model>
```

## Train the Variational Autoencoder (VAE)
0. Collect images by manually driving the car around the track. Don't forget to store the images captured by the camera on the car into a folder.

```
python manual_control.py 
```
1. Train a VAE.

```
python -m vae.train --n-epochs 50 --verbose 0 --z-size 64 -f <path-to-recorded-images>
```

2. Explore Latent Space

```
python -m vae.enjoy_latent -vae <path-to-vae>

## Smooth Control
With the current reward function used, the car doesn't achieve a smooth control. A way around the is to restrict the maximum change in the steering angle at each step. This is generally acceptable since humans drive in a similar manner. For more information check this awesome [post](https://medium.com/@araffin/learning-to-drive-smoothly-in-minutes-450a7cdb35f4).  

Set the following values in `config.py`.

```python
MAX_STEERING_DIFF = 0.15 # 0.1 for very smooth control, but it requires more steps
MAX_THROTTLE = 0.6 # MAX_THROTTLE = 0.5 is fine, but we can go faster
```

## Credits
- Most of the code has been adapted from ["Learning to Drive Smoothly in minutes"](https://github.com/araffin/learning-to-drive-in-5-minutes/).
- Related Paper: ["Learning to Drive in a Day"](https://arxiv.org/pdf/1807.00412.pdf).
- [Wayve.ai](https://wayve.ai) for idea and inspiration.
- [Stable-Baselines](https://github.com/hill-a/stable-baselines) for DDPG/SAC and PPO implementations.