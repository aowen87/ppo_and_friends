from .utils import *
import numpy as np
import torch
import gym


class CartPoleEnvManager(object):

    def __init__(self):
        super(CartPoleEnvManager, self)

        self.env            = gym.make("CartPole-v0").unwrapped
        self.current_screen = None
        self.done           = False
        self.action_space   = self.env.action_space

        self.env.reset()
        screen_size = self.get_screen_height() * self.get_screen_width() * 3
        self.observation_space = CustomObservationSpace((screen_size,))

    def reset(self):
        self.env.reset()
        self.current_screen = None
        return self.get_screen_state()

    def close(self):
        self.env.close()

    def render(self, mode='human'):
        return self.env.render(mode)

    def num_actions_available(self):
        return self.env.action_space.n

    def step(self, action):
        _, reward, self.done, info = self.env.step(action.item())
        obs = self.get_screen_state()
        return obs, reward, self.done, info

    def just_starting(self):
        return self.current_screen is None

    def get_screen_state(self):
        if self.just_starting() or self.done:
            self.current_screen = self.get_processed_screen()
            black_screen = np.zeros_like(self.current_screen)
            return black_screen.flatten()
        else:
            screen_1 = self.current_screen
            screen_2 = self.get_processed_screen()
            self.current_screen = screen_2
            return (screen_2 - screen_1).flatten()

    def get_screen_height(self):
        return self.get_processed_screen().shape[2]

    def get_screen_width(self):
        return self.get_processed_screen().shape[3]

    def get_processed_screen(self):
        screen = self.render("rgb_array").transpose((2, 0, 1))
        screen = self.crop_screen(screen)
        return self.transform_screen_data(screen)

    def crop_screen(self, screen):

        screen_height = screen.shape[1]
        top           = int(screen_height * 0.4)
        bottom        = int(screen_height * 0.8)

        screen_width  = screen.shape[2]
        left          = int(screen_width * 0.1)
        right         = int(screen_width * 0.9)
        screen        = screen[:, top : bottom, left : right]

        return screen

    def transform_screen_data(self, screen):

        screen = np.ascontiguousarray(screen, dtype=np.float32) / 255.
        screen = torch.from_numpy(screen)

        resize = t_transforms.Compose([
            t_transforms.ToPILImage(),
            t_transforms.Resize((40, 90)),
            t_transforms.ToTensor()])

        return resize(screen).unsqueeze(0).numpy()


#FIXME: change to inherit from gym.Wrapper
#TODO: create a pixel wrapper that stacks frames rather than uses diff.
class AtariEnvWrapper(object):

    def __init__(self,
                 env,
                 min_lives = -1,
                 **kwargs):

        super(AtariEnvWrapper, self).__init__()

        self.min_lives          = min_lives
        self.env                = env
        self.action_space       = env.action_space

    def _end_game(self, done):
        return done or self.env.ale.lives() < self.min_lives


class AtariGrayScale(AtariEnvWrapper):

    def __init__(self,
                 env,
                 min_lives = -1):

        super(AtariGrayScale, self).__init__(
            env,
            min_lives)

    def rgb_to_cropped_gray_fp(self,
                               rgb_frame,
                               h_start = 0,
                               h_stop  = -1,
                               w_start = 0,
                               w_stop  = -1):

        rgb_frame  = rgb_frame.astype(np.float32) / 255.
        gray_dot   = np.array([0.2989, 0.587 , 0.114 ], dtype=np.float32)
        gray_frame = np.expand_dims(np.dot(rgb_frame, gray_dot), axis=0)

        return gray_frame[:, h_start : h_stop, w_start : w_stop]


class PixelDifferenceEnvWrapper(AtariGrayScale):

    def __init__(self,
                 env,
                 min_lives = -1):
        #
        # Allow these variables to be over-written by sub-classes.
        #
        self.h_start = 0
        self.h_stop  = -1
        self.w_start = 0
        self.w_stop  = -1

        super(PixelDifferenceEnvWrapper, self).__init__(
            env       = env,
            min_lives = min_lives)

        self.prev_frame   = None
        self.action_space = env.action_space
        self.h_start      = 0

        prev_shape = env.observation_space.shape
        new_shape  = (1, prev_shape[0], prev_shape[1])
        self.observation_space = CustomObservationSpace(new_shape)

    def reset(self):
        cur_frame = self.env.reset()
        cur_frame = self.rgb_to_cropped_gray_fp(
            cur_frame,
            h_start = self.h_start,
            h_stop  = self.h_stop,
            w_start = self.w_start,
            w_stop  = self.w_stop)

        self.prev_frame = cur_frame

        return self.prev_frame.copy()

    def step(self, action):
        cur_frame, reward, done, info = self.env.step(action)

        cur_frame = self.rgb_to_cropped_gray_fp(
            cur_frame,
            h_start = self.h_start,
            h_stop  = self.h_stop,
            w_start = self.w_start,
            w_stop  = self.w_stop)

        diff_frame      = cur_frame - self.prev_frame
        self.prev_frame = cur_frame.copy()

        done = self._end_game(done)

        return diff_frame, reward, done, info

    def render(self):
        self.env.render()


class PixelHistEnvWrapper(AtariGrayScale):

    def __init__(self,
                 env,
                 hist_size = 2,
                 min_lives = -1):

        prev_shape = env.observation_space.shape

        #
        # Allow these variables to be over-written by sub-classes.
        #
        self.h_start = 0
        self.h_stop  = prev_shape[0]
        self.w_start = 0
        self.w_stop  = prev_shape[1]

        super(PixelHistEnvWrapper, self).__init__(
            env       = env,
            min_lives = min_lives)

        self.frame_cache  = None
        self.action_space = env.action_space
        self.hist_size    = hist_size

        new_shape  = (hist_size, prev_shape[0], prev_shape[1])
        self.observation_space = CustomObservationSpace(new_shape)

    def reset(self):
        cur_frame = self.env.reset()
        cur_frame = self.rgb_to_cropped_gray_fp(
            cur_frame,
            h_start = self.h_start,
            h_stop  = self.h_stop,
            w_start = self.w_start,
            w_stop  = self.w_stop)

        self.frame_cache = np.tile(cur_frame, (self.hist_size, 1, 1))

        return self.frame_cache.copy()

    def step(self, action):
        cur_frame, reward, done, info = self.env.step(action)

        cur_frame = self.rgb_to_cropped_gray_fp(
            cur_frame,
            h_start = self.h_start,
            h_stop  = self.h_stop,
            w_start = self.w_start,
            w_stop  = self.w_stop)

        #FIXME: remove
        #from PIL import Image
        #import sys
        #foo = (cur_frame.squeeze() * 255.).astype(np.uint8)
        #img = Image.fromarray(foo, 'L')
        #img.show()
        #sys.exit(1)

        self.frame_cache = np.roll(self.frame_cache, 1, axis=0)
        self.frame_cache[-1] = cur_frame.copy()

        done = self._end_game(done)

        return self.frame_cache, reward, done, info

    def render(self):
        self.env.render()


class RAMHistEnvWrapper(AtariEnvWrapper):

    def __init__(self,
                 env,
                 hist_size = 2,
                 min_lives = -1,
                 **kwargs):

        super(RAMHistEnvWrapper, self).__init__(
            env       = env,
            min_lives =  min_lives)

        ram_shape   = env.observation_space.shape
        cache_shape = (ram_shape[0] * hist_size,)

        self.observation_space = CustomObservationSpace(
            cache_shape)

        self.min_lives          = min_lives
        self.ram_size           = ram_shape[0]
        self.cache_size         = cache_shape[0]
        self.hist_size          = hist_size
        self.env                = env
        self.ram_cache          = None
        self.action_space       = env.action_space

    def _reset_ram_cache(self,
                         cur_ram):
        self.ram_cache = np.tile(cur_ram, self.hist_size)

    def reset(self):
        cur_ram  = self.env.reset()
        cur_ram  = cur_ram.astype(np.float32) / 255.
        self._reset_ram_cache(cur_ram)

        return self.ram_cache.copy()

    def step(self, action):
        cur_ram, reward, done, info = self.env.step(action)
        cur_ram  = cur_ram.astype(np.float32) / 255.

        self.ram_cache = np.roll(self.ram_cache, -self.ram_size)

        offset = self.cache_size - self.ram_size
        self.ram_cache[offset :] = cur_ram.copy()

        done = self._end_game(done)

        return self.ram_cache.copy(), reward, done, info

    def render(self):
        self.env.render()


class BreakoutEnvWrapper():

    def __init__(self,
                 env,
                 **kwargs):

        super(BreakoutEnvWrapper, self).__init__(env, **kwargs)

        if "Breakout" not in env.spec._env_name:
            msg  = "ERROR: expected env to be a variation of Breakout "
            msg += "but received {}".format(env.spec._env_name)
            sys.stderr.write(msg)
            sys.exit(1)

        #
        # Breakout doesn't auto-launch the ball, which is a bit of a pain.
        # I don't care to teach the model that it needs to launch the ball
        # itself, so let's launch it autmatically when we reset. Also, let's
        # change the action space to only be (no-op, left, right) since we're
        # removing the ball launch action.
        #
        self.action_space = CustomActionSpace(
            env.action_space.dtype,
            3)

        self.action_map = [0, 2, 3]

    def _set_random_start_pos(self):
        #
        # 20 steps in either direction should be enough to
        # reach either wall.
        #
        rand_step   = np.random.randint(2, 4)
        rand_repeat = np.random.randint(1, 20)

        for _ in range(rand_repeat):
            self.env.step(rand_step)


class BreakoutRAMEnvWrapper(BreakoutEnvWrapper, RAMHistEnvWrapper):

    def __init__(self,
                 env,
                 hist_size = 2,
                 min_lives = -1):

        super(BreakoutRAMEnvWrapper, self).__init__(
            env       = env,
            hist_size = hist_size,
            min_lives = min_lives)

    def step(self, action):
        action = self.action_map[action]
        return RAMHistEnvWrapper.step(self, action)

    def reset(self):
        self.env.reset()

        #
        # First, we need to randomly place the paddle somewhere. This
        # will change where the ball is launched from.
        #
        self._set_random_start_pos()

        #
        # Next, launch the ball.
        #
        cur_ram, _, _, _ = self.env.step(1)

        cur_ram  = cur_ram.astype(np.float32) / 255.
        self._reset_ram_cache(cur_ram)

        return self.ram_cache.copy()


class BreakoutPixelsEnvWrapper(BreakoutEnvWrapper, PixelHistEnvWrapper):

    def __init__(self,
                 env,
                 hist_size = 2,
                 min_lives = -1):


        super(BreakoutPixelsEnvWrapper, self).__init__(
            env       = env,
            hist_size = hist_size,
            min_lives = min_lives)

        #
        # Crop the images by removing the "score" information.
        #
        prev_shape   = env.observation_space.shape
        self.h_start = 20
        self.h_stop  = prev_shape[0]
        self.w_start = 0
        self.w_stop  = prev_shape[1]
        new_shape    = (hist_size, prev_shape[0] - self.h_start, prev_shape[1])
        self.observation_space = CustomObservationSpace(new_shape)

    def step(self, action):
        action = self.action_map[action]
        return PixelHistEnvWrapper.step(self, action)

    def reset(self):
        self.env.reset()

        #
        # First, we need to randomly place the paddle somewhere. This
        # will change where the ball is launched from. 20 steps in either
        # direction from the default start is enough to get to the wall.
        #
        self._set_random_start_pos()

        #
        # Next, launch the ball.
        #
        cur_frame, _, _, _ = self.env.step(1)


        cur_frame = self.rgb_to_cropped_gray_fp(
            cur_frame,
            h_start = self.h_start,
            h_stop  = self.h_stop,
            w_start = self.w_start,
            w_stop  = self.w_stop)

        self.frame_cache = np.tile(cur_frame, (self.hist_size, 1, 1))

        return self.frame_cache.copy()