import gymnasium as gym
from ppo_and_friends.environments.gym.wrappers import SingleAgentGymWrapper
from ppo_and_friends.policies.utils import get_single_policy_defaults
from ppo_and_friends.runners.env_runner import GymRunner
from ppo_and_friends.networks.ppo_networks.feed_forward import FeedForwardNetwork
from ppo_and_friends.utils.schedulers import *
import torch.nn as nn
from ppo_and_friends.runners.runner_tags import ppoaf_runner

@ppoaf_runner
class BipedalWalkerHardcoreRunner(GymRunner):

    def run(self):
        env_generator = lambda : \
            SingleAgentGymWrapper(gym.make('BipedalWalkerHardcore-v3',
                render_mode = self.get_gym_render_mode()))

        actor_kw_args = {}
        actor_kw_args["std_offset"]  = 0.1
        actor_kw_args["activation"]  = nn.LeakyReLU()
        actor_kw_args["hidden_size"] = 256

        critic_kw_args = actor_kw_args.copy()
        critic_kw_args["hidden_size"] = 512

        #
        # This environment is a pretty challenging one and can be
        # very finicky. Learning rate and reward clipping have a
        # pretty powerfull impact on results, and it can be tricky
        # to get these right.
        #
        lr = LinearStepScheduler(
            status_key      = "iteration",
            initial_value   = 0.0001,
            status_triggers = [3900,],
            step_values     = [0.00001,])

        reward_clip_min = LinearStepScheduler(
            status_key      = "iteration",
            initial_value   = -1.,
            status_triggers = [4000,],
            step_values     = [-10.,])

        policy_args = {\
            "ac_network"       : FeedForwardNetwork,
            "actor_kw_args"    : actor_kw_args,
            "critic_kw_args"   : critic_kw_args,
            "lr"               : lr,
        }

        policy_settings, policy_mapping_fn = get_single_policy_defaults(
            env_generator = env_generator,
            policy_args   = policy_args)

        ts_per_rollout = self.get_adjusted_ts_per_rollout(512)

        self.run_ppo(env_generator      = env_generator,
                     policy_settings    = policy_settings,
                     policy_mapping_fn  = policy_mapping_fn,
                     batch_size         = 512,
                     max_ts_per_ep      = 32,
                     ts_per_rollout     = ts_per_rollout,
                     normalize_obs      = True,
                     normalize_rewards  = True,
                     obs_clip           = (-10., 10.),
                     reward_clip        = (reward_clip_min, 10.),
                     **self.kw_run_args)
