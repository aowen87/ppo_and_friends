from pettingzoo.mpe import simple_adversary_v3
from ppo_and_friends.policies.utils import get_single_policy_defaults
from ppo_and_friends.environments.petting_zoo.wrappers import PPOParallelZooWrapper
from ppo_and_friends.runners.env_runner import GymRunner
from ppo_and_friends.networks.actor_critic_networks import FeedForwardNetwork
from ppo_and_friends.utils.schedulers import *
import torch.nn as nn

class MPESimpleAdversaryRunner(GymRunner):

    def run(self):

        policy_map = lambda name : 'adversary' if 'adversary' in name \
            else 'agent'

        env_generator = lambda : \
            PPOParallelZooWrapper(
                simple_adversary_v3.parallel_env(
                    N=2,
                    max_cycles=25,
                    continuous_actions=False,
                    render_mode = self.get_gym_render_mode()),
                #
                # Each agent observes all other agents in this
                # scenario.
                #
                critic_view       = "local",
                policy_mapping_fn = policy_map)

        #
        # Extra args for the actor critic models.
        # I find that leaky relu does much better with the lunar
        # lander env.
        #
        actor_kw_args = {}

        actor_kw_args["activation"]  = nn.LeakyReLU()
        actor_kw_args["hidden_size"] = 64

        critic_kw_args = actor_kw_args.copy()
        critic_kw_args["hidden_size"] = 128

        critic_kw_args = actor_kw_args.copy()

        lr = 0.0003

        #
        # Running with 2 processors works well here.
        #
        ts_per_rollout = self.get_adjusted_ts_per_rollout(128)

        policy_args = {\
            "ac_network"       : FeedForwardNetwork,
            "actor_kw_args"    : actor_kw_args,
            "critic_kw_args"   : critic_kw_args,
            "lr"               : lr,
        }

        policy_settings = { 
            "agent" : \
                (None,
                 env_generator().observation_space["agent_0"],
                 env_generator().critic_observation_space["agent_0"],
                 env_generator().action_space["agent_0"],
                 policy_args),
            "adversary" : \
                (None,
                 env_generator().observation_space["adversary_0"],
                 env_generator().critic_observation_space["adversary_0"],
                 env_generator().action_space["adversary_0"],
                 policy_args),
        }

        self.run_ppo(env_generator       = env_generator,
                     policy_settings     = policy_settings,
                     policy_mapping_fn   = policy_map,
                     max_ts_per_ep       = 16,
                     ts_per_rollout      = ts_per_rollout,
                     batch_size          = 32,
                     normalize_obs       = True,
                     normalize_rewards   = True,
                     obs_clip            = (-10., 10.),
                     reward_clip         = (-10., 10.),
                     **self.kw_run_args)