import numpy as np
import os
import torch
from copy import deepcopy
from functools import reduce
from torch.optim import Adam
from ppo_and_friends.utils.episode_info import EpisodeInfo, PPODataset
from ppo_and_friends.networks.ppo_networks.icm import ICM
from ppo_and_friends.utils.mpi_utils import rank_print
from ppo_and_friends.utils.misc import get_action_dtype
from gymnasium.spaces import Box, Discrete, MultiDiscrete, MultiBinary
from ppo_and_friends.utils.mpi_utils import broadcast_model_parameters
from ppo_and_friends.utils.misc import update_optimizer_lr
from ppo_and_friends.networks.ppo_networks.feed_forward import FeedForwardNetwork
from ppo_and_friends.networks.actor_critic.wrappers import to_actor, to_critic
from ppo_and_friends.utils.schedulers import LinearScheduler, CallableValue

from mpi4py import MPI
comm      = MPI.COMM_WORLD
rank      = comm.Get_rank()
num_procs = comm.Get_size()

class AgentPolicy():
    """
        A class representing a policy. A policy can be
        used by more than one agent, and more than one policy
        can exist in a given learning environment.
    """

    def __init__(self,
                 name,
                 action_space,
                 actor_observation_space,
                 critic_observation_space,
                 ac_network          = FeedForwardNetwork,
                 actor_kw_args       = {},
                 critic_kw_args      = {},
                 icm_kw_args         = {},
                 target_kl           = 100.,
                 surr_clip           = 0.2,
                 vf_clip             = None,
                 gradient_clip       = 0.5,
                 lr                  = 3e-4,
                 icm_lr              = 3e-4,
                 entropy_weight      = 0.01,
                 kl_loss_weight      = 0.0,
                 use_gae             = True,
                 gamma               = 0.99,
                 lambd               = 0.95,
                 enable_icm          = False,
                 icm_network         = ICM,
                 intr_reward_weight  = 1.0,
                 icm_beta            = 0.8,
                 test_mode           = False):
        """
            Arguments:
                 name                 The name of this policy.
                 action_space         The action space of this policy.

                 actor_observation_space   The actor's observation space.
                 critic_observation-space  The critic's observation space.

                 ac_network           The type of network to use for the actor
                                      and critic.
                 actor_kw_args        Keyword arguments for the actor network.
                 critic_kw_args       Keyword arguments for the critic network.
                 icm_kw_args          Keyword arguments for the icm network.
                 target_kl            KL divergence used for early stopping.
                                      This is typically set in the range
                                      [0.1, 0.5]. Use high values to disable.
                 surr_clip            The clip value applied to the surrogate
                                      (standard PPO approach).
                 vf_clip              An optional clip parameter used for
                                      clipping the value function loss.
                 gradient_clip        An optional clip value to use on the
                                      gradient update.
                 lr                   The learning rate. Can be
                                      a number or a scheduler class from
                                      utils/schedulers.py.
                 icm_lr               The learning rate. Can be
                                      a number or a scheduler class from
                                      utils/schedulers.py.
                 entropy_weight       The entropy weight. Can be
                                      a number or a scheduler class from
                                      utils/schedulers.py.
                 kl_loss_weight       A "kl coefficient" when adding kl
                                      divergence to the actor's loss. This
                                      is only used when > 0.0, and is off
                                      by default.
                 use_gae              Should we use Generalized Advantage
                                      Estimations? If not, fall back on the
                                      vanilla advantage calculation.
                 gamma                The gamma parameter used in calculating
                                      rewards to go.
                 lambd                The 'lambda' value for calculating GAEs.
                 enable_icm           Boolean flag. Enable ICM?
                 icm_network          The network to use for ICM applications.
                 intr_reward_weight   When using ICM, this weight will be
                                      applied to the intrinsic reward.
                                      Can be a number or a class from
                                      utils/schedulers.py.
                 icm_beta             The beta value used within the ICM.
                 test_mode            Boolean flag. Are we in test mode?
        """
        self.name               = name
        self.action_space       = action_space
        self.actor_obs_space    = actor_observation_space
        self.critic_obs_space   = critic_observation_space
        self.enable_icm         = enable_icm
        self.test_mode          = test_mode
        self.use_gae            = use_gae
        self.gamma              = gamma
        self.lambd              = lambd
        self.using_lstm         = False
        self.dataset            = None
        self.device             = torch.device("cpu")
        self.agent_ids          = set()
        self.episodes           = {}
        self.icm_beta           = icm_beta
        self.target_kl          = target_kl
        self.surr_clip          = surr_clip
        self.vf_clip            = vf_clip
        self.gradient_clip      = gradient_clip
        self.kl_loss_weight     = kl_loss_weight

        if callable(lr):
            self.lr = lr
        else:
            self.lr = CallableValue(lr)

        if callable(icm_lr):
            self.icm_lr = icm_lr
        else:
            self.icm_lr = CallableValue(icm_lr)

        if callable(entropy_weight):
            self.entropy_weight = entropy_weight
        else:
            self.entropy_weight = CallableValue(entropy_weight)

        if callable(intr_reward_weight):
            self.intr_reward_weight = intr_reward_weight
        else:
            self.intr_reward_weight = CallableValue(intr_reward_weight)

        self.action_dtype = get_action_dtype(self.action_space)

        if self.action_dtype == "unknown":
            msg  = "ERROR: unknown action type: "
            msg += f"{type(self.action_space)} with dtype "
            msg += f"{self.action_space.dtype}."
            rank_print(msg)
            comm.Abort()
        else:
            rank_print("{} policy using {} actions.".format(
                self.name, self.action_dtype))

        self._initialize_networks(
            ac_network     = ac_network,
            enable_icm     = enable_icm,
            icm_network    = icm_network,
            actor_kw_args  = actor_kw_args,
            critic_kw_args = critic_kw_args,
            icm_kw_args    = icm_kw_args)

    def finalize(self, status_dict):
        """
            Perfrom any finalizing tasks before we start using the policy.

            Arguments:
                status_dict    The status dict for training.
        """
        self.lr.finalize(status_dict)
        self.icm_lr.finalize(status_dict)
        self.entropy_weight.finalize(status_dict)
        self.intr_reward_weight.finalize(status_dict)

        self.actor_optim  = Adam(
            self.actor.parameters(), lr=self.lr(), eps=1e-5)
        self.critic_optim = Adam(
            self.critic.parameters(), lr=self.lr(), eps=1e-5)

        if self.enable_icm:
            self.icm_optim = Adam(self.icm_model.parameters(),
                lr=self.icm_lr(), eps=1e-5)
        else:
            self.icm_optim = None


    def register_agent(self, agent_id):
        """
            Register an agent with this policy.

            Arguments:
                agent_id    The id of the agent to register.
        """
        self.agent_ids = self.agent_ids.union({agent_id})

    def to(self, device):
        """
            Send this policy to a specified device.

            Arguments:
                device    The device to send this policy to.
        """
        self.device    = device
        self.actor     = self.actor.to(self.device)
        self.critic    = self.critic.to(self.device)

        if self.enable_icm:
            self.icm_model = self.icm_model.to(self.device)

    def _initialize_networks(
        self,
        ac_network, 
        enable_icm,
        icm_network,
        actor_kw_args,
        critic_kw_args,
        icm_kw_args):
        """
            Initialize our networks.

            Arguments:
                ac_network        The network class to use for the actor
                                  and critic.
                enable_icm        Whether or not to enable ICM.
                icm_network       The network class to use for ICM (when
                                  enabled).
                actor_kw_args     Keyword args for the actor network.
                critic_kw_args    Keyword args for the critic network.
                icm_kw_args       Keyword args for the ICM network.
        """
        #
        # Initialize our networks: actor, critic, and possibly ICM.
        #
        for base in ac_network.__bases__:
            if base.__name__ == "PPOLSTMNetwork":
                self.using_lstm = True

        #
        # arXiv:2006.05990v1 suggests initializing the output layer
        # of the actor network with a weight that's ~100x smaller
        # than the rest of the layers. We initialize layers with a
        # value near 1.0 by default, so we set the last layer to
        # 0.01. The same paper also suggests that the last layer of
        # the value network doesn't matter so much. I can't remember
        # where I got 1.0 from... I'll try to track that down.
        #
        self.actor = to_actor(ac_network)(
            name         = "actor", 
            obs_space    = self.actor_obs_space,
            out_init     = 0.01,
            action_space = self.action_space,
            test_mode    = self.test_mode,
            **actor_kw_args)

        self.critic = to_critic(ac_network)(
            name         = "critic", 
            obs_space    = self.critic_obs_space,
            out_init     = 1.0,
            test_mode    = self.test_mode,
            **critic_kw_args)

        self.actor  = self.actor.to(self.device)
        self.critic = self.critic.to(self.device)

        broadcast_model_parameters(self.actor)
        broadcast_model_parameters(self.critic)
        comm.barrier()

        if enable_icm:
            self.icm_model = icm_network(
                name         = "icm",
                obs_space    = self.actor_obs_space,
                action_space = self.action_space,
                test_mode    = self.test_mode,
                **icm_kw_args)

            self.icm_model = self.icm_model.to(self.device)
            broadcast_model_parameters(self.icm_model)
            comm.barrier()

    def initialize_episodes(self, env_batch_size, status_dict):
        """
            Initialize episodes for rollout collection. This should be called
            at the start of a rollout.

            Arguments:
                env_batch_size    The number of environments per processor.
                status_dict       The status dictionary.
        """
        #
        # NOTE that different agents will map to different polices, meaning
        # that our dictionaries can be different sizes for each policy, but
        # the number of environment instances will be consistent.
        #
        self.episodes = {}
        for agent_id in self.agent_ids:
            self.episodes[agent_id] = np.array([None] * env_batch_size,
                dtype=object)

            for ei_idx in range(env_batch_size):
                self.episodes[agent_id][ei_idx] = EpisodeInfo(
                    starting_ts    = 0,
                    use_gae        = self.use_gae,
                    gamma          = self.gamma,
                    lambd          = self.lambd)

    def initialize_dataset(self):
        """
            Initialize a rollout dataset. This should be called at the
            onset of a rollout.
        """
        sequence_length = 1
        if self.using_lstm:
            self.actor.reset_hidden_state(
                batch_size = 1,
                device     = self.device)

            self.critic.reset_hidden_state(
                batch_size = 1,
                device     = self.device)

            sequence_length = self.actor.sequence_length

        self.dataset = PPODataset(
            device          = self.device,
            action_dtype    = self.action_dtype,
            sequence_length = sequence_length)

    def validate_agent_id(self, agent_id):
        """
            Assert that a given agent id is associated with this policy.
            This will Abort if the id is invalid.

            Arguments:
                agent_id    The agent id in question.
        """
        if agent_id not in self.agent_ids:
            msg  = f"ERROR: agent {agent_id} has not been registered with "
            msg += "policy {self.name}. Make sure that you've set up your "
            msg += "policies correctly."
            rank_print(msg)
            comm.Abort()

    def add_episode_info(
        self, 
        agent_id,
        critic_observations,
        observations,
        next_observations,
        raw_actions, 
        actions, 
        values, 
        log_probs, 
        rewards, 
        where_done):
        """
            Log information about a single step in an episode during a rollout.
            Note that our observaionts, etc, will be batched across environment
            instances, so we can have multiple observations for a single step.

            Arguments:
                agent_id             The agent id that this episode info is
                                     associated with.
                critic_observations  The critic observation(s).
                observations         The actor observation(s).
                next_observations    The actor observation(s.
                raw_actions          The raw action(s).
                actions              The actions(s) taken in the environment.
                values               The value(s) from our critic.
                log_probs            The log_prob(s) of our action distribution.
                rewards              The reward(s) received.
                where_done           Indicies mapping to which environments are
                                     done.
        """
        self.validate_agent_id(agent_id)
        env_batch_size = self.episodes[agent_id].size

        #
        # When using lstm networks, we need to save the hidden states
        # encountered during the rollouts. These will later be used to
        # initialize the hidden states when updating the models.
        # Note that we pass in empty arrays when not using lstm networks.
        #
        if self.using_lstm:

            actor_hidden  = self.actor.hidden_state[0].clone()
            actor_cell    = self.actor.hidden_state[1].clone()

            critic_hidden = self.critic.hidden_state[0].clone()
            critic_cell   = self.critic.hidden_state[1].clone()

            if where_done.size > 0:
                actor_zero_hidden, actor_zero_cell = \
                    self.actor.get_zero_hidden_state(
                        batch_size = env_batch_size,
                        device     = self.device)

                actor_hidden[:, where_done, :] = \
                    actor_zero_hidden[:, where_done, :]

                actor_cell[:, where_done, :] = \
                    actor_zero_cell[:, where_done, :]

                critic_zero_hidden, critic_zero_cell = \
                    self.critic.get_zero_hidden_state(
                        batch_size = env_batch_size,
                        device     = self.device)

                critic_hidden[:, where_done, :] = \
                    critic_zero_hidden[:, where_done, :]

                critic_cell[:, where_done, :] = \
                    critic_zero_cell[:, where_done, :]

        else:
            empty_shape = (0, env_batch_size, 0)

            actor_hidden, actor_cell, critic_hidden, critic_cell  = \
                (np.empty(empty_shape),
                 np.empty(empty_shape),
                 np.empty(empty_shape),
                 np.empty(empty_shape))

        for ei_idx in range(env_batch_size):
            self.episodes[agent_id][ei_idx].add_info(
                critic_observation = critic_observations[ei_idx],
                observation        = observations[ei_idx],
                next_observation   = next_observations[ei_idx],
                raw_action         = raw_actions[ei_idx],
                action             = actions[ei_idx],
                value              = values[ei_idx].item(),
                log_prob           = log_probs[ei_idx],
                reward             = rewards[ei_idx].item(),
                actor_hidden       = actor_hidden[:, [ei_idx], :],
                actor_cell         = actor_cell[:, [ei_idx], :],
                critic_hidden      = critic_hidden[:, [ei_idx], :],
                critic_cell        = critic_cell[:, [ei_idx], :])

    def end_episodes(
        self,
        agent_id,
        env_idxs,
        episode_lengths,
        terminal,
        ending_values,
        ending_rewards,
        status_dict):
        """
            End a rollout episode.

            Arguments:
                agent_id         The associated agent id.
                env_idxs         The associated environment indices.
                episode_lengths  The lenghts of the ending episode(s).
                terminal         Which episodes are terminally ending.
                ending_values    Ending values for the episode(s).
                ending_rewards   Ending rewards for the episode(s)
                status_dict      The status dictionary.
        """
        self.validate_agent_id(agent_id)

        for idx, env_i in enumerate(env_idxs):
            self.episodes[agent_id][env_i].end_episode(
                ending_ts      = episode_lengths[env_i],
                terminal       = terminal[idx],
                ending_value   = ending_values[idx].item(),
                ending_reward  = ending_rewards[idx].item())

            self.dataset.add_episode(self.episodes[agent_id][env_i])

            #
            # If we're terminal, the start of the next episode is 0.
            # Otherwise, we pick up where we left off.
            #
            starting_ts = 0 if terminal[idx] else episode_lengths[env_i]

            self.episodes[agent_id][env_i] = EpisodeInfo(
                starting_ts    = starting_ts,
                use_gae        = self.use_gae,
                gamma          = self.gamma,
                lambd          = self.lambd)

    def finalize_dataset(self):
        """
            Build our rollout dataset. This should be called after
            the rollout has taken place and before training begins.
        """
        self.dataset.build()

    def clear_dataset(self):
        """
            Clear existing datasets. This should be called before
            a new rollout.
        """
        self.dataset  = None
        self.episodes = {}

    def get_training_actions(self, obs):
        """
            Given observations from our environment, determine what the
            next actions should be taken while allowing natural exploration.

            This method is explicitly meant to be used in training and will
            return more than just the environment actions.

            Arguments:
                obs    The environment observations.

            Returns:
                A tuple of form (raw_action, action, log_prob) s.t. "raw_action"
                is the distribution sample before any "squashing" takes place,
                "action" is the the action value that should be fed to the
                environment, and log_prob is the log probabilities from our
                probability distribution.
        """
        if len(obs.shape) < 2:
            msg  = "ERROR: _get_action_with_exploration expects a "
            msg ++ "batch of observations but "
            msg += "instead received shape {}.".format(obs.shape)
            rank_print(msg)
            comm.Abort()

        t_obs = torch.tensor(obs, dtype=torch.float).to(self.device)

        with torch.no_grad():
            action_pred = self.actor(t_obs)

        action_pred = action_pred.cpu().detach()
        dist        = self.actor.distribution.get_distribution(action_pred)

        #
        # Our distribution gives us two potentially distinct actions, one of
        # which is guaranteed to be a raw sample from the distribution. The
        # other might be altered in some way (usually to enforce a range).
        #
        action, raw_action = self.actor.distribution.sample_distribution(dist)
        log_prob = self.actor.distribution.get_log_probs(dist, raw_action)

        action     = action.detach().numpy()
        raw_action = raw_action.detach().numpy()

        return raw_action, action, log_prob.detach()

    def get_inference_actions(self, obs, explore):
        """
            Given observations from our environment, determine what the
            actions should be.

            This method is meant to be used for inference only, and it
            will return the environment actions alone.

            Arguments:
                obs       The environment observation.
                explore   Should we allow exploration?

            Returns:
                Predicted actions to perform in the environment.
        """
        if explore:
            return self._get_action_with_exploration(obs)
        return self._get_action_without_exploration(obs)

    def _get_action_with_exploration(self, obs):
        """
            Given observations from our environment, determine what the
            next actions should be taken while allowing natural exploration.

            Arguments:
                obs    The environment observations.

            Returns:
                A tuple of form (raw_action, action, log_prob) s.t. "raw_action"
                is the distribution sample before any "squashing" takes place,
                "action" is the the action value that should be fed to the
                environment, and log_prob is the log probabilities from our
                probability distribution.
        """
        if len(obs.shape) < 2:
            msg  = "ERROR: _get_action_with_exploration expects a "
            msg ++ "batch of observations but "
            msg += "instead received shape {}.".format(obs.shape)
            rank_print(msg)
            comm.Abort()

        t_obs = torch.tensor(obs, dtype=torch.float).to(self.device)

        with torch.no_grad():
            action_pred = self.actor(t_obs)

        action_pred = action_pred.cpu().detach()
        dist        = self.actor.distribution.get_distribution(action_pred)

        #
        # Our distribution gives us two potentially distinct actions, one of
        # which is guaranteed to be a raw sample from the distribution. The
        # other might be altered in some way (usually to enforce a range).
        #
        action, _ = self.actor.distribution.sample_distribution(dist)
        return action

    def _get_action_without_exploration(self, obs):
        """
            Given observations from our environment, determine what the
            next actions should be while not allowing any exploration.

            Arguments:
                obs    The environment observations.

            Returns:
                The next actions to perform.
        """
        if len(obs.shape) < 2:
            msg  = "ERROR: _get_action_without_exploration expects a "
            msg ++ "batch of observations but "
            msg += "instead received shape {}.".format(obs.shape)
            rank_print(msg)
            comm.Abort()

        t_obs = torch.tensor(obs, dtype=torch.float).to(self.device)

        with torch.no_grad():
            return self.actor.get_refined_prediction(t_obs)

    def evaluate(self, batch_critic_obs, batch_obs, batch_actions):
        """
            Given a batch of observations, use our critic to approximate
            the expected return values. Also use a batch of corresponding
            actions to retrieve some other useful information.

            Arguments:
                batch_critic_obs   A batch of observations for the critic.
                batch_obs          A batch of standard observations.
                batch_actions      A batch of actions corresponding to the batch of
                                   observations.

            Returns:
                A tuple of form (values, log_probs, entropies) s.t. values are
                the critic predicted value, log_probs are the log probabilities
                from our probability distribution, and entropies are the
                entropies from our distribution.
        """
        values      = self.critic(batch_critic_obs).squeeze()
        action_pred = self.actor(batch_obs).cpu()
        dist        = self.actor.distribution.get_distribution(action_pred)

        if self.action_dtype == "continuous" and len(batch_actions.shape) < 2:
            log_probs = self.actor.distribution.get_log_probs(
                dist,
                batch_actions.unsqueeze(1).cpu())
        else:
            log_probs = self.actor.distribution.get_log_probs(
                dist,
                batch_actions.cpu())

        entropy = self.actor.distribution.get_entropy(dist, action_pred)

        return values, log_probs.to(self.device), entropy.to(self.device)

    def get_intrinsic_reward(self,
                             prev_obs,
                             obs,
                             action):
        """
            Query the ICM for an intrinsic reward.

            Arguments:
                prev_obs    The previous observation (before the latest
                            action).
                obs         The current observation.
                action      The action taken.
        """
        if len(obs.shape) < 2:
            msg  = "ERROR: get_intrinsic_reward expects a batch of "
            msg += "observations but "
            msg += "instead received shape {}.".format(obs.shape)
            rank_print(msg)
            comm.Abort()

        obs_1 = torch.tensor(prev_obs,
            dtype=torch.float).to(self.device)
        obs_2 = torch.tensor(obs,
            dtype=torch.float).to(self.device)

        if self.action_dtype in ["discrete", "multi-discrete"]:
            action = torch.tensor(action,
                dtype=torch.long).to(self.device)

        elif self.action_dtype in ["continuous", "multi-binary"]:
            action = torch.tensor(action,
                dtype=torch.float).to(self.device)

        if len(action.shape) != 2:
            action = action.unsqueeze(1)

        with torch.no_grad():
            intr_reward, _, _ = self.icm_model(obs_1, obs_2, action)

        batch_size   = obs.shape[0]
        intr_reward  = intr_reward.detach().cpu().numpy()
        intr_reward  = intr_reward.reshape((batch_size, -1))
        intr_reward *= self.intr_reward_weight()

        return intr_reward

    def update_learning_rate(self):
        """
            Update the learning rate.
        """
        update_optimizer_lr(self.actor_optim, self.lr())
        update_optimizer_lr(self.critic_optim, self.lr())

        if self.enable_icm:
            update_optimizer_lr(self.icm_optim, self.icm_lr())

    def save(self, save_path):
        """
            Save our policy.

            Arguments:
                save_path    The path to save the policy to.
        """
        policy_dir = "{}-policy".format(self.name)
        policy_save_path = os.path.join(save_path, policy_dir)

        if rank == 0 and not os.path.exists(policy_save_path):
            os.makedirs(policy_save_path)

        comm.barrier()

        self.actor.save(policy_save_path)
        self.critic.save(policy_save_path)

        if self.enable_icm:
            self.icm_model.save(policy_save_path)

    def load(self, load_path):
        """
            Load our policy.

            Arguments:
                load_path    The path to load the policy from.
        """
        policy_dir = "{}-policy".format(self.name)
        policy_load_path = os.path.join(load_path, policy_dir)

        self.actor.load(policy_load_path)
        self.critic.load(policy_load_path)

        if self.enable_icm:
            self.icm_model.load(policy_load_path)

    def eval(self):
        """
            Set the policy to evaluation mode.
        """
        self.actor.eval()
        self.critic.eval()

        if self.enable_icm:
            self.icm_model.eval()

    def train(self):
        """
            Set the policy to train mode.
        """
        self.actor.train()
        self.critic.train()

        if self.enable_icm:
            self.icm_model.train()

    def __getstate__(self):
        """
            Override the getstate method for pickling. We only want to keep
            things that won't upset pickle. The environment is something
            that we can't guarantee can be pickled.

            Returns:
                The state dictionary minus the environment.
        """
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        """
            Override the setstate method for pickling.

            Arguments:
                The state loaded from a pickled PPO object.
        """
        self.__dict__.update(state)

    def __eq__(self, other):
        """
            Compare two policies.

            Arguments:
                other        An instance of AgentPolicy.
        """
        #
        # TODO: we currently don't compare optimizers because that
        # requires extra effort, and our current implementation will
        # enforce they're equal when the learning rate is equal.
        # We should update this at some point.
        #
        # FIXME: bootstrap clip is difficult to compare without using
        # functions that define __eq__, so we're skipping it.
        #
        is_equal = (
            isinstance(other, AgentPolicy)
            and self.action_space       == other.action_space
            and self.actor_obs_space    == other.actor_obs_space
            and self.critic_obs_space   == other.critic_obs_space
            and self.enable_icm         == other.enable_icm
            and self.test_mode          == other.test_mode
            and self.use_gae            == other.use_gae
            and self.gamma              == other.gamma
            and self.lambd              == other.lambd
            and self.using_lstm         == other.using_lstm
            and self.action_dtype       == other.action_dtype)

        return is_equal

    def __str__(self):
        """
            A string representation of the policy.
        """
        str_self  = "AgentPolicy:\n"
        str_self += "    action space: {}\n".format(self.action_space)
        str_self += "    actor obs space: {}\n".format(self.actor_obs_space)
        str_self += "    critic obs space: {}\n".format(self.critic_obs_space)
        str_self += "    enable icm: {}\n".format(self.enable_icm)
        str_self += "    intr reward weight: {}\n".format(self.intr_reward_weight)
        str_self += "    lr: {}\n".format(self.lr)
        str_self += "    icm_lr: {}\n".format(self.icm_lr)
        str_self += "    entropy_weight: {}\n".format(self.entropy_weight)
        str_self += "    test mode: {}\n".format(self.test_mode)
        str_self += "    use gae: {}\n".format(self.use_gae)
        str_self += "    gamma: {}\n".format(self.gamma)
        str_self += "    lambd: {}\n".format(self.lambd)
        str_self += "    using lstm: {}\n".format(self.using_lstm)
        str_self += "    action dtype: {}\n".format(self.action_dtype)
        str_self += "    actor optim: {}\n".format(self.actor_optim)
        str_self += "    critic optim: {}\n".format(self.critic_optim)
        str_self += "    icm optim: {}\n".format(self.icm_optim)
        return str_self
