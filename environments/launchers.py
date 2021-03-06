"""
    A home for environment "launchers", defined as simple functions
    that initialize training for a specific environment.
"""
import gym
from ppo_and_friends.ppo import PPO
from ppo_and_friends.testing import test_policy
from ppo_and_friends.networks.actor_critic_networks import FeedForwardNetwork, AtariPixelNetwork
from ppo_and_friends.networks.actor_critic_networks import SplitObsNetwork
from ppo_and_friends.networks.actor_critic_networks import LSTMNetwork
from ppo_and_friends.networks.icm import ICM
from ppo_and_friends.networks.encoders import LinearObservationEncoder
from .gym_wrappers import *
import torch.nn as nn
from ppo_and_friends.utils.iteration_mappers import *
from mpi4py import MPI

comm      = MPI.COMM_WORLD
rank      = comm.Get_rank()
num_procs = comm.Get_size()


def run_ppo(env_generator,
            ac_network,
            device,
            random_seed,
            is_multi_agent      = False,
            envs_per_proc       = 1,
            icm_network         = ICM,
            batch_size          = 256,
            ts_per_rollout      = num_procs * 1024,
            epochs_per_iter     = 10,
            target_kl           = 100.,
            lr                  = 3e-4,
            min_lr              = 1e-4,
            lr_dec              = None,
            entropy_weight      = 0.01,
            min_entropy_weight  = 0.01,
            entropy_dec         = None,
            max_ts_per_ep       = 200,
            use_gae             = True,
            use_icm             = False,
            save_best_only      = False,
            icm_beta            = 0.8,
            ext_reward_weight   = 1.0,
            intr_reward_weight  = 1.0,
            actor_kw_args       = {},
            critic_kw_args      = {},
            icm_kw_args         = {},
            gamma               = 0.99,
            lambd               = 0.95,
            surr_clip           = 0.2,
            bootstrap_clip      = (-10.0, 10.0),
            dynamic_bs_clip     = False,
            gradient_clip       = 0.5,
            mean_window_size    = 100,
            normalize_adv       = True,
            normalize_obs       = True,
            normalize_rewards   = True,
            normalize_values    = True,
            obs_clip            = None,
            reward_clip         = None,
            render              = False,
            render_gif          = False,
            load_state          = False,
            state_path          = "./",
            num_timesteps       = 1,
            test                = False,
            pickle_class        = False,
            use_soft_resets     = True,
            obs_augment         = False,
            num_test_runs       = 1):

    ppo = PPO(env_generator      = env_generator,
              ac_network         = ac_network,
              icm_network        = icm_network,
              device             = device,
              is_multi_agent     = is_multi_agent,
              random_seed        = random_seed,
              batch_size         = batch_size,
              envs_per_proc      = envs_per_proc,
              ts_per_rollout     = ts_per_rollout,
              lr                 = lr,
              target_kl          = target_kl,
              min_lr             = min_lr,
              lr_dec             = lr_dec,
              max_ts_per_ep      = max_ts_per_ep,
              use_gae            = use_gae,
              use_icm            = use_icm,
              save_best_only     = save_best_only,
              ext_reward_weight  = ext_reward_weight,
              intr_reward_weight = intr_reward_weight,
              entropy_weight     = entropy_weight,
              min_entropy_weight = min_entropy_weight,
              entropy_dec        = entropy_dec,
              icm_kw_args        = icm_kw_args,
              actor_kw_args      = actor_kw_args,
              critic_kw_args     = critic_kw_args,
              gamma              = gamma,
              lambd              = lambd,
              surr_clip          = surr_clip,
              bootstrap_clip     = bootstrap_clip,
              dynamic_bs_clip    = dynamic_bs_clip,
              gradient_clip      = gradient_clip,
              normalize_adv      = normalize_adv,
              normalize_obs      = normalize_obs,
              normalize_rewards  = normalize_rewards,
              normalize_values   = normalize_values,
              obs_clip           = obs_clip,
              reward_clip        = reward_clip,
              mean_window_size   = mean_window_size,
              render             = render,
              load_state         = load_state,
              state_path         = state_path,
              pickle_class       = pickle_class,
              use_soft_resets    = use_soft_resets,
              obs_augment        = obs_augment,
              test_mode          = test)

    if test:
        test_policy(ppo,
                    render_gif,
                    num_test_runs,
                    device)
    else: 
        ppo.learn(num_timesteps)


###############################################################################
#                            Classic Control                                  #
###############################################################################

def cartpole_ppo(state_path,
                 load_state,
                 render,
                 render_gif,
                 num_timesteps,
                 device,
                 envs_per_proc,
                 random_seed,
                 test = False,
                 num_test_runs = 1):

    env_generator = lambda : gym.make('CartPole-v0')

    actor_kw_args = {}
    actor_kw_args["activation"]  = nn.LeakyReLU()
    critic_kw_args = actor_kw_args.copy()

    lr     = 0.0002
    min_lr = 0.0002

    lr_dec = LinearDecrementer(
        max_iteration  = 1,
        max_value      = lr,
        min_value      = min_lr)

    ts_per_rollout = num_procs * 256

    run_ppo(env_generator      = env_generator,
            random_seed        = random_seed,
            batch_size         = 256,
            actor_kw_args      = actor_kw_args,
            critic_kw_args     = critic_kw_args,
            ac_network         = FeedForwardNetwork,
            ts_per_rollout     = ts_per_rollout,
            max_ts_per_ep      = 32,
            use_gae            = True,
            normalize_obs      = True,
            normalize_rewards  = True,
            normalize_adv      = True,
            obs_clip           = (-10., 10.),
	    reward_clip        = (-10., 10.),
            state_path         = state_path,
            load_state         = load_state,
            render             = render,
            render_gif         = render_gif,
            num_timesteps      = num_timesteps,
            device             = device,
            envs_per_proc      = envs_per_proc,
            lr                 = lr,
            min_lr             = min_lr,
            lr_dec             = lr_dec,
            test               = test,
            num_test_runs      = num_test_runs)


def pendulum_ppo(state_path,
                 load_state,
                 render,
                 render_gif,
                 num_timesteps,
                 device,
                 envs_per_proc,
                 random_seed,
                 test = False,
                 num_test_runs = 1):

    env_generator = lambda : gym.make('Pendulum-v1')

    actor_kw_args = {}
    actor_kw_args["activation"]  = nn.LeakyReLU()
    actor_kw_args["hidden_size"] = 32

    critic_kw_args = actor_kw_args.copy()
    critic_kw_args["hidden_size"] = 256

    lr     = 0.0003
    min_lr = 0.0003

    lr_dec = LinearDecrementer(
        max_iteration = 1000,
        max_value     = lr,
        min_value     = min_lr)

    ts_per_rollout = num_procs * 512

    run_ppo(env_generator      = env_generator,
            random_seed        = random_seed,
            ac_network         = FeedForwardNetwork,
            actor_kw_args      = actor_kw_args,
            critic_kw_args     = critic_kw_args,
            max_ts_per_ep      = 32,
            ts_per_rollout     = ts_per_rollout,
            use_gae            = True,
            normalize_obs      = True,
            normalize_rewards  = True,
            dynamic_bs_clip    = True,
            obs_clip           = (-10., 10.),
            reward_clip        = (-10., 10.),
            state_path         = state_path,
            load_state         = load_state,
            render             = render,
            render_gif         = render_gif,
            num_timesteps      = num_timesteps,
            device             = device,
            envs_per_proc      = envs_per_proc,
            lr                 = lr,
            min_lr             = min_lr,
            lr_dec             = lr_dec,
            test               = test,
            num_test_runs      = num_test_runs)


def mountain_car_ppo(state_path,
                     load_state,
                     render,
                     render_gif,
                     num_timesteps,
                     device,
                     envs_per_proc,
                     random_seed,
                     test = False,
                     num_test_runs = 1):

    env_generator = lambda : gym.make('MountainCar-v0')

    actor_kw_args = {"activation" : nn.LeakyReLU()}
    actor_kw_args["hidden_size"] = 128

    critic_kw_args = actor_kw_args.copy()
    critic_kw_args["hidden_size"] = 128

    lr     = 0.0003
    min_lr = 0.0001

    lr_dec = LinearStepMapper(
        step_type    = "iteration",
        steps        = [200,],
        step_values  = [0.0003,],
        ending_value = 0.0001)

    #
    # NOTE: This environment performs dramatically  better when
    # max_ts_per_ep is set to the total timesteps allowed by the
    # environment. It's not 100% clear to me why this is the case.
    # We should probably explore this a bit more. MountainCarContinuous
    # doesn't seem to exhibit this behavior, so it's unlikely an issue
    # with ICM.
    # Also, the extrinsic reward weight fraction is very important
    # for good performance.
    #
    run_ppo(env_generator      = env_generator,
            random_seed        = random_seed,
            ac_network         = FeedForwardNetwork,
            actor_kw_args      = actor_kw_args,
            critic_kw_args     = critic_kw_args,
            dynamic_bs_clip    = True,
            max_ts_per_ep      = 200,
            ext_reward_weight  = 1./100.,
            lr_dec             = lr_dec,
            lr                 = lr,
            min_lr             = min_lr,
            use_icm            = True,
            use_gae            = True,
            normalize_obs      = False,
            normalize_rewards  = False,
            normalize_values   = False,
            obs_clip           = None,
            reward_clip        = None,
            bootstrap_clip     = (-10, 10),
            state_path         = state_path,
            load_state         = load_state,
            render             = render,
            render_gif         = render_gif,
            num_timesteps      = num_timesteps,
            device             = device,
            envs_per_proc      = envs_per_proc,
            test               = test,
            num_test_runs      = num_test_runs)


def mountain_car_continuous_ppo(state_path,
                                load_state,
                                render,
                                render_gif,
                                num_timesteps,
                                device,
                                envs_per_proc,
                                random_seed,
                                test = False,
                                num_test_runs = 1):

    env_generator = lambda : gym.make('MountainCarContinuous-v0')

    #
    # Extra args for the actor critic models.
    #
    actor_kw_args = {}
    actor_kw_args["activation"]  =  nn.LeakyReLU()
    actor_kw_args["hidden_size"] = 64

    critic_kw_args = actor_kw_args.copy()
    critic_kw_args["hidden_size"] = 128

    lr     = 0.0003
    min_lr = 0.0003

    lr_dec = LinearDecrementer(
        max_iteration = 1,
        max_value     = lr,
        min_value     = min_lr)

    #
    # I've noticed that normalizing rewards and observations
    # can slow down learning at times. It's not by much (maybe
    # 10-50 iterations).
    #
    run_ppo(env_generator      = env_generator,
            random_seed        = random_seed,
            ac_network         = FeedForwardNetwork,
            max_ts_per_ep      = 128,
            batch_size         = 512,
            lr_dec             = lr_dec,
            lr                 = lr,
            min_lr             = min_lr,
            actor_kw_args      = actor_kw_args,
            critic_kw_args     = critic_kw_args,
            use_icm            = True,
            use_gae            = True,
            normalize_obs      = False,
            normalize_rewards  = False,
            normalize_values   = False,
            obs_clip           = None,
            reward_clip        = None,
            normalize_adv      = True,
            bootstrap_clip     = (-10., 10.),
            dynamic_bs_clip    = True,
            ext_reward_weight  = 1./100.,
            intr_reward_weight = 50.,
            state_path         = state_path,
            load_state         = load_state,
            render             = render,
            render_gif         = render_gif,
            num_timesteps      = num_timesteps,
            device             = device,
            envs_per_proc      = envs_per_proc,
            test               = test,
            num_test_runs      = num_test_runs)


def acrobot_ppo(state_path,
                load_state,
                render,
                render_gif,
                num_timesteps,
                device,
                envs_per_proc,
                random_seed,
                test = False,
                num_test_runs = 1):

    env_generator = lambda : gym.make('Acrobot-v1')

    actor_kw_args = {}
    actor_kw_args["hidden_size"] = 64

    critic_kw_args = {}
    critic_kw_args["hidden_size"] = 128

    lr     = 0.0003
    min_lr = 0.0

    lr_dec = LinearDecrementer(
        max_iteration = 2000,
        max_value     = lr,
        min_value     = min_lr)

    ts_per_rollout = num_procs * 512

    run_ppo(env_generator      = env_generator,
            random_seed        = random_seed,
            ac_network         = FeedForwardNetwork,
            max_ts_per_ep      = 32,
            ts_per_rollout     = ts_per_rollout,
            lr_dec             = lr_dec,
            lr                 = lr,
            min_lr             = min_lr,
            use_gae            = True,
            actor_kw_args      = actor_kw_args,
            critic_kw_args     = critic_kw_args,
            normalize_obs      = True,
            normalize_rewards  = True,
            obs_clip           = (-10., 10.),
            reward_clip        = (-10., 10.),
            bootstrap_clip     = (-10., 10.),
            state_path         = state_path,
            load_state         = load_state,
            render             = render,
            render_gif         = render_gif,
            num_timesteps      = num_timesteps,
            device             = device,
            envs_per_proc      = envs_per_proc,
            test               = test,
            num_test_runs      = num_test_runs)


###############################################################################
#                                Box 2D                                       #
###############################################################################

def lunar_lander_ppo(state_path,
                     load_state,
                     render,
                     render_gif,
                     num_timesteps,
                     device,
                     envs_per_proc,
                     random_seed,
                     test = False,
                     num_test_runs = 1):

    env_generator = lambda : gym.make('LunarLander-v2')

    #
    # Extra args for the actor critic models.
    # I find that leaky relu does much better with the lunar
    # lander env.
    #
    actor_kw_args = {}

    actor_kw_args["activation"]  = nn.LeakyReLU()
    actor_kw_args["hidden_size"] = 64

    critic_kw_args = actor_kw_args.copy()
    critic_kw_args["hidden_size"] = 256

    critic_kw_args = actor_kw_args.copy()

    lr     = 0.0003
    min_lr = 0.0001

    lr_dec = LinearDecrementer(
        max_iteration = 200,
        max_value     = lr,
        min_value     = min_lr)

    #
    # Running with 2 processors works well here.
    #
    ts_per_rollout = num_procs * 1024

    run_ppo(env_generator       = env_generator,
            random_seed         = random_seed,
            ac_network          = FeedForwardNetwork,
            max_ts_per_ep       = 128,
            ts_per_rollout      = ts_per_rollout,
            batch_size          = 512,
            use_gae             = True,
            normalize_obs       = True,
            normalize_rewards   = True,
            obs_clip            = (-10., 10.),
            reward_clip         = (-10., 10.),
            bootstrap_clip      = (-10., 10.),
            actor_kw_args       = actor_kw_args,
            critic_kw_args      = critic_kw_args,
            state_path          = state_path,
            load_state          = load_state,
            render              = render,
            render_gif          = render_gif,
            num_timesteps       = num_timesteps,
            device              = device,
            envs_per_proc       = envs_per_proc,
            lr_dec              = lr_dec,
            lr                  = lr,
            min_lr              = min_lr,
            test                = test,
            num_test_runs       = num_test_runs)


def lunar_lander_continuous_ppo(state_path,
                                load_state,
                                render,
                                render_gif,
                                num_timesteps,
                                device,
                                envs_per_proc,
                                random_seed,
                                test = False,
                                num_test_runs = 1):

    env_generator = lambda : gym.make('LunarLanderContinuous-v2')

    #
    # Lunar lander observations are organized as follows:
    #    Positions: 2
    #    Positional velocities: 2
    #    Angle: 1
    #    Angular velocities: 1
    #    Leg contact: 2
    #
    actor_kw_args = {}

    #
    # Extra args for the actor critic models.
    # I find that leaky relu does much better with the lunar
    # lander env.
    #
    actor_kw_args["activation"]  = nn.LeakyReLU()
    actor_kw_args["hidden_size"] = 64

    critic_kw_args = actor_kw_args.copy()
    critic_kw_args["hidden_size"] = 256

    lr     = 0.0003
    min_lr = 0.0001

    lr_dec = LinearDecrementer(
        max_iteration = 100,
        max_value     = lr,
        min_value     = min_lr)

    #
    # Running with 2 processors works well here.
    #
    ts_per_rollout = num_procs * 1024

    run_ppo(env_generator       = env_generator,
            random_seed         = random_seed,
            ac_network          = FeedForwardNetwork,
            max_ts_per_ep       = 32,
            ts_per_rollout      = ts_per_rollout,
            batch_size          = 512,
            actor_kw_args       = actor_kw_args,
            critic_kw_args      = critic_kw_args,
            use_gae             = True,
            normalize_obs       = True,
            normalize_rewards   = True,
            obs_clip            = (-10., 10.),
            reward_clip         = (-10., 10.),
            bootstrap_clip      = (-10., 10.),
            target_kl           = 0.015,
            state_path          = state_path,
            load_state          = load_state,
            render              = render,
            render_gif          = render_gif,
            num_timesteps       = num_timesteps,
            device              = device,
            envs_per_proc       = envs_per_proc,
            lr_dec              = lr_dec,
            lr                  = lr,
            min_lr              = min_lr,
            test                = test,
            num_test_runs       = num_test_runs)


def bipedal_walker_ppo(state_path,
                       load_state,
                       render,
                       render_gif,
                       num_timesteps,
                       device,
                       envs_per_proc,
                       random_seed,
                       test = False,
                       num_test_runs = 1):

    env_generator = lambda : gym.make('BipedalWalker-v3')

    #
    # The lidar observations are the last 10.
    #
    actor_kw_args = {}

    #
    # I've found that a lower std offset greatly improves performance
    # stability in this environment. Also, most papers suggest that using Tanh
    # provides the best performance, but I find that LeakyReLU works better
    # here.
    #
    actor_kw_args["std_offset"] = 0.1
    actor_kw_args["activation"] = nn.LeakyReLU()

    #
    # You can also use an LSTM or Split Observation network here,
    # but I've found that a simple MLP learns faster both in terms
    # of iterations and wall-clock time. The LSTM is the slowest
    # of the three options, which I would assume is related to the
    # fact that velocity information is already contained in the
    # observations, but it's a bit surprising that we can't infer
    # extra "history" information from the lidar.
    #
    actor_kw_args["hidden_size"] = 128

    critic_kw_args = actor_kw_args.copy()
    critic_kw_args["hidden_size"] = 256

    lr     = 0.0003
    min_lr = 0.0001

    lr_dec = LinearDecrementer(
        max_iteration = 200,
        max_value     = lr,
        min_value     = min_lr)

    ts_per_rollout = num_procs * 512

    #
    # Thresholding the reward to a low of -1 doesn't drastically
    # change learning, but it does help a bit. Clipping the bootstrap
    # reward to the same range seems to help with stability.
    #
    run_ppo(env_generator       = env_generator,
            random_seed         = random_seed,
            ac_network          = FeedForwardNetwork,
            actor_kw_args       = actor_kw_args,
            critic_kw_args      = critic_kw_args,
            batch_size          = 512,
            max_ts_per_ep       = 32,
            ts_per_rollout      = ts_per_rollout,
            use_gae             = True,
            normalize_adv       = True,
            normalize_obs       = True,
            normalize_rewards   = True,
            obs_clip            = (-10., 10.),
            reward_clip         = (-1., 10.),
            bootstrap_clip      = (-1., 10.),
            lr_dec              = lr_dec,
            lr                  = lr,
            min_lr              = min_lr,
            state_path          = state_path,
            load_state          = load_state,
            render              = render,
            render_gif          = render_gif,
            num_timesteps       = num_timesteps,
            device              = device,
            envs_per_proc       = envs_per_proc,
            test                = test,
            num_test_runs       = num_test_runs)


def bipedal_walker_hardcore_ppo(state_path,
                                load_state,
                                render,
                                render_gif,
                                num_timesteps,
                                device,
                                envs_per_proc,
                                random_seed,
                                test = False,
                                num_test_runs = 1):

    env_generator = lambda : gym.make('BipedalWalkerHardcore-v3')

    actor_kw_args = {}
    actor_kw_args["std_offset"]  = 0.1
    actor_kw_args["activation"]  = nn.LeakyReLU()
    actor_kw_args["hidden_size"] = 256

    critic_kw_args = actor_kw_args.copy()
    critic_kw_args["hidden_size"] = 512

    lr     = 0.0001
    min_lr = 0.00001

    #
    # This environment is a pretty challenging one and can be
    # very finicky. Learning rate and reward clipping have a
    # pretty powerfull impact on results, and it can be tricky
    # to get these right. Here's what I've found works best:
    #
    #   1. First, run the problem with a pretty standard learning
    #      rate (0.0001 works well), and use a conservative reward
    #      clipping of (-1, 10). Clipping the reward at -1 results
    #      in the agent learning a good gait pretty early on.
    #
    #   2. After a while (roughly 7000->8000 iterations), the agent should
    #      have a pretty solid policy. Running tests will show that
    #      it can regularly reach scores over 300, but averaging
    #      over 100 runs will likely be in the 200s. My impression
    #      here is that the -1 reward clipping, while allowing the
    #      agent to learn a good gait quickly, also causes the agent
    #      be less concerned with failures. So, at this point, I find
    #      that adjusting the lower bound of the clip to the standard
    #      -10 value allows the agent to learn that falling is
    #      actually really bad. I also lower the learning rate here
    #      to help with stability in this last phase. This last bit
    #      of learning can take a while (~10,000 -> 11,000 iterations).
    #
    # The above is all automated with the settings used below. I typically
    # run with 4 processors. The resulting policy can regularly reach average
    # scores of 320+ over 100 test runs.
    #
    lr_dec = LinearStepMapper(
        step_type    = "iteration",
        steps        = [3900,],
        step_values  = [0.0001,],
        ending_value = 0.00001)

    reward_clip_min = LinearStepMapper(
        step_type    = "iteration",
        steps        = [4000,],
        step_values  = [-1.,],
        ending_value = -10.)

    bs_clip_min = LinearStepMapper(
        step_type    = "iteration",
        steps        = [4000,],
        step_values  = [-1.,],
        ending_value = -10.)

    ts_per_rollout = num_procs * 512

    run_ppo(env_generator       = env_generator,
            random_seed         = random_seed,
            ac_network          = FeedForwardNetwork,
            actor_kw_args       = actor_kw_args,
            critic_kw_args      = critic_kw_args,
            batch_size          = 512,
            max_ts_per_ep       = 32,
            ts_per_rollout      = ts_per_rollout,
            use_gae             = True,
            normalize_obs       = True,
            normalize_rewards   = True,
            obs_clip            = (-10., 10.),
            reward_clip         = (reward_clip_min, 10.),
            bootstrap_clip      = (bs_clip_min, 10.),
            lr_dec              = lr_dec,
            lr                  = lr,
            min_lr              = min_lr,
            state_path          = state_path,
            load_state          = load_state,
            render              = render,
            render_gif          = render_gif,
            num_timesteps       = num_timesteps,
            device              = device,
            envs_per_proc       = envs_per_proc,
            test                = test,
            use_soft_resets     = True,
            num_test_runs       = num_test_runs)


###############################################################################
#                                Atari                                        #
###############################################################################


def breakout_pixels_ppo(state_path,
                        load_state,
                        render,
                        render_gif,
                        num_timesteps,
                        device,
                        envs_per_proc,
                        random_seed,
                        test = False,
                        num_test_runs = 1):

    if render:
        #
        # NOTE: we don't want to explicitly call render for atari games.
        # They have more advanced ways of rendering.
        #
        render = False

        env_generator = lambda : gym.make(
            'Breakout-v0',
            repeat_action_probability = 0.0,
            frameskip = 1,
            render_mode = 'human')
    else:
        env_generator = lambda : gym.make(
            'Breakout-v0',
            repeat_action_probability = 0.0,
            frameskip = 1)

    wrapper_generator = lambda : BreakoutPixelsEnvWrapper(
        env              = env_generator(),
        allow_life_loss  = test,
        hist_size        = 4,
        skip_k_frames    = 4)

    actor_kw_args = {}
    actor_kw_args["activation"]  = nn.LeakyReLU()
    critic_kw_args = actor_kw_args.copy()

    lr     = 0.0003
    min_lr = 0.0

    lr_dec = LinearDecrementer(
        max_iteration = 4000,
        max_value     = lr,
        min_value     = min_lr)

    ts_per_rollout = num_procs * 512

    run_ppo(env_generator        = wrapper_generator,
            random_seed          = random_seed,
            ac_network           = AtariPixelNetwork,
            actor_kw_args        = actor_kw_args,
            critic_kw_args       = critic_kw_args,
            batch_size           = 512,
            ts_per_rollout       = ts_per_rollout,
            max_ts_per_ep        = 64,
            epochs_per_iter      = 30,
            reward_clip          = (-1., 1.),
            bootstrap_clip       = (-1., 1.),
            target_kl            = 0.2,
            lr_dec               = lr_dec,
            lr                   = lr,
            min_lr               = min_lr,
            use_gae              = True,
            state_path           = state_path,
            load_state           = load_state,
            render               = render,
            render_gif           = render_gif,
            num_timesteps        = num_timesteps,
            device               = device,
            envs_per_proc        = envs_per_proc,
            test                 = test,
            num_test_runs        = num_test_runs)


def breakout_ram_ppo(state_path,
                     load_state,
                     render,
                     render_gif,
                     num_timesteps,
                     device,
                     envs_per_proc,
                     random_seed,
                     test = False,
                     num_test_runs = 1):

    if render:
        #
        # NOTE: we don't want to explicitly call render for atari games.
        # They have more advanced ways of rendering.
        #
        render = False

        env_generator = lambda : gym.make(
            'Breakout-ram-v0',
            repeat_action_probability = 0.0,
            frameskip = 1,
            render_mode = 'human')
    else:
        env_generator = lambda : gym.make(
            'Breakout-ram-v0',
            repeat_action_probability = 0.0,
            frameskip = 1)

    wrapper_generator = lambda : BreakoutRAMEnvWrapper(
        env              = env_generator(),
        allow_life_loss  = test,
        hist_size        = 4,
        skip_k_frames    = 4)

    actor_kw_args = {}
    actor_kw_args["activation"]  = nn.LeakyReLU()
    actor_kw_args["hidden_size"] = 128

    critic_kw_args = actor_kw_args.copy()
    critic_kw_args["hidden_size"] = 256

    lr     = 0.0003
    min_lr = 0.0

    lr_dec = LinearDecrementer(
        max_iteration = 4000,
        max_value     = lr,
        min_value     = min_lr)

    ts_per_rollout = num_procs * 512

    run_ppo(env_generator      = wrapper_generator,
            random_seed        = random_seed,
            ac_network         = FeedForwardNetwork,
            actor_kw_args      = actor_kw_args,
            critic_kw_args     = critic_kw_args,
            batch_size         = 512,
            ts_per_rollout     = ts_per_rollout,
            max_ts_per_ep      = 64,
            use_gae            = True,
            epochs_per_iter    = 30,
            reward_clip        = (-1., 1.),
            bootstrap_clip     = (-1., 1.),
            target_kl          = 0.2,
            lr_dec             = lr_dec,
            lr                 = lr,
            min_lr             = min_lr,
            state_path         = state_path,
            load_state         = load_state,
            render             = render,
            render_gif         = render_gif,
            num_timesteps      = num_timesteps,
            device             = device,
            envs_per_proc      = envs_per_proc,
            test               = test,
            num_test_runs      = num_test_runs)


###############################################################################
#                                MuJoCo                                       #
###############################################################################


def inverted_pendulum_ppo(state_path,
                          load_state,
                          render,
                          render_gif,
                          num_timesteps,
                          device,
                          envs_per_proc,
                          random_seed,
                          test = False,
                          num_test_runs = 1):

    env_generator = lambda : gym.make('InvertedPendulum-v2')

    ts_per_rollout = num_procs * 512

    run_ppo(env_generator       = env_generator,
            random_seed         = random_seed,
            ac_network          = FeedForwardNetwork,
            use_gae             = True,
            use_icm             = False,
            ts_per_rollout      = ts_per_rollout,
            state_path          = state_path,
            load_state          = load_state,
            render              = render,
            render_gif          = render_gif,
            num_timesteps       = num_timesteps,
            device              = device,
            envs_per_proc       = envs_per_proc,
            test                = test,
            num_test_runs       = num_test_runs)


def inverted_double_pendulum_ppo(state_path,
                                 load_state,
                                 render,
                                 render_gif,
                                 num_timesteps,
                                 device,
                                 envs_per_proc,
                                 random_seed,
                                 test = False,
                                 num_test_runs = 1):

    env_generator = lambda : gym.make('InvertedDoublePendulum-v2')

    #
    # Pendulum observations are organized as follows:
    #    Positions: 1
    #    Angles: 4
    #    Velocities: 3
    #    Contact forces: 3
    #
    actor_kw_args = {}

    actor_kw_args["activation"]  = nn.LeakyReLU()
    actor_kw_args["hidden_size"] = 64

    critic_kw_args = actor_kw_args.copy()
    critic_kw_args["hidden_size"] = 128

    lr     = 0.0001
    min_lr = 0.0001

    lr_dec = LinearDecrementer(
        max_iteration = 1.,
        max_value     = lr,
        min_value     = min_lr)

    ts_per_rollout = num_procs * 512

    run_ppo(env_generator       = env_generator,
            random_seed         = random_seed,
            ac_network          = FeedForwardNetwork,
            actor_kw_args       = actor_kw_args,
            critic_kw_args      = critic_kw_args,
            batch_size          = 512,
            max_ts_per_ep       = 16,
            ts_per_rollout      = ts_per_rollout,
            use_gae             = True,
            normalize_obs       = True,
            normalize_rewards   = True,
            obs_clip            = (-10., 10.),
            reward_clip         = (-10., 10.),
            bootstrap_clip      = (-10., 10.),
            entropy_weight      = 0.0,
            lr_dec              = lr_dec,
            lr                  = lr,
            min_lr              = min_lr,
            state_path          = state_path,
            load_state          = load_state,
            render              = render,
            render_gif          = render_gif,
            num_timesteps       = num_timesteps,
            device              = device,
            envs_per_proc       = envs_per_proc,
            test                = test,
            num_test_runs       = num_test_runs)


def ant_ppo(state_path,
            load_state,
            render,
            render_gif,
            num_timesteps,
            device,
            envs_per_proc,
            random_seed,
            test = False,
            num_test_runs = 1):

    env_generator = lambda : gym.make('Ant-v3')

    #
    # Ant observations are organized as follows:
    #    Positions: 13
    #    Velocities: 14
    #    Contact forces: 84
    #
    actor_kw_args = {}
    actor_kw_args["activation"]  = nn.Tanh()
    actor_kw_args["hidden_size"] = 128

    critic_kw_args = actor_kw_args.copy()
    critic_kw_args["hidden_size"] = 256

    lr     = 0.00025
    min_lr = 0.0001

    lr_dec = LinearDecrementer(
        max_iteration = 100,
        max_value     = lr,
        min_value     = min_lr)

    ts_per_rollout = num_procs * 512

    run_ppo(env_generator       = env_generator,
            random_seed         = random_seed,
            ac_network          = FeedForwardNetwork,
            actor_kw_args       = actor_kw_args,
            critic_kw_args      = critic_kw_args,
            batch_size          = 512,
            max_ts_per_ep       = 64,
            ts_per_rollout      = ts_per_rollout,
            use_gae             = True,
            normalize_obs       = True,
            normalize_rewards   = True,
            obs_clip            = (-30., 30.),
            reward_clip         = (-10., 10.),
            bootstrap_clip      = (-10., 10.),
            lr_dec              = lr_dec,
            lr                  = lr,
            min_lr              = min_lr,
            state_path          = state_path,
            load_state          = load_state,
            render              = render,
            render_gif          = render_gif,
            num_timesteps       = num_timesteps,
            device              = device,
            envs_per_proc       = envs_per_proc,
            test                = test,
            num_test_runs       = num_test_runs)


def humanoid_ppo(state_path,
                 load_state,
                 render,
                 render_gif,
                 num_timesteps,
                 device,
                 envs_per_proc,
                 random_seed,
                 test = False,
                 num_test_runs = 1):

    env_generator = lambda : gym.make('Humanoid-v3')

    #
    # Humanoid observations are a bit mysterious. See
    # https://github.com/openai/gym/issues/585
    # Here's a best guess:
    #
    #    Positions: 22
    #    Velocities: 23
    #    Center of mass based on inertia (?): 140
    #    Center of mass based on velocity (?): 84
    #    Actuator forces (?): 23
    #    Contact forces: 84
    #
    # UPDATE: more complete information on the observations cane be found
    # here:
    # https://github.com/openai/gym/blob/master/gym/envs/mujoco/humanoidstandup.py
    #
    actor_kw_args = {}

    # TODO: the current settings work pretty well, but it
    # takes a while to train. Can we do better? Some things
    # that need more exploring:
    #    std offset: is the default optimal?
    #    activation: How does leaky relu do?
    #    target_kl: we could experiment more with this.
    #    obs_clip: this seems to negatively impact results. Does that hold?
    #    entropy: we could allow entropy reg, but I'm guessing it won't help
    #             too much.
    #

    actor_kw_args["activation"]       = nn.Tanh()
    actor_kw_args["distribution_min"] = -0.4
    actor_kw_args["distribution_max"] = 0.4
    actor_kw_args["hidden_size"]      = 256

    critic_kw_args = actor_kw_args.copy()
    critic_kw_args["hidden_size"] = 512

    lr     = 0.0001
    min_lr = 0.0001

    lr_dec = LinearDecrementer(
        max_iteration = 1.0,
        max_value     = lr,
        min_value     = min_lr)

    ts_per_rollout = num_procs * 512

    run_ppo(env_generator       = env_generator,
            random_seed         = random_seed,
            ac_network          = FeedForwardNetwork,
            actor_kw_args       = actor_kw_args,
            critic_kw_args      = critic_kw_args,
            batch_size          = 512,
            max_ts_per_ep       = 16,
            ts_per_rollout      = ts_per_rollout,
            use_gae             = True,
            normalize_obs       = True,
            normalize_rewards   = True,
            reward_clip         = (-10., 10.),
            lr_dec              = lr_dec,
            lr                  = lr,
            min_lr              = min_lr,
            state_path          = state_path,
            load_state          = load_state,
            render              = render,
            render_gif          = render_gif,
            num_timesteps       = num_timesteps,
            device              = device,
            envs_per_proc       = envs_per_proc,
            test                = test,
            num_test_runs       = num_test_runs)


def humanoid_stand_up_ppo(state_path,
                          load_state,
                          render,
                          render_gif,
                          num_timesteps,
                          device,
                          envs_per_proc,
                          random_seed,
                          test = False,
                          num_test_runs = 1):

    #
    # NOTE: this is an UNSOVLED environment.
    #
    env_generator = lambda : gym.make('HumanoidStandup-v2')

    #
    #    Positions: 22
    #    Velocities: 23
    #    Center of mass based on inertia (?): 140
    #    Center of mass based on velocity (?): 84
    #    Actuator forces (?): 23
    #    Contact forces: 84
    #
    # UPDATE: more complete information on the observations cane be found
    # here:
    # https://github.com/openai/gym/blob/master/gym/envs/mujoco/humanoidstandup.py
    #
    actor_kw_args = {}
    actor_kw_args["activation"]       = nn.Tanh()
    actor_kw_args["distribution_min"] = -0.4
    actor_kw_args["distribution_max"] = 0.4
    actor_kw_args["hidden_size"]      = 256

    critic_kw_args = actor_kw_args.copy()
    critic_kw_args["hidden_size"] = 512

    lr     = 0.0003
    min_lr = 0.0001

    lr_dec = LinearDecrementer(
        max_iteration = 200.0,
        max_value     = lr,
        min_value     = min_lr)

    ts_per_rollout = num_procs * 512

    run_ppo(env_generator       = env_generator,
            random_seed         = random_seed,
            ac_network          = FeedForwardNetwork,
            actor_kw_args       = actor_kw_args,
            critic_kw_args      = critic_kw_args,
            batch_size          = 512,
            max_ts_per_ep       = 32,
            ts_per_rollout      = ts_per_rollout,
            use_gae             = True,
            normalize_obs       = True,
            normalize_rewards   = True,
            obs_clip            = None,
            reward_clip         = (-10., 10.),
            lr_dec              = lr_dec,
            lr                  = lr,
            min_lr              = min_lr,
            state_path          = state_path,
            load_state          = load_state,
            render              = render,
            render_gif          = render_gif,
            num_timesteps       = num_timesteps,
            device              = device,
            envs_per_proc       = envs_per_proc,
            test                = test,
            num_test_runs       = num_test_runs)


def walker2d_ppo(state_path,
                 load_state,
                 render,
                 render_gif,
                 num_timesteps,
                 device,
                 envs_per_proc,
                 random_seed,
                 test = False,
                 num_test_runs = 1):

    env_generator = lambda : gym.make('Walker2d-v3')

    actor_kw_args = {}
    actor_kw_args["activation"]  = nn.Tanh()
    actor_kw_args["hidden_size"] = 64

    critic_kw_args = actor_kw_args.copy()
    critic_kw_args["hidden_size"] = 256

    lr     = 0.0003
    min_lr = 0.0001

    lr_dec = LinearDecrementer(
        max_iteration = 600,
        max_value     = lr,
        min_value     = min_lr)

    ts_per_rollout = num_procs * 1024

    #
    # arXiv:2006.05990v1 suggests that value normalization significantly hurts
    # performance in walker2d. I also find this to be the case.
    #
    run_ppo(env_generator       = env_generator,
            random_seed         = random_seed,
            ac_network          = FeedForwardNetwork,
            actor_kw_args       = actor_kw_args,
            critic_kw_args      = critic_kw_args,
            batch_size          = 512,
            max_ts_per_ep       = 16,
            ts_per_rollout      = ts_per_rollout,
            use_gae             = True,
            normalize_obs       = True,
            normalize_rewards   = True,
            normalize_values    = False,
            obs_clip            = (-10., 10.),
            reward_clip         = (-10., 10.),
            entropy_weight      = 0.0,
            lr_dec              = lr_dec,
            lr                  = lr,
            min_lr              = min_lr,
            state_path          = state_path,
            load_state          = load_state,
            render              = render,
            render_gif          = render_gif,
            num_timesteps       = num_timesteps,
            device              = device,
            envs_per_proc       = envs_per_proc,
            test                = test,
            num_test_runs       = num_test_runs)


def hopper_ppo(state_path,
               load_state,
               render,
               render_gif,
               num_timesteps,
               device,
               envs_per_proc,
               random_seed,
               test = False,
               num_test_runs = 1):

    env_generator = lambda : gym.make('Hopper-v3')

    actor_kw_args = {}
    actor_kw_args["activation"]  = nn.Tanh()
    actor_kw_args["hidden_size"] = 64

    critic_kw_args = actor_kw_args.copy()
    critic_kw_args["hidden_size"] = 256

    lr     = 0.0003
    min_lr = 0.0001

    lr_dec = LinearStepMapper(
        step_type    = "iteration",
        steps        = [400,],
        step_values  = [0.0003,],
        ending_value = 0.0001)

    ts_per_rollout = num_procs * 1024

    #
    # I find that value normalization hurts the hopper environment training.
    # That may be a result of it's combination with other settings in here.
    #
    run_ppo(env_generator       = env_generator,
            random_seed         = random_seed,
            ac_network          = FeedForwardNetwork,
            actor_kw_args       = actor_kw_args,
            critic_kw_args      = critic_kw_args,
            batch_size          = 512,
            max_ts_per_ep       = 16,
            ts_per_rollout      = ts_per_rollout,
            use_gae             = True,
            normalize_obs       = True,
            normalize_rewards   = True,
            normalize_values    = False,
            obs_clip            = (-10., 10.),
            reward_clip         = (-10., 10.),
            entropy_weight      = 0.0,
            lr_dec              = lr_dec,
            lr                  = lr,
            min_lr              = min_lr,
            state_path          = state_path,
            load_state          = load_state,
            render              = render,
            render_gif          = render_gif,
            num_timesteps       = num_timesteps,
            device              = device,
            envs_per_proc       = envs_per_proc,
            test                = test,
            num_test_runs       = num_test_runs)


def half_cheetah_ppo(state_path,
                     load_state,
                     render,
                     render_gif,
                     num_timesteps,
                     device,
                     envs_per_proc,
                     random_seed,
                     test = False,
                     num_test_runs = 1):

    env_generator = lambda : gym.make('HalfCheetah-v3')

    actor_kw_args = {}
    actor_kw_args["activation"]  = nn.LeakyReLU()
    actor_kw_args["hidden_size"] = 128

    critic_kw_args = actor_kw_args.copy()
    critic_kw_args["hidden_size"] = 256

    lr     = 0.0001
    min_lr = 0.0001

    lr_dec = LinearDecrementer(
        max_iteration = 1.0,
        max_value     = lr,
        min_value     = min_lr)

    ts_per_rollout = num_procs * 512

    #
    # Normalizing values seems to stabilize results in this env.
    #
    run_ppo(env_generator       = env_generator,
            random_seed         = random_seed,
            ac_network          = FeedForwardNetwork,
            actor_kw_args       = actor_kw_args,
            critic_kw_args      = critic_kw_args,
            batch_size          = 512,
            max_ts_per_ep       = 32,
            ts_per_rollout      = ts_per_rollout,
            use_gae             = True,
            normalize_obs       = True,
            normalize_rewards   = True,
            obs_clip            = (-10., 10.),
            reward_clip         = (-10., 10.),
            lr_dec              = lr_dec,
            lr                  = lr,
            min_lr              = min_lr,
            state_path          = state_path,
            load_state          = load_state,
            render              = render,
            render_gif          = render_gif,
            num_timesteps       = num_timesteps,
            device              = device,
            envs_per_proc       = envs_per_proc,
            test                = test,
            num_test_runs       = num_test_runs)


def swimmer_ppo(state_path,
                load_state,
                render,
                render_gif,
                num_timesteps,
                device,
                envs_per_proc,
                random_seed,
                test = False,
                num_test_runs = 1):

    env_generator = lambda : gym.make('Swimmer-v3')

    actor_kw_args = {}
    actor_kw_args["activation"]  = nn.LeakyReLU()
    actor_kw_args["hidden_size"] = 64

    critic_kw_args = actor_kw_args.copy()
    critic_kw_args["hidden_size"] = 256

    lr     = 0.0001
    min_lr = 0.0001

    lr_dec = LinearDecrementer(
        max_iteration = 1.0,
        max_value     = lr,
        min_value     = min_lr)

    ts_per_rollout = num_procs * 1024

    run_ppo(env_generator       = env_generator,
            random_seed         = random_seed,
            ac_network          = FeedForwardNetwork,
            actor_kw_args       = actor_kw_args,
            critic_kw_args      = critic_kw_args,
            batch_size          = 512,
            max_ts_per_ep       = 32,
            ts_per_rollout      = ts_per_rollout,
            use_gae             = True,
            normalize_obs       = True,
            normalize_rewards   = True,
            obs_clip            = (-10., 10.),
            reward_clip         = (-10., 10.),
            lr_dec              = lr_dec,
            lr                  = lr,
            min_lr              = min_lr,
            state_path          = state_path,
            load_state          = load_state,
            render              = render,
            render_gif          = render_gif,
            num_timesteps       = num_timesteps,
            device              = device,
            envs_per_proc       = envs_per_proc,
            test                = test,
            num_test_runs       = num_test_runs)


###############################################################################
#                              Multi-Agent                                    #
###############################################################################

def robot_warehouse_tiny(
    state_path,
    load_state,
    render,
    render_gif,
    num_timesteps,
    device,
    envs_per_proc,
    random_seed,
    test          = False,
    num_test_runs = 1):

    env_generator = lambda : gym.make('rware-tiny-3ag-v1')

    actor_kw_args = {}
    actor_kw_args["activation"]  = nn.LeakyReLU()
    actor_kw_args["hidden_size"] = 256

    critic_kw_args = actor_kw_args.copy()
    critic_kw_args["hidden_size"] = 512

    lr     = 0.001
    min_lr = 0.00001

    lr_dec = LinearDecrementer(
        max_timestep  = 40000000,
        max_value     = lr,
        min_value     = min_lr)

    entropy_weight     = 0.05
    min_entropy_weight = 0.01

    entropy_dec = LinearDecrementer(
        max_timestep  = 40000000,
        max_value     = entropy_weight,
        min_value     = min_entropy_weight)
    #
    # Each rank has 3 agents, which will be interpreted as individual
    # environments, so (internally) we multiply our ts_per_rollout by
    # the number of agents. We want each rank to see ~2 episodes =>
    # num_ranks * 2 * 512.
    #
    ts_per_rollout = num_procs * 2 * 512

    #
    # This environment comes from arXiv:2006.07869v4.
    # This is a very sparse reward environment, and there are series of
    # complex actions that must occur in between rewards. Because of this,
    # using a large maximum timesteps per episode results in faster learning.
    # arXiv:2103.01955v2 suggests using smaller epoch counts for complex
    # environments and large batch sizes (single batches if possible).
    # Because of the sparse rewards, I've also increased the entropy
    # weight to incentivize exploration. We could also experiment with
    # using ICM here. I've disabled rewards and observation normalization
    # and clipping, mainly because they aren't mentioned in arXiv:2103.01955v2.
    # I've noticed that performance tends to go down a bit when these
    # normalizations are enabled.
    #
    run_ppo(env_generator       = env_generator,
            random_seed         = random_seed,
            ac_network          = FeedForwardNetwork,
            actor_kw_args       = actor_kw_args,
            critic_kw_args      = critic_kw_args,
            batch_size          = 10000,
            epochs_per_iter     = 5,
            max_ts_per_ep       = 512,
            ts_per_rollout      = ts_per_rollout,
            is_multi_agent      = True,
            use_gae             = True,
            normalize_values    = True,
            normalize_obs       = False,
            obs_clip            = None,
            normalize_rewards   = False,
            reward_clip         = None,
            entropy_weight      = entropy_weight,
            min_entropy_weight  = min_entropy_weight,
            entropy_dec         = entropy_dec,
            lr_dec              = lr_dec,
            lr                  = lr,
            min_lr              = min_lr,
            state_path          = state_path,
            load_state          = load_state,
            render              = render,
            render_gif          = render_gif,
            num_timesteps       = num_timesteps,
            device              = device,
            envs_per_proc       = envs_per_proc,
            test                = test,
            num_test_runs       = num_test_runs)

def robot_warehouse_small(
    state_path,
    load_state,
    render,
    render_gif,
    num_timesteps,
    device,
    envs_per_proc,
    random_seed,
    test          = False,
    num_test_runs = 1):

    env_generator = lambda : gym.make('rware-small-4ag-v1')

    actor_kw_args = {}
    actor_kw_args["activation"]  = nn.LeakyReLU()
    actor_kw_args["hidden_size"] = 256

    critic_kw_args = actor_kw_args.copy()
    critic_kw_args["hidden_size"] = 512

    lr     = 0.001
    min_lr = 0.00001

    lr_dec = LinearDecrementer(
        max_iteration = 6000,
        max_value     = lr,
        min_value     = min_lr)

    entropy_weight     = 0.05
    min_entropy_weight = 0.01

    entropy_dec = LinearDecrementer(
        max_iteration = 6000,
        max_value     = entropy_weight,
        min_value     = min_entropy_weight)

    #
    # Each rank has 4 agents, which will be interpreted as individual
    # environments, so (internally) we multiply our ts_per_rollout by
    # the number of agents. We want each rank to see ~4 episodes =>
    # num_ranks * 4 * 512.
    #
    ts_per_rollout = num_procs * 4 * 512

    #
    # This is a very sparse reward environment, and there are series of
    # complex actions that must occur in between rewards. Because of this,
    # using a large maximum timesteps per episode results in faster learning.
    # arXiv:2103.01955v2 suggests using smaller epoch counts for complex
    # environments and large batch sizes (single batches if possible).
    # Because of the sparse rewards, I've also increased the entropy
    # weight to incentivize exploration. We could also experiment with
    # using ICM here. I've disabled rewards and observation normalization
    # and clipping, mainly because they aren't mentioned in arXiv:2103.01955v2.
    # I've noticed that performance tends to go down a bit when these
    # normalizations are enabled.
    #
    run_ppo(env_generator       = env_generator,
            random_seed         = random_seed,
            ac_network          = FeedForwardNetwork,
            actor_kw_args       = actor_kw_args,
            critic_kw_args      = critic_kw_args,
            batch_size          = 10000,
            epochs_per_iter     = 5,
            max_ts_per_ep       = 512,
            ts_per_rollout      = ts_per_rollout,
            is_multi_agent      = True,
            use_gae             = True,
            normalize_values    = True,
            normalize_obs       = False,
            obs_clip            = None,
            normalize_rewards   = False,
            reward_clip         = None,
            entropy_weight      = entropy_weight,
            min_entropy_weight  = min_entropy_weight,
            entropy_dec         = entropy_dec,
            lr_dec              = lr_dec,
            lr                  = lr,
            min_lr              = min_lr,
            state_path          = state_path,
            load_state          = load_state,
            render              = render,
            render_gif          = render_gif,
            num_timesteps       = num_timesteps,
            device              = device,
            envs_per_proc       = envs_per_proc,
            test                = test,
            num_test_runs       = num_test_runs)
