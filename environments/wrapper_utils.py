from ppo_and_friends.environments.ppo_env_wrappers import VectorizedEnv
from ppo_and_friends.environments.filter_wrappers import ObservationNormalizer, ObservationClipper
from ppo_and_friends.environments.filter_wrappers import RewardNormalizer, RewardClipper
from ppo_and_friends.environments.filter_wrappers import ObservationAugmentingWrapper
from ppo_and_friends.utils.mpi_utils import rank_print
from collections.abc import Iterable

def wrap_environment(
    env_generator,
    add_agent_ids         = False,
    envs_per_proc         = 1,
    random_seed           = 2,
    obs_augment           = False,
    normalize_obs         = True,
    normalize_rewards     = True,
    obs_clip              = None,
    reward_clip           = None,
    gamma                 = 0.99,
    test_mode             = False):
    """
    """
    #
    # Begin adding wrappers. Order matters!
    # The first wrapper will always be either a standard vectorization
    # or a multi-agent wrapper. We currently don't support combining them.
    #
    env = VectorizedEnv(
        env_generator = env_generator,
        num_envs      = envs_per_proc,
        test_mode     = test_mode)

    #
    # For reproducibility, we need to set the environment's random
    # seeds. Let's allow testing to be random.
    #
    if not test_mode:
        env.set_random_seed(random_seed)

    #
    # The second wrapper should always be the augmenter. This is because
    # our environment should receive pre-normalized data for augmenting.
    #
    if obs_augment:
        env = ObservationAugmentingWrapper(
            env,
            test_mode = test_mode)

    if normalize_obs:
        env = ObservationNormalizer(
            env          = env,
            test_mode    = test_mode,
            update_stats = not test_mode)

    status_dict = {}
    if obs_clip != None and type(obs_clip) == tuple:
        env = ObservationClipper(
            env         = env,
            test_mode   = test_mode,
            status_dict = status_dict,
            clip_range  = obs_clip)

    #
    # There are multiple ways to go about normalizing rewards.
    # The approach in arXiv:2006.05990v1 is to normalize before
    # sending targets to the critic and then de-normalize when predicting.
    # We're taking the OpenAI approach of normalizing the rewards straight
    # from the environment and keeping them normalized at all times.
    #
    if normalize_rewards:
        env = RewardNormalizer(
            env          = env,
            test_mode    = test_mode,
            update_stats = not test_mode,
            gamma        = gamma)

    if reward_clip != None and type(reward_clip) == tuple:
        env = RewardClipper(
            env         = env,
            test_mode   = test_mode,
            status_dict = status_dict,
            clip_range  = reward_clip)

    return env, status_dict
