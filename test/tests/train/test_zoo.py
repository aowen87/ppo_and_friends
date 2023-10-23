from utils import run_training, high_score_test

#
# MPE tests
#
def test_mpe_simple_tag_mpi():
    num_timesteps = 400000
    passing_scores = {"adversary_1" : 30.0}

    run_training(
        baseline_runner = 'mpe_simple_tag.py',
        num_timesteps   = num_timesteps,
        num_ranks       = 2,
        options         = '--test-explore')

    high_score_test('mpi mpe simple tag',
        'mpe_simple_tag.py', 10, passing_scores)

