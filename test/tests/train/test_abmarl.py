from utils import run_training, high_score_test
import pytest

@pytest.mark.skip(reason="Need Abmarl's update to gymansium")
def test_abmarl_maze_mpi():
    num_timesteps = 50000
    passing_scores = {"navigator" : 0.0}

    run_training(
        baseline_runner = 'abmarl_maze.py',
        num_timesteps   = num_timesteps,
        num_ranks       = 2,
        options         = '--test-explore')

    high_score_test('mpi abmarl maze',
        'abmarl_maze.py', 10, passing_scores)
