import sys
import dill as pickle
import subprocess
import os
from pathlib import Path

def get_root_dir():
    """
        Get the root directory for ppo_and_friends.

        Returns:
            The full path to the root directory of the ppo_and_friends
            repo.
    """
    test_path = os.path.realpath(__file__)
    root_dir  = Path(str(test_path).split("ppo_and_friends")[0])
    root_dir  = root_dir / "ppo_and_friends"
    return root_dir

def get_final_status(stdout):
    """
        Get the final status report from a string of training stdout.

        Arguments:
            stdout    A string capturing the stdout of a training.

        Returns:
            The final status report from the stdout.
    """
    try:
        final_status = stdout.split("Status Report:")[-1]
    except:
        final_status = None
    return final_status

def run_training(train_command):
    """
        Run training on a sub-process.

        Arguments:
            train_command    The training command.
    """
    root_dir = get_root_dir()
    cur_dir  = Path(os.getcwd())

    train_command += f" --state-path {cur_dir}"
    print(f"Running command: {train_command}")

    os.chdir(root_dir)
    result = subprocess.run(train_command.split(),
        capture_output=True, text=True)
    os.chdir(cur_dir)

def average_score_test(name, test_command, passing_scores, state_dir):
    """
        Run a testing phase using a trained model and determine if
        the model reaches passing scores.

        Arguments:
            name             The name of the test.
            test_command     The test command to run.
            passing_scores   A dict containing passing scores for each
                             agent.
            state_dir        The name of the state directory.
    """
    root_dir = get_root_dir()
    cur_dir  = Path(os.getcwd())

    test_command += f" --state-path {cur_dir}"
    print(f"Running command: {test_command}")

    os.chdir(root_dir)
    result = subprocess.run(test_command.split(),
        capture_output=True, text=True)
    os.chdir(cur_dir)

    score_file = os.path.join("saved_states", state_dir, "test-scores.pickle")
    with open(score_file, "rb") as in_f:
        scores = pickle.load(in_f)

    fail_msg  = f"\n************{name} FAILED************"
    fail_msg += f"\nExpected avg scores:\n {passing_scores}"
    fail_msg += f"\nActual scores:\n {scores}"

    for agent_id in passing_scores:
        msg = f"ERROR: unable to find agent {agent_id} in the test scores."
        assert agent_id in scores, msg

        assert scores[agent_id]["avg_score"] >= passing_scores[agent_id], fail_msg

    print(f"\n************{name} PASSED************")
