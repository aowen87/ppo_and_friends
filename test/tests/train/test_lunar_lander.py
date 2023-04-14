from utils import run_training, average_score_test, high_score_test

def test_lunar_lander_mpi():

    num_timesteps = 400000
    cmd  = f"mpirun -n 4 python train_baseline.py "
    cmd += f"LunarLander --clobber --num-timesteps {num_timesteps}"

    run_training(cmd)

    cmd  = f"python train_baseline.py "
    cmd += f"LunarLander --test --test-explore --num-test-runs 10 "
    cmd += f"--save-test-scores"

    passing_scores = {"agent0" : 200.}

    average_score_test("lunar-lander-mpi", cmd,
        passing_scores, "LunarLander")

def test_binary_lunar_lander_mpi():

    num_timesteps = 400000
    cmd  = f"mpirun -n 4 python train_baseline.py "
    cmd += f"BinaryLunarLander --clobber --num-timesteps {num_timesteps}"

    run_training(cmd)

    cmd  = f"python train_baseline.py "
    cmd += f"BinaryLunarLander --test --test-explore --num-test-runs 10 "
    cmd += f"--save-test-scores"

    passing_scores = {"agent0" : 200.}

    high_score_test("binary-lunar-lander-mpi", cmd,
        passing_scores, "BinaryLunarLander")

if __name__ == "__main__":
    test_lunar_lander_mpi()
    test_binary_lunar_lander_mpi()