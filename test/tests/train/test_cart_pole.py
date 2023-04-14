from utils import run_training, average_score_test

def run_cart_pole_test(name, num_test_runs=10):

    cmd  = f"python train_baseline.py "
    cmd += f"CartPole --test "
    cmd += f"--num-test-runs {num_test_runs} "
    cmd += f"--save-test-scores"

    passing_scores = {"agent0" : 200.}

    average_score_test(name, cmd,
        passing_scores, "CartPole")

def test_cart_pole_serial():

    num_timesteps = 17000
    cmd  = f"python train_baseline.py "
    cmd += f"CartPole --clobber --num-timesteps {num_timesteps}"

    run_training(cmd)
    run_cart_pole_test("cart-pole-serial")

def test_cart_pole_mpi():

    num_timesteps = 40000
    cmd  = f"mpirun -n 4 python train_baseline.py "
    cmd += f"CartPole --clobber --num-timesteps {num_timesteps}"

    run_training(cmd)
    run_cart_pole_test("cart-pole-mpi")

def test_cart_pole_multi_envs():

    num_timesteps = 80000
    cmd  = f"python train_baseline.py "
    cmd += f"CartPole --clobber --num-timesteps {num_timesteps} "
    cmd += f"--envs-per-proc 2"

    run_training(cmd)
    run_cart_pole_test("cart-pole-multi-env")

def test_cart_pole_multi_envs_mpi():

    num_timesteps = 40000
    cmd  = f"mpirun -n 4 python train_baseline.py "
    cmd += f"CartPole --clobber --num-timesteps {num_timesteps} "
    cmd += f"--envs-per-proc 2"

    run_training(cmd)
    run_cart_pole_test("cart-pole-multi-env-mpi")

def test_binary_cart_pole_serial():

    num_timesteps = 17000
    cmd  = f"python train_baseline.py "
    cmd += f"BinaryCartPole --clobber --num-timesteps {num_timesteps}"

    run_training(cmd)
    run_cart_pole_test("cart-pole-binary-serial")


if __name__ == "__main__":
    test_cart_pole_serial()
    test_cart_pole_mpi()
    test_cart_pole_multi_envs()
    test_cart_pole_multi_envs_mpi()
    test_binary_cart_pole_serial()