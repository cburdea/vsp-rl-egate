import argparse

def args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--N_JOBS', default=100)
    parser.add_argument('-b', '--BATCH', default=64)
    parser.add_argument('-b', '--N_EPOCHS', default=1)
    parser.add_argument('--LR', default=3e-4)
    parser.add_argument('--N_ROLLOUT', default=20)
    parser.add_argument('--ROLLOUT_STEPS', default=10)
    parser.add_argument('--N_STEPS', default=200)
    parser.add_argument('--init_T', default=100.0)
    parser.add_argument('--final_T', default=1.0)
    parser.add_argument('--device', default="cpu")
    return parser.parse_args()
