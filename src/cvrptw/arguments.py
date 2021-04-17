import argparse

def args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--N_JOBS', default=213)
    parser.add_argument('-b', '--BATCH', default=2)
    parser.add_argument('--LR', default=3e-4)
    parser.add_argument('--N_ROLLOUT', default=2)
    parser.add_argument('--ROLLOUT_STEPS', default=2)
    parser.add_argument('--N_STEPS', default=200)
    parser.add_argument('--init_T', default=100.0)
    parser.add_argument('--final_T', default=1.0)
    parser.add_argument('--device', default="cpu")
    return parser.parse_args()
