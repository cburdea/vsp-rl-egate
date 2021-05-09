import argparse

def args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--N_JOBS', default=100)
    parser.add_argument('-b', '--BATCH', default=2)
    parser.add_argument('-e', '--N_EPOCHS', default=4)
    parser.add_argument('--EVAL_MODE', default="operational") #alternative is vehicle
    parser.add_argument('--RANDOMIZE', default=False)
    parser.add_argument('--LR', default=3e-4)
    parser.add_argument('--N_ROLLOUT', default=20)
    parser.add_argument('--ROLLOUT_STEPS', default=10)
    parser.add_argument('--N_STEPS', default=100)
    parser.add_argument('--init_T', default=100.0)
    parser.add_argument('--final_T', default=1.0)
    parser.add_argument('--device', default="cpu")
    return parser.parse_args()
