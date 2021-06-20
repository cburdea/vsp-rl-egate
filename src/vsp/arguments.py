import argparse

def args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--N_JOBS', default=10)
    parser.add_argument('-b', '--BATCH', default=1)
    parser.add_argument('-e', '--N_EPOCHS', default=1)
    parser.add_argument('--EVAL_MODE', default="vehicle") #alternative is vehicle or operational
    parser.add_argument('--RANDOMIZE', default=True)
    parser.add_argument('--LR', default=3e-4)
    parser.add_argument('--N_ROLLOUT', default=1) #Standard: 20
    parser.add_argument('--ROLLOUT_STEPS', default=1) #Standard: 10
    parser.add_argument('--N_STEPS', default=100)
    parser.add_argument('--PRE_STEPS', default=0)
    parser.add_argument('--init_T', default=5000.0)
    parser.add_argument('--final_T', default=1.0)
    parser.add_argument('--device', default="cpu")
    return parser.parse_args()
