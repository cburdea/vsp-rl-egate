# Solving the SDVSP with Reinforcement Learning


This implementation tackles the single-depot vehicle scheduling problem by customizing the work of Gao et al. (2020). Their work can be found with this link:

  
> <cite> Learn to Design the Heuristics for Vehicle Routing Problem [arxiv link](https://arxiv.org/abs/2002.08539)</cite>

Please install vsp_env-0.1.1 before training or evaluation. The needed dependencies can be also found in the requirements.txt file.
Run train_model.py to train a vsp model, and evaluation.py to evaluate on the test data. The
default arguments can be found in arguments.py. For example: 
```
python train_model.py -e 1000
```

