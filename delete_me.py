train(model = model,envs = envs,epochs= 1000, n_rollout = 20, rollout_steps=10,train_steps = 4,n_remove=10)

roll_out(model = model,envs = envs,states = states,n_steps = rollout_steps,_lambda=0.99, batch_size=batch_size, n_remove=n_remove,is_last=False, greedy=False)

def roll_out(model,envs,states,n_steps=10,_lambda=0.99,batch_size=batch_size,n_remove=10,is_last=False,greedy=False):
    buffer = create_replay_buffer()
    with torch.no_grad():
        model.eval()
        nodes,edges = states
        _sum = 0
        _entropy = []

        for i in range(n_steps):
            data = buffer.create_data(nodes,edges)
            data = data.to(device)
            actions,log_p,values,entropy = model(data,n_remove,greedy,128)
#             print (actions)
            new_nodes,new_edges,rewards = envs.step(actions.cpu().numpy())
            rewards = np.array(rewards)
            _sum = _sum + rewards
            rewards = reward_norm(rewards)
            _entropy.append(entropy.mean().cpu().numpy())

            buffer.obs(nodes,edges,actions.cpu().numpy(),rewards,log_p.cpu().numpy(),values.cpu().numpy())
            nodes,edges = new_nodes,new_edges

        mean_value = _sum.mean()
#         print ("mean rewards:",mean_value)
#         print ("entropy:",np.mean(_entropy))
#         print ("mean cost:",np.mean([env.cost for env in envs.envs]))

        if not is_last:
#             print ("not last")
            data = buffer.create_data(nodes,edges)
            data = data.to(device)
            actions,log_p,values,entropy = model(data,n_remove,greedy)
            values = values.cpu().numpy()
        else:
            values = 0

        dl = buffer.gen_datas(values,_lambda = _lambda,batch_size=batch_size)
        return dl,(nodes,edges)

