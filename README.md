# RL_Path_Planning

To load conda environment, run conda env create -f environment.yml

Things to do:

- Make Plotting Clearer
- Find source of jittery-ness by adjusting one variable at a time
- Figure out how to use tensorboard (check actor-critic loss)
- Evaluate best model vs final model
- Look into tensorboard hyperparameter sweeping
- Look into custom tensorboard callbacks
- Save Evaluation Data to File
- Add rollout period as parameter
- Run standard SAC Example to see what desired outputs should look like
- Keep dt/T the same, change them, and compare (shrinking dt vs Increasing T)


Later:
- Import PVTOL
- Update Reward to be 1 - e^-r
- Train models on different total times; evaluate on same total_time


Base Parameters:

param_dict = {
    #shared params
    'dt': 0.1,
    'init_low': -5,
    'init_high': 5,
    'test': False,
    #RL_env parameters
    'total_time': 10,
    'total_timesteps': 10000,
    'cost_weights': [10, 10, 1],
    'test_sizes': [0.2, 1, 3],
    #base env parameters
    'b' : -2,
    'action_high': 10,
    'action_low': -10,
    #reference env parameters
    'internal_matrix': [[0, -1], [1, 0]],
    'path_matrix': [0, 1],
    #model parameters
    'policy_kwarg': dict(activation_fn=th.nn.Tanh),
    'eval_freq': 1000,
    'gamma': 0.98,
}


Runs:

 - Have one sweep over each parameter
 - Train a couple models on different total_times and train on same total_time
 - Run for long enough to see actor and critic loss
 - Compare Regular reward with updated Reward to be 1 - e^-r

 Questions:

 - Deterministic vs Stochastic Action in Eval Callback?
 - "Rollback" = policy update?? (same as episode?)
 		- specific parameter; look into tuning
 - which models should I continue running with?