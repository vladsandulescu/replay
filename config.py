PARAMS = {
    # dqn
    'layers': 1,
    'hidden': 256,
    'Q_lr': 0.001,
    'experience_replay_size': int(1e6),
    'polyak_tau': 0.95,
    # training
    'n_epochs': 200, # number of training epochs
    'n_cycles': 50,  # per epoch
    'n_episodes': 16, # number of training episodes
    'n_optimization_steps': 40, # per cycle
    'gamma': 0.98, # Q learning discount factor
    'polyak_tau': 0.95, # polyak tau for Q learning target network update
    'batch_size': 128,  # mini-batch size
}