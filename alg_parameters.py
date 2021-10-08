
# training parameters
actor_lr = 5e-4
critic_lr = 5e-4
weight_lr =  5e-4
GAMMA = 0.99
LAM = 0.95
CLIP_RANGE = 0.2
MAX_GRAD_NORM = 20
ENTROPY_COEF = 0.01
VALUE_COEF = 1
POLICY_COEF = 1
N_STEPS = 2**9
N_MINIBATCHES = 16
N_EPOCHS = 10
N_ENVS = 16
N_MAX_STEPS = 1e7
N_UPDATES = int(N_MAX_STEPS // (N_STEPS * N_ENVS))
BATCH_SIZE = int(N_STEPS * N_ENVS)
MINIBATCH_SIZE = int(N_STEPS * N_ENVS // N_MINIBATCHES)

# environment parameter
N_AGENTS=3
N_ACTIONS= 10
EPISODE_LEN= 200

# network parameter
ACTOR_INPUT_LEN =  30
CRITIC_INPUT_LEN = 78

h_num=32
h1_scheduler = 32
h2_scheduler = 32
h_critic = 64
h1_critic = h_critic
h2_critic = h_critic
h3_critic = h_critic

# communication parameter
s_num =2
capacity=10

#  other parameters
SEED = 1234
SAVE_INTERVAL = 200
EVALUE_INTERVAL = 5000  #step
EVALUE_EPISODES = 16
EXPERIMENT_NAME = 'Schednet'
USER_NAME = 'Testing'

algArgs = {'actor_lr': actor_lr, 'critic_lr': critic_lr, 'GAMMA': GAMMA, 'LAM': LAM, 'CLIPRANGE': CLIP_RANGE,
           'MAX_GRAD_NORM': MAX_GRAD_NORM, 'ENTROPY_COEF': ENTROPY_COEF, 'VALUE_COEF': VALUE_COEF,
           'POLICY_COEF': POLICY_COEF, 'N_STEPS': N_STEPS, 'N_MINIBATCHES': N_MINIBATCHES,
           'N_EPOCHS': N_EPOCHS, 'N_ENVS': N_ENVS, 'N_MAX_STEPS': N_MAX_STEPS,
           'N_UPDATES': N_UPDATES, 'MINIBATCH_SIZE': MINIBATCH_SIZE, 'SEED': SEED,
           'SAVE_INTERVAL': SAVE_INTERVAL, 'EVALUE_INTERVAL': EVALUE_INTERVAL, 'EVALUE_EPISODES': EVALUE_EPISODES,
           'EXPERIMENT_NAME': EXPERIMENT_NAME,
           'USER_NAME': USER_NAME,"weight_lr":weight_lr,"s_num":s_num,"capacity":capacity}




