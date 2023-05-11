args_dict = {
    #File paths
    'TREE_TRAIN_URDF_PATH': {
        'type': str,
        'default': './ur_e_description/urdf/trees/train',
        'help': 'path to the train URDF file for the tree environment'
    },
    'TREE_TRAIN_OBJ_PATH': {
        'type': str,
        'default': './ur_e_description/meshes/trees/train',
        'help': 'path to the train OBJ file for the tree environment'
    },
    'TREE_TEST_URDF_PATH': {
        'type': str,
        'default': './ur_e_description/urdf/trees/test',
        'help': 'path to the test URDF file for the tree environment'
    },
    'TREE_TEST_OBJ_PATH': {
        'type': str,
        'default': './ur_e_description/meshes/trees/test',
        'help': 'path to the test OBJ file for the tree environment'
    },
    #PPO parameters
    'EMB_SIZE': {
        'type': int,
        'default': 128,
        'help': 'size of the embedding layer'
    },
    'ACTION_DIM_ACTOR': {
        'type': int,
        'default': 6,
        'help': 'dimension of the action space for the actor network'
    },
    'STATE_DIM': {
        'type': int,
        'default': 72+16*2,
        'help': 'dimension of the state space'
    },
    #Gym envirionment parameters
    'N_ENVS': {
        'type': int,
        'default': 1,
        'help': 'number of parallel environments to run'
    },
    'RENDER': {
        'type': bool,
        'default': False,
        'help': 'whether to render the environment'
    },
    #Training parameters
    'TOTAL_TIMESTEPS': {
        'type': int,
        'default': 1000000,
        'help': 'total number of timesteps to train for'
    },
    'STEPS_PER_EPOCH': {
        'type': int,
        'default': 20,
        'help': 'number of timesteps per epoch'
    },
    'EPOCHS': {
        'type': int,
        'default': 10,
        'help': 'number of epochs to train for'
    },
    'BATCH_SIZE': {
        'type': int,
        'default': 100,
        'help': 'batch size'
    },
    'LEARNING_RATE': {
        'type': float,
        'default': 0.00001,
        'help': 'learning rate'
    },
    #Evaluation parameters
    'EVAL_FREQ': {
        'type': int,
        'default': 100,
        'help': 'frequency of evaluation'
    },
    'EVAL_EPISODES': {
        'type': int,
        'default': 1,
        'help': 'number of episodes to run during evaluation'
    },
    'EVAL_POINTS': {
        'type': int,
        'default': 10,
        'help': 'number of points to sample in a tree during evaluation'
    },
}