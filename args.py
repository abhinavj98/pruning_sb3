args_dict = {
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
    'N_ENVS': {
        'type': int,
        'default': 8,
        'help': 'number of parallel environments to run'
    },
    'RENDER': {
        'type': bool,
        'default': False,
        'help': 'whether to render the environment'
    },
    'STATE_DIM': {
        'type': int,
        'default': 72+10*3,
        'help': 'dimension of the state space'
    },
}