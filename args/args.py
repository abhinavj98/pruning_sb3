# Repace bool elements with flags (Bool doesnt work)
args = {
    'args_callback': {
        'eval_freq': {
            'type': int,
            'default': 4000,
            'help': 'frequency of evaluation'
        },
        'n_eval_episodes': {
            'type': int,
            'default': 40,
            'help': 'number of episodes to run during evaluation'
        },
    },

    'args_env': {
        # Reward parameters
        'movement_reward_scale': {
            'type': float,
            'default': 5,
            'help': 'scaling factor for the movement reward'
        },
        'distance_reward_scale': {
            'type': float,
            'default': 0,  # 1/70,
            'help': 'scaling factor for the distance reward'
        },
        'condition_reward_scale': {
            'type': float,
            'default': 0,
            'help': 'scaling factor for the condition reward'
        },
        'terminate_reward_scale': {
            'type': float,
            'default': 0.4,
            'help': 'scaling factor for the terminate reward'
        },
        'collision_reward_scale': {
            'type': float,
            'default': -0.005,
            'help': 'scaling factor for the collision reward'
        },
        'slack_reward_scale': {
            'type': float,
            'default': -0.01,
            'help': 'scaling factor for the slack reward'
        },
        'pointing_orientation_reward_scale': {
            'type': float,
            'default': 2,
            'help': 'scaling factor for the orientation reward'
        },
        'perpendicular_orientation_reward_scale': {
            'type': float,
            'default': 3,
            'help': 'scaling factor for the orientation reward'
        },
        # Gym envirionment parameters

        'max_steps': {
            'type': int,
            'default': 100,
            'help': 'maximum number of steps per episode'
        },
        'use_ik': {
            'action': "store_true",
            'default': True,
            'help': 'whether to render the environment'
        },

        'action_scale': {
            'type': float,
            'default': 0.1,
            'help': 'scaling factor for the action space'
        },
        'action_dim': {
            'type': int,
            'default': 6,
            'help': 'dimension of the action space for the actor network'
        },

    },

    'args_train': {
        'name': {
            'type': str,
            'default': 'trainenv'
        },
        'tree_urdf_path': {
            'type': str,
            'default': './meshes_and_urdf/urdf/trees/envy/train',
            'help': 'path to the train URDF file for the tree environment'
        },
        'tree_obj_path': {
            'type': str,
            'default': './meshes_and_urdf/meshes/trees/envy/train',
            'help': 'path to the train OBJ file for the tree environment'
        },
        'tree_labelled_path': {
            'type': str,
            'default': './meshes_and_urdf/meshes/trees/envy/train_labelled',
            'help': 'path to the train OBJ file for the tree environment'
        },
        'renders': {
            'action': "store_true",
            'default': False,
            'help': 'whether to render the environment'
        },
        'curriculum_distances': {
            'type': tuple,
            'default': (0.8,)
        },
        'curriculum_level_steps': {
            'type': tuple,
            'default': ()
        },
        'tree_count': {
            'type': int,
            'default': 100,
            'help': 'number of trees to load'
        },

    },

    'args_test': {
        'tree_urdf_path': {
            'type': str,
            'default': './meshes_and_urdf/urdf/trees/envy/test',
            'help': 'path to the test URDF file for the tree environment'
        },
        'tree_obj_path': {
            'type': str,
            'default': './meshes_and_urdf/meshes/trees/envy/test',
            'help': 'path to the test OBJ file for the tree environment'
        },
        'tree_labelled_path': {
            'type': str,
            'default': './meshes_and_urdf/meshes/trees/envy/test_labelled',
            'help': 'path to the test OBJ file for the tree environment'
        },
         'renders': {
            'action': "store_true",
            'default': False,
            'help': 'whether to render the environment'
        },
        'num_points': {
            'type': int,
            'default': 40,
            'help': 'number of points to sample in a tree during evaluation'
        },
        'curriculum_distances': {
            'type': tuple,
            'default': (0.8,)
        },
        'curriculum_level_steps': {
            'type': tuple,
            'default': ()
        },
        'name': {
            'type': str,
            'default': 'testenv'
        },
        'tree_count': {
            'type': int,
            'default': 1,
            'help': 'number of trees to load'
        },
        'make_trees': {
            'action': "store_true",
            'default': True,
            'help': 'whether to render the environment'
        },

    },

    'args_eval': {
        'tree_urdf_path': {
            'type': str,
            'default': './meshes_and_urdf/urdf/trees/envy/test',
            'help': 'path to the test URDF file for the tree environment'
        },
        'tree_obj_path': {
            'type': str,
            'default': './meshes_and_urdf/meshes/trees/envy/test',
            'help': 'path to the test OBJ file for the tree environment'
        },
         'renders': {
            'action': "store_true",
            'default': False,
            'help': 'whether to render the environment'
        },

        'curriculum_distances': {
            'type': tuple,
            'default': (0.8,)
        },
        'curriculum_level_steps': {
            'type': tuple,
            'default': ()
        },
        'name': {
            'type': str,
            'default': 'evalenv'
        },

    },

    'args_record': {
        'name': {
            'type': str,
            'default': 'recordenv'
        },
    },

    'args_global': {
        'n_envs': {
            'type': int,
            'default': 16,
            'help': 'number of parallel environments to run'
        },
        'run_name': {
            'type': str,
            'default': 'run'
        },
        'load_path': {
            'type': str,
            'default': None
        },
    },

    'args_policy': {
        # PPO parameters
        'emb_size': {
            'type': int,
            'default': 256,
            'help': 'size of the embedding layer'
        },
        'state_dim': {
            'type': int,
            'default': 72 + 33,
            'help': 'dimension of the state space'
        },
        # Training parameters
        'total_timesteps': {
            'type': int,
            'default': 20_000_000,
            'help': 'total number of timesteps to train for'
        },
        # For logging purposes keep as multiple of episode length
        'steps_per_epoch': {
            'type': int,
            'default': 300,
            'help': 'number of timesteps per epoch'
        },
        'epochs': {
            'type': int,
            'default': 5,
            'help': 'number of epochs to train for'
        },
        'batch_size': {
            'type': int,
            'default': 64,
            'help': 'batch size'
        },
        'learning_rate': {
            'type': float,
            'default': 0.0005,
            'help': 'learning rate'
        },
        'learning_rate_ae': {
            'type': float,
            'default': 0.0005,
            'help': 'learning rate for the autoencoder'
        },
        'log_std_init': {
            'type': float,
            'default': -0.8,#-3.5,
            'help': 'initial value for the log standard deviation'
        },
        'ae_coeff': {
            'type': int,
            'default': 1,
        }
    }
}
