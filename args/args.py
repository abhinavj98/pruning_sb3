# Repace bool elements with flags (Bool doesnt work)
args = {
    'args_callback': {
        'save_freq': {
            'type': int,
            'default': 4000,
            'help': 'frequency of evaluation'
        },
        'n_eval_episodes': {
            'type': int,
            'default': 10,
            'help': 'number of episodes to run during evaluation'
        },
        'n_eval_orientations': {
            'type': int,
            'default': 1000,
            'help': 'number of orientations to evaluate'
        },
        'n_points_per_orientation': {
            'type': int,
            'default': 3,
            'help': 'number of points to sample per orientation'
        },
        'train_record_freq': {
            'type': int,
            'default': 100,
            'help': 'frequency of recording the training environment'
        },
        'verbose': {
            'type': int,
            'default': 1,
            'help': 'verbosity level'
        },
    },

    'args_env': {
        # Reward parameters
        'movement_reward_scale': {
            'type': float,
            'default': 10,
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
            'default': 2.,
            'help': 'scaling factor for the terminate reward'
        },
        'collision_reward_scale': {
            'type': float,
            'default': -0.05,
            'help': 'scaling factor for the collision reward'
        },
        'slack_reward_scale': {
            'type': float,
            'default': -0.03,
            'help': 'scaling factor for the slack reward'
        },
        'pointing_orientation_reward_scale': {
            'type': float,
            'default': 3,
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
            'default': 120,
            'help': 'maximum number of steps per episode'
        },
        'use_ik': {
            'action': "store_true",
            'default': True,
            'help': 'whether to render the environment'
        },

        'action_scale': {
            'type': float,
            'default': 0.2,
            'help': 'scaling factor for the action space'
        },
        'action_dim': {
            'type': int,
            'default': 6,
            'help': 'dimension of the action space for the actor network'
        },

        'randomize_ur5_pose': {
            'action': "store_true",
            'default': False,
            'help': 'whether to randomize the UR5 pose'
        },
        'randomize_tree_pose': {
            'action': "store_true",
            'default': False,
            'help': 'whether to randomize the tree pose'
        },
        'curriculum_distances': {
            'type': tuple,
            'default': (0.95,)
        },
        'curriculum_level_steps': {
            'type': tuple,
            'default': ()
        },
        'cam_height': {
            'type': int,
            'default': 240,
            'help': 'height of the camera image'
        },
        'cam_width': {
            'type': int,
            'default': 424,
            'help': 'width of the camera image'
        },
        'algo_height': {
            'type': int,
            'default': 240,  # divisible by 8
            'help': 'height of the algorithm image'
        },
        'algo_width': {
            'type': int,
            'default': 424,  # divisible by 8
            'help': 'width of the algorithm image'
        },
        'verbose': {
            'type': int,
            'default': 1,
            'help': 'verbosity level'
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

        'name': {
            'type': str,
            'default': 'testenv'
        },
        'tree_count': {
            'type': int,
            'default': 10,
            'help': 'number of trees to load'
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
        'run_type': {
            'type': str,
            'default': 'train'
        },
        'load_timestep': {
            'type': int,
            'default': 0
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
            'default': 128 + 33,
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
            'default': 100,
            'help': 'number of timesteps per epoch'
        },
        'epochs': {
            'type': int,
            'default': 5,
            'help': 'number of epochs to train for'
        },
        'batch_size': {
            'type': int,
            'default': 32,
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
            'default': -1.2,  # -3.5,
            'help': 'initial value for the log standard deviation'
        },
        'ae_coeff': {
            'type': int,
            'default': 0,
        },
        'verbose': {
            'type': int,
            'default': 1,
            'help': 'verbosity level'
        },
        'use_offline_data': {
            'action': "store_true",
            'default': False,
            'help': 'whether to use offline data'
        },

        'use_online_data': {
            'action': "store_true",
            'default': False,
            'help': 'whether to use online data'
        },

        'mix_data': {
            'action': "store_true",
            'default': False,
            'help': 'whether to mix data'
        },

        'use_online_bc': {
            'action': "store_true",
            'default': False,
            'help': 'whether to use online bc'
        },
        'dont_normalize_advantage': {
            'action': "store_false",
            'default': True,
            'help': 'whether to normalize the advantage'
        },
    },
    'args_baseline': {
        'planner': {
            'type': str,
            'default': 'rrt_connect',
            'help': 'planner to use (rrt_connect, informed_rrt_star)'
        },
        'dataset_type': {
            'type': str,
            'default': 'uniform',
            'help': 'type of dataset to use (uniform, analysis)'
        },
        'results_save_path': {
            'type': str,
            'default': 'results.csv',
            'help': 'path to save the results'
        },
        'save_video': {
            'action': "store_true",
            'default': False,
            'help': 'whether to save the video'
        },
        'shortcutting': {
            'action': "store_true",
            'default': False,
            'help': 'whether to smooth the path'
        },
        'load_file_path': {
            'type': str,
            'default': 'results.csv',
            'help': 'path to load the results to smooth'
        },

    }
}
