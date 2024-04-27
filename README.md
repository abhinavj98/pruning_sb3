How to make trees 

https://github.com/OSUrobotics/treesim_lpy

python dataset_maker/make_n.py
Move .ply files to meshes_and_urdf/meshes/trees/envy/envy_ply
python2 testfiles/ply_to_obj.py

python .\training_files\train_ppo_lstm.py --args_global_n_envs 19 --args_policy_learning_rate 0.0005 --args_policy_learning_rate_ae 0.0005 --args_policy_steps_per_epoch 300  --args_global_run_name new_try_latest --args_env_use_ik --args_env_randomize_ur5_pose --args_env_randomize_tree_pose
