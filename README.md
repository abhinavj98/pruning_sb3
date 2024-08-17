How to make trees

https://github.com/OSUrobotics/treesim_lpy

python dataset_maker/make_n.py
Move .ply files to meshes_and_urdf/meshes/trees/envy/ply
python2 testfiles/ply_to_obj.py

Move the created object files to meshes_and_urdf/meshes/trees/envy/train
and train_labelled

run create_urdf_from_obj.py

Blender - Run meshes_and_urdf/add_texture_to_tree.blend

Move tree_0 to test

In all .mtl files change the path to the texture to the correct path (Relative and remove Kd Ks)

python .\training_files\train_ppo_lstm.py --args_global_n_envs 19 --args_policy_learning_rate 0.0005
--args_policy_learning_rate_ae 0.0005 --args_policy_steps_per_epoch 300 --args_global_run_name new_try_latest
--args_env_use_ik --args_env_randomize_ur5_pose --args_env_randomize_tree_pose


To create expert dataset:

python .\baselines\run_baseline.py --args_global_n_envs 10  --args_env_verbose 1 --args_callback_n_eval_orientations 200 --args_callback_n_points_per_orientation 6 --args_baseline_results_save_path rrt_connect_paths_goal_new
 This generates RRT Paths and stores them to a csv file

 python .\baselines\load_and_smooth_paths.py  --args_global_n_envs 5 --args_baseline_load_file_path rrt_connect_paths_goal_new^

Load the paths and smooth them and convert to velocities and store them as pkl

