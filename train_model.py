import os
#get name of job from user
batchfile = "#!/bin/bash\n"
job_name = input("Enter job name")
batchfile += "#SBATCH -J " + job_name + "\n"
account_partion_list = [["eecs", "dgx"], ["eecs2", "virl-grp"]]
print("Available accounts and partitions are:"  + str(account_partion_list))
account = input("Enter account/partition index (0 or 1)")
batchfile += "#SBATCH -A " + account_partion_list[int(account)][0] + "\n"
batchfile += "#SBATCH -p " + account_partion_list[int(account)][1] + "\n"
batchfile += "#SBATCH -o " + job_name + ".out\n"
batchfile += "#SBATCH -e " + job_name + ".err\n"
num_days = input("Enter number of days to run the job")
batchfile += "#SBATCH -t " + num_days + "-00:00:00\n"
num_cores = "16"#input("Enter number of cores")
batchfile += "#SBATCH -c " + num_cores + "\n"
num_gpus = "1"#input("Enter number of GPUs")
batchfile += "#SBATCH --gres=gpu:" + num_gpus + "\n"
batchfile += "module load cuda/11.6\n"
batchfile += "cd /nfs/stak/users/jainab/hpc-share/codes/pruning_sb3\n"
branch_name = input("Enter branch name")
# run git checkout branch_name
code = os.system("git checkout " + branch_name)
if code != 0:
    print("Error in checking out branch")
    exit()
env_name = input("Enter conda environment name")
# run conda activate env_name
# code = os.system("source /nfs/stak/users/jainab/hpc-share/anaconda/etc/profile.d/conda.sh")

# code = os.system("")
batchfile += "python train_ppo_test.py"

with open ('run.sh', 'w') as rsh:
    rsh.write(batchfile)

code = os.system('''source /nfs/stak/users/jainab/hpc-share/anaconda/etc/profile.d/conda.sh
                    conda activate {}
                    conda info
                    sbatch run.sh'''.format(env_name))

if code != 0:
    print("Error in activating conda environment")
    exit()

