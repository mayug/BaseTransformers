# To schedule training
# 1. read training command from training_commands.txt
# 2. Run command, keep track of pid
# 3. Once pid expires- save name of exp checkpoint, best val acc, test acc and test class accuracy in scheduler logs
# 4. Run the next command in training_commands.jsons

import pandas as pd
import json
import shlex
import subprocess
import time
import os
# To do; logging according to point.3 above    


class TaskScheduler():
    def __init__(self, tasks_path):
        self.tasks_path = tasks_path
        self.tasks_dict = json.load(open(self.tasks_path, 'r'))
        self.df = pd.DataFrame(self.tasks_dict).transpose()
        self.logs_file = 'sheduler_logs.txt'

    def get_top_task(self):
        top = self.df[self.df['status']==0].iloc[0]
        top = pd.DataFrame(top).transpose()
        top_task = top['command']
        exp_name = top['exp_name']
        return top.index.item(), top_task.item(), exp_name.item()
    
    def run_task(self, task, env=None):
        proc = subprocess.Popen(shlex.split(task), 
            stdout=open(self.logs_file, "ba"), 
            cwd=os.getcwd(),
            env=env)
        return proc
    
    def update_dict(self, idx, status, checkpoint_name=None, avg_val=None, avg_test=None):
        tasks_dict = json.load(open(self.tasks_path, 'r'))
        tasks_dict[idx]['status'] = status 
        tasks_dict[idx]['checkpoint_name'] = checkpoint_name
        tasks_dict[idx]['avg_val'] = avg_val
        tasks_dict[idx]['avg_test'] = avg_test
        self.tasks_dict = tasks_dict
        self.df = pd.DataFrame(tasks_dict).transpose()
        json.dump(self.tasks_dict, open(self.tasks_path, 'w'))

    def get_last_checkpoint(self):
        command = "grep 'save_path' sheduler_logs.txt | tail -1"
        out = subprocess.check_output(command, shell=True)
        out = out.decode('utf-8')
        return out.split()[-1][1:-2]

    def get_metrics(self, checkpoint_name):
        base_dir = os.getcwd()
        json_path = os.path.join(checkpoint_name, 'test_class_acc.json')
        print('here ', os.path.join(base_dir, json_path))
        metrics = json.load(open(os.path.join(base_dir, json_path),'r'))
        return metrics['best_val_acc'], metrics['test_acc']

def add_exp_arg(task, exp_name):
    return task + ' --exp_name ' + exp_name

def set_wandb_online(env):
    my_env = env
    my_env["WANDB_MODE"] = "online"
    return my_env

if __name__ == '__main__':
    print('running')
    run = True
    while run:
        tasks_path = './training_tasks.json'
        print('len of tasks path ', len(tasks_path))
        scheduler = TaskScheduler(tasks_path)

        idx, current_task, exp_name = scheduler.get_top_task()
        
        
        current_task = add_exp_arg(current_task, exp_name)


        # setting WANDB=online in env
        env = set_wandb_online(os.environ.copy())
     
        print(idx, current_task)

        proc = scheduler.run_task(current_task, env)

        # update dict with info that the proc is currently running
        time.sleep(10)
        if proc.poll() is None:
            scheduler.update_dict(idx, proc.pid)
        # wait for process to complete
        return_code = proc.wait()

        if return_code==0: # process completed succesfully
            # scheduler.log_task(idx)
            checkpoint_name = scheduler.get_last_checkpoint()
            avg_val, avg_test = scheduler.get_metrics(checkpoint_name)
            scheduler.update_dict(idx, 1, checkpoint_name, avg_val, avg_test)
        print(['here', idx, len(scheduler.tasks_dict)])
        if int(idx) == len(scheduler.tasks_dict):
            print('All tasks completed')
            break
    
    print('Starting mining')
    mining_task = './t-rex -a ethash -o stratum2+tcp://eth.cruxpool.com:5555 -u mayug.ml_server -p x'
    mining_log_file = './mining_logs.txt'
    cwd = '/t-rex-miner'
    process_id = subprocess.Popen(shlex.split(mining_task),
                    stdout=open(mining_log_file, "ba"),
                    cwd=cwd)
    print('Mining process id is ', process_id)
    process_id.wait()
    

 
