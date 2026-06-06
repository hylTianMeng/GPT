out_dir = 'out-wikitext'
eval_interval = 250 
eval_iters = 200
log_interval = 10 

always_save_checkpoint = False

wandb_log = False 
wandb_project = 'wikitext_large'
wandb_run_name = 'mini-gpt'

dataset = 'wikitext_large'
gradient_accumulation_steps = 4
batch_size = 16
block_size = 256 

n_layer = 8
n_head = 8
n_embd = 512
dropout = 0.2

learning_rate = 1e-3 
max_iters = 20000
lr_decay_iters = 20000 
min_lr = 1e-4 
beta2 = 0.99 

warmup_iters = 100 

