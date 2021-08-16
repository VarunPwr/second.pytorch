from typing import Any
import pytorch_lightning as pl
import os
import yaml
from data_module import TinyDataModule
from networks import PlNet
from arguments import get_args
from pytorch_lightning import loggers as pl_loggers

args = get_args()
with open(os.path.join(os.getcwd(), 'task_config.yaml'), 'r') as f:
    cfg = yaml.load(f, Loader=yaml.SafeLoader)

task_config = cfg["task"][args.task_name]
input_dict = task_config["input_dict"]
output_dict = task_config["output_dict"]

if args.use_additional_info:
    input_dict.extend(cfg["additional_info"])

dm = TinyDataModule(file_name="mpc_data", batch_size=1024,
                    input_dict=input_dict, output_dict=output_dict)

net = PlNet("mlp", dm.input_size, [
            1000, 1000, 1000], dm.output_size, grad_hook=False, tanh_loss=args.tanh_loss)

if args.use_additional_info:
    logdir = os.path.join(args.logdir, args.task_name + '_additional_info')
else:
    logdir = os.path.join(args.logdir, args.task_name)

tb_logger = pl_loggers.TensorBoardLogger(logdir)

trainer = pl.Trainer(gpus=args.device, weights_summary="full",
                     max_epochs=500, logger=tb_logger, gradient_clip_val=0.5)
# lr_finder = trainer.tuner.lr_find(trainer, net, num_training=100, max_lr=1e-2, min_lr=1e-6)

# # Results can be found in
# lr_finder.results

# # Pick point based on plot, or get suggestion
# new_lr = lr_finder.suggestion()
# print("the suggested lr is ", new_lr)
trainer.fit(net, dm)
