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

dm = TinyDataModule(file_name="mpc_data_1m", batch_size=2048,
                    input_dict=input_dict, output_dict=output_dict)


net = PlNet("mlp", dm.input_size, [500, 500, 500], dm.output_size, grad_hook=False)

if args.use_additional_info:
    logdir = os.path.join(args.logdir, args.task_name + '_additional_info')
else:
    logdir = os.path.join(args.logdir, args.task_name)

tb_logger = pl_loggers.TensorBoardLogger(logdir)

trainer = pl.Trainer(gpus=args.device, weights_summary="full", max_epochs=500, logger=tb_logger)
trainer.fit(net, dm)
