from typing import Any
import pytorch_lightning as pl
import os
import yaml
from data_module import TinyDataModule
from networks import PlNet
from arguments import get_args

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


# net = PlNet("mlp", dm.state_size, [500, 500], 4 * 13 * 13, dimension_size=3, grad_hook=False)

# trainer = pl.Trainer(gpus=1, weights_summary="full", max_epochs=10000)
# trainer.fit(net, dm)
