import numpy as np
import torch
from model.trainer.fsl_trainer import FSLTrainer
from model.utils import (
    pprint, set_gpu,
    get_command_line_parser,
    postprocess_args,
)
import json

# from ipdb import launch_ipdb_on_exception

if __name__ == '__main__':
    parser = get_command_line_parser()
    args = postprocess_args(parser.parse_args())
    # with launch_ipdb_on_exception():

    if args.config:
        config_dict = json.load(open(args.config, 'rb'))
        a = vars(args)
        a.update(config_dict)
    pprint(vars(args))

    set_gpu(args.gpu)
    trainer = FSLTrainer(args)

    if args.test is None:
        trainer.train()
    trainer.evaluate_test()
    trainer.final_record()
    print(args.save_path)



