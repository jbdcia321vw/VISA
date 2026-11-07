# coding: utf-8
# @email: enoche.chow@gmail.com

"""
Main entry
# UPDATED: 2022-Feb-15
##########################
"""

import os
import ast
import argparse
from utils.quick_start import quick_start
os.environ['NUMEXPR_MAX_THREADS'] = '48'


def _parse_bool(v):
    if isinstance(v, bool):
        return v
    return str(v).lower() in ('true', '1', 'yes')

 
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='VISA', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='clothing', help='name of datasets')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--cfg', nargs='*', default=[],
                        help='extra key=value config overrides, e.g. --cfg uu_align_reg=0.01')

    args, _ = parser.parse_known_args() 
 
    config_dict = {
        'gpu_id': args.gpu,
    }
    for kv in args.cfg:
        if '=' in kv:
            k, v = kv.split('=', 1)
            lowered = v.lower()
            if lowered in ('true', 'false'):
                config_dict[k] = lowered == 'true'
            elif lowered in ('none', 'null'): 
                config_dict[k] = None
            else:
                try:
                    config_dict[k] = ast.literal_eval(v)
                except (ValueError, SyntaxError):
                    config_dict[k] = v
        else:
            config_dict[kv] = True

    quick_start(model=args.model, dataset=args.dataset,
                config_dict=config_dict, save_model=True)
