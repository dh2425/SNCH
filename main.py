import torch
import numpy as np
import argparse
from CFGH import GCFH
from utils import logger

if __name__ == '__main__':
    seeds = 2023
    torch.manual_seed(seeds) 
    torch.cuda.manual_seed(seeds)  
    torch.cuda.manual_seed_all(seeds)  
    np.random.seed(seeds)


    parser = argparse.ArgumentParser()  
    parser.add_argument('--dataset', type=str, default='mscoco', help='Dataset name: mscoco, flickr25k, nus-wide')
    parser.add_argument('--data_pth', type=str, default='data')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--feat_lens',type=int, default=512)
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--hash_lens', type=int, default=64)
    parser.add_argument('--device', type=int, default=0, help='cuda device')
    parser.add_argument('--is_train', type=bool, default=True)
    parser.add_argument('--model_dir', type=str, default='./checkpoints')
    parser.add_argument('--result_dir', type=str, default='./result')
    parser.add_argument('--freq', type=int, default=1,help='eval interval')
    parser.add_argument('--evl_epoch', type=int, default=10,help='Number of epochs to start eval')


    parser.add_argument('--warmup', type=int, default=2, help='')
    parser.add_argument('--max_modify', type=float, default=6)
    parser.add_argument('--epsilon', type=float, default=0.05)
    parser.add_argument('--T', type=int, default=2)
    parser.add_argument('--α', type=float, default=4.0)
    parser.add_argument('--threshld', type=float, default=0.98)

    config = parser.parse_args()


    log = logger(config)
    log.info('--- config: {}'.format(config))

    task = str(config.hash_lens) + " bits"
    log.info('=============== {}--{}--Total epochs:{} ==============='.format(config.dataset, task, config.epoch))
    model = GCFH(log ,config)
    if True:
        log.info('...Training is beginning...')
        model.train()
    else:
        log.info('...Test is beginning...')
        model.test()


