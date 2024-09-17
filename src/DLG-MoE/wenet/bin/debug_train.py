# Copyright (c) 2021 Mobvoi Inc. (authors: Binbin Zhang)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import argparse
import copy
import os
import torch
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader
# from thop import profile
from debug_path import setup_pythonpath
from wenet.dataset.dataset import Dataset
from wenet.utils.checkpoint import (load_checkpoint, save_checkpoint,
                                    load_trained_modules)
from wenet.utils.executor import Executor
from wenet.utils.file_utils import read_symbol_table, read_non_lang_symbols
from wenet.utils.scheduler import WarmupLR, NoamHoldAnnealing
from wenet.utils.config import override_config
from wenet.utils.init_model import init_model
from wenet.utils.my_logger import get_Mylogger
# from wenet.utils.test import test


def get_args():
    config = "conf/DLG-8e-dynamic.yaml"
    train_data = "data/debug.list"
    # train_data = "data/test_flops.list"
    cv_data = ["data/debug.list"]
    checkpoint = None
    model_dir = "./exp"
    model_dir = "/Users/huanghukai/Documents/exp"
    symbol_table = "data/dict/mix_dict.txt"
    bpe_model = "data/dict/train_960_unigram5000.model"
    enc_init = None
    cmvn = "data/train/global_cmvn"
    parser = argparse.ArgumentParser(description='training your network')
    parser.add_argument('--config', default=config, help='config file')
    parser.add_argument('--data_type',
                        default='raw',
                        choices=['raw', 'shard'],
                        help='train and cv data type')
    parser.add_argument('--train_data', default=train_data, help='train')
    parser.add_argument('--cv_datasets', default=cv_data, help='cv data file1')
    parser.add_argument('--gpu',
                        type=int,
                        default=-1,
                        help='gpu id for this local rank, -1 for cpu')
    parser.add_argument('--model_dir', default=model_dir, help='save')
    parser.add_argument('--checkpoint', default=checkpoint, help='checkpoint')
    parser.add_argument('--num_workers',
                        default=4,
                        type=int,
                        help='num of subprocess workers for reading')
    parser.add_argument('--pin_memory',
                        action='store_true',
                        default=False,
                        help='Use pinned memory buffers used for reading')
    parser.add_argument('--use_amp',
                        action='store_true',
                        default=False,
                        help='Use automatic mixed precision training')
    parser.add_argument('--fp16_grad_sync',
                        action='store_true',
                        default=False,
                        help='Use fp16 gradient sync for ddp')
    parser.add_argument('--cmvn', default=cmvn, help='global cmvn file')
    parser.add_argument('--symbol_table',
                        default=symbol_table,
                        help='model unit symbol table for training')
    parser.add_argument("--non_lang_syms",
                        help="non-linguistic symbol file")
    parser.add_argument('--prefetch',
                        default=100,
                        type=int,
                        help='prefetch number')
    parser.add_argument('--bpe_model',
                        default=bpe_model,
                        type=str,
                        help='bpe model for english part')
    parser.add_argument('--override_config',
                        action='append',
                        default=[],
                        help="override yaml config")
    parser.add_argument("--enc_init",
                        default=enc_init,
                        type=str,
                        help="Pre-trained model to initialize encoder")
    parser.add_argument("--enc_init_mods",
                        default="encoder.",
                        type=lambda s: [str(mod) for mod in s.split(",") if s != ""],
                        help="List of encoder modules \
                        to initialize ,separated by a comma")
    parser.add_argument('--lfmmi_dir',
                        default='',
                        required=False,
                        help='LF-MMI dir')

    args = parser.parse_args()
    return args


def main():
    current_dir = os.getcwd()
    os.chdir(os.path.join(current_dir, 'Language-Group/Language-Group'))
    args = get_args()
    logger = get_Mylogger("train", args.model_dir)
    logger.info(args)
    logger_model = get_Mylogger("model", args.model_dir)
    # os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    torch.manual_seed(777)
    with open(args.config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)
    if len(args.override_config) > 0:
        configs = override_config(configs, args.override_config)
    symbol_table = read_symbol_table(args.symbol_table)
    batch_conf = configs['dataset_conf']['batch_conf']
    train_conf = configs['dataset_conf']
    cv_conf = copy.deepcopy(train_conf)
    cv_conf['speed_perturb'] = False
    cv_conf['spec_aug'] = False
    cv_conf['spec_sub'] = False
    cv_conf['spec_trim'] = False
    cv_conf['shuffle'] = False
    non_lang_syms = read_non_lang_symbols(args.non_lang_syms)
    train_dataset = Dataset(args.data_type, args.train_data, symbol_table,
                            train_conf, args.bpe_model, non_lang_syms, True)
    cv_datasets = []
    cv_dataloaders = []
    for cv_dataset in args.cv_datasets:
        temp_dataset = Dataset(args.data_type,
                               cv_dataset,
                               symbol_table,
                               cv_conf,
                               args.bpe_model,
                               non_lang_syms,
                               partition=False)
        cv_data_loader = DataLoader(temp_dataset,
                                    batch_size=None,
                                    pin_memory=args.pin_memory,
                                    num_workers=args.num_workers,
                                    prefetch_factor=args.prefetch)
        cv_datasets.append(temp_dataset)
        cv_dataloaders.append(cv_data_loader)

    train_data_loader = DataLoader(train_dataset,
                                   batch_size=None,
                                   pin_memory=args.pin_memory,
                                   num_workers=args.num_workers,
                                   prefetch_factor=args.prefetch)

    if 'fbank_conf' in configs['dataset_conf']:
        input_dim = configs['dataset_conf']['fbank_conf']['num_mel_bins']
    else:
        input_dim = configs['dataset_conf']['mfcc_conf']['num_mel_bins']
    vocab_size = len(symbol_table)
    # Save configs to model_dir/train.yaml for inference and export
    configs['input_dim'] = input_dim
    configs['output_dim'] = vocab_size
    configs['cmvn_file'] = args.cmvn
    configs['is_json_cmvn'] = True
    configs['lfmmi_dir'] = args.lfmmi_dir
    saved_config_path = os.path.join(args.model_dir, 'train.yaml')
    with open(saved_config_path, 'w') as fout:
        data = yaml.dump(configs)
        fout.write(data)
    logger.info(configs)
    # Init asr model from configs
    model = init_model(configs)
    num_params = sum(p.numel() for p in model.parameters())
    logger_model.debug('the number of model params: {:,d}'.format(num_params))
    # script_model = torch.jit.script(model)
    # script_model.save(os.path.join(args.model_dir, 'init.zip'))
    executor = Executor()
    # If specify checkpoint, load some info from checkpoint
    if args.checkpoint is not None:
        infos = load_checkpoint(model, args.checkpoint)
    elif args.enc_init is not None:
        logger.debug('load pretrained encoders: {}'.format(args.enc_init))
        infos = load_trained_modules(model, args)
    else:
        infos = {}
    start_epoch = infos.get('epoch', -1) + 1
    cv_loss = infos.get('cv_loss', 0.0)
    step = infos.get('step', -1)

    num_epochs = configs.get('max_epoch', 100)
    model_dir = args.model_dir
    writer = None
    distributed = False
    device = torch.device('cpu')
    model = model.to(device)

    if configs['optim'] == 'adam':
        optimizer = optim.Adam(model.parameters(), **configs['optim_conf'])
    elif configs['optim'] == 'adamw':
        optimizer = optim.AdamW(model.parameters(), **configs['optim_conf'])
    else:
        raise ValueError("unknown optimizer: " + configs['optim'])
    if configs['scheduler'] == 'warmuplr':
        scheduler = WarmupLR(optimizer, **configs['scheduler_conf'])
    elif configs['scheduler'] == 'NoamHoldAnnealing':
        scheduler = NoamHoldAnnealing(optimizer, **configs['scheduler_conf'])
    else:
        raise ValueError("unknown scheduler: " + configs['scheduler'])
    configs['is_distributed'] = distributed
    configs['use_amp'] = args.use_amp
    if start_epoch == 0:
        save_model_path = os.path.join(model_dir, 'init.pt')
        save_checkpoint(model, save_model_path)

    # Start training loop
    executor.step = step
    scheduler.set_step(step)
    # used for pytorch amp mixed precision training
    scaler = None
    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()
    for epoch in range(start_epoch, num_epochs):
        train_dataset.set_epoch(epoch)
        configs['epoch'] = epoch
        lr = optimizer.param_groups[0]['lr']
        logger.debug('Epoch {} TRAIN info lr {}'.format(epoch, lr))
        train_acc_asr, train_seen_utts, train_time, log_line = executor.train(model, optimizer, scheduler, train_data_loader, device, writer, configs, scaler)
        cv_loss_all = []
        wer_asr_all = []
        for cv_data_loader in cv_dataloaders:
            total_loss, num_seen_utts, att_acc_asr, _ = executor.cv(model, cv_data_loader, device, configs)
            wer = (1 - round(att_acc_asr, 4))*100
            wer_asr_all.append(wer)
            cv_loss = total_loss / num_seen_utts
            cv_loss_all.append(cv_loss)
            logger.debug('Epoch {} CV info cv_loss {} att_acc_asr {}'.format(epoch, cv_loss, att_acc_asr))
        save_model_path = os.path.join(model_dir, '{}.pt'.format(epoch))
        save_checkpoint(
            model, save_model_path, {
                'epoch': epoch,
                'batch_conf': batch_conf,
                'moe_conf': configs['moe_conf'],
                'lr': lr,
                'cv_loss': cv_loss,
                'step': executor.step,
                'cv-utterance_nums': num_seen_utts,
                'cv-wer_acc_asr': wer_asr_all,
                'train-utterance_nums': train_seen_utts,
                'train-att_acc_asr': train_acc_asr,
                'train-time': train_time,
                'gpu_nums': 1,
                'loss_config': log_line
            })
        # writer.add_scalar('epoch/cv_loss', cv_loss, epoch)
        # writer.add_scalar('epoch/lr', lr, epoch)


if __name__ == '__main__':
    setup_pythonpath()
    main()
