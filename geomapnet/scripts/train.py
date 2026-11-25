"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
 
"""
Main training script for MapNet
"""
import set_paths
from common.train import Trainer
from common.optimizer import Optimizer
from common.criterion import PoseNetCriterion, MapNetCriterion,\
  MapNetOnlineCriterion
from models.posenet import PoseNet, MapNet
from dataset_loaders.composite import MF, MFOnline
import os.path as osp
import numpy as np
import argparse
import configparser
import json
import torch
from torch import nn
from torchvision import transforms, models
from torchvision.models import ResNet34_Weights
from pathlib import Path

# ============================================================
# Repository paths
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
GEOMAPNET_DIR = SCRIPT_DIR.parent
REPO_ROOT = GEOMAPNET_DIR.parent

CONFIG_DIR = REPO_ROOT / 'configs'
DATA_DIR = GEOMAPNET_DIR / 'data'
DEEPSLAM_DATA_DIR = DATA_DIR / 'deepslam_data'

parser = argparse.ArgumentParser(description='Training script for PoseNet and'
                                             'MapNet variants')
parser.add_argument('--dataset', type=str, 
                    choices=('7Scenes', 'RobotCar', 'Bebop'),
                    help='Dataset')
parser.add_argument('--scene', type=str, help='Scene name')
parser.add_argument('--config_file', type=str, help='configuration file')
parser.add_argument('--model', choices=('posenet', 'mapnet', 'mapnet++'),
  help='Model to train')
parser.add_argument('--device', type=str, default='0',
  help='value to be set to $CUDA_VISIBLE_DEVICES')
parser.add_argument('--checkpoint', type=str, help='Checkpoint to resume from',
  default=None)
parser.add_argument('--learn_beta', action='store_true',
  help='Learn the weight of translation loss')
parser.add_argument('--learn_gamma', action='store_true',
  help='Learn the weight of rotation loss')
parser.add_argument('--resume_optim', action='store_true',
  help='Resume optimization (only effective if a checkpoint is given')
parser.add_argument('--suffix', type=str, default='',
                    help='Experiment name suffix (as is)')
args = parser.parse_args()

if args.dataset == 'Bebop' and args.model != 'mapnet':
  parser.error(
    'Por ahora, Bebop solamente está integrado con --model mapnet.'
  )

# ============================================================
# Configuration
# ============================================================

settings = configparser.ConfigParser()

config_path = Path(args.config_file)

if not config_path.is_absolute():
  if config_path.parts[0] == 'configs':
    config_path = REPO_ROOT / config_path
  else:
    config_path = CONFIG_DIR / config_path

config_path = config_path.resolve()

if not config_path.is_file():
  raise FileNotFoundError(
    f'Configuration file not found: {config_path}'
  )

with open(config_path, 'r') as f:
  settings.read_string(f.read())

print(f'Using configuration: {config_path}')

section = settings['optimization']
optim_config = {k: json.loads(v) for k,v in section.items() if k != 'opt'}
opt_method = section['opt']
lr = optim_config.pop('lr')
weight_decay = optim_config.pop('weight_decay')

section = settings['hyperparameters']
dropout = section.getfloat('dropout')
color_jitter = section.getfloat('color_jitter', 0)
sax = 0.0
saq = section.getfloat('beta')
if args.model.find('mapnet') >= 0:
  skip = section.getint('skip')
  real = section.getboolean('real')
  variable_skip = section.getboolean('variable_skip')
  srx = 0.0
  srq = section.getfloat('gamma')
  steps = section.getint('steps')
if args.model.find('++') >= 0:
  vo_lib = section.get('vo_lib', 'orbslam')
  print('Using {:s} VO'.format(vo_lib))

section = settings['training']
seed = section.getint('seed')

# model
feature_extractor = models.resnet34(weights=ResNet34_Weights.DEFAULT)
posenet = PoseNet(feature_extractor, droprate=dropout, pretrained=True,
                  filter_nans=(args.model=='mapnet++'))
if args.model == 'posenet':
  model = posenet
elif args.model.find('mapnet') >= 0:
  model = MapNet(mapnet=posenet)
else:
  raise NotImplementedError

# loss function
if args.model == 'posenet':
  train_criterion = PoseNetCriterion(sax=sax, saq=saq, learn_beta=args.learn_beta)
  val_criterion = PoseNetCriterion()
elif args.model.find('mapnet') >= 0:
  kwargs = dict(sax=sax, saq=saq, srx=srx, srq=srq, learn_beta=args.learn_beta,
                learn_gamma=args.learn_gamma)
  if args.model.find('++') >= 0:
    kwargs = dict(kwargs, gps_mode=(vo_lib=='gps') )
    train_criterion = MapNetOnlineCriterion(**kwargs)
    val_criterion = MapNetOnlineCriterion()
  else:
    train_criterion = MapNetCriterion(**kwargs)
    val_criterion = MapNetCriterion()
else:
  raise NotImplementedError

# optimizer
param_list = [{'params': model.parameters()}]
if args.learn_beta and hasattr(train_criterion, 'sax') and \
    hasattr(train_criterion, 'saq'):
  param_list.append({'params': [train_criterion.sax, train_criterion.saq]})
if args.learn_gamma and hasattr(train_criterion, 'srx') and \
    hasattr(train_criterion, 'srq'):
  param_list.append({'params': [train_criterion.srx, train_criterion.srq]})
optimizer = Optimizer(params=param_list, method=opt_method, base_lr=lr,
  weight_decay=weight_decay, **optim_config)

# ============================================================
# Dataset statistics
# ============================================================

if args.dataset == 'Bebop':
  # La ubicación definitiva de las estadísticas de Bebop
  # se establecerá después de verificar su estructura real.
  stats_candidates = [
    DATA_DIR / 'Bebop' / args.scene / 'stats.txt',
    DATA_DIR / 'Bebop' / 'bebop' / args.scene / 'stats.txt',
    DEEPSLAM_DATA_DIR / 'Bebop' / args.scene / 'stats.txt',
    DEEPSLAM_DATA_DIR / 'Bebop' / 'bebop' / args.scene / 'stats.txt',
  ]

  stats_file = next(
    (path for path in stats_candidates if path.is_file()),
    None
  )

  if stats_file is None:
    raise FileNotFoundError(
      'Could not find Bebop stats.txt.\n'
      'Checked the following locations:\n' +
      '\n'.join(str(p) for p in stats_candidates)
    )

else:
  stats_file = DATA_DIR / args.dataset / args.scene / 'stats.txt'

  if not stats_file.is_file():
    raise FileNotFoundError(
      f'Dataset statistics not found: {stats_file}'
    )

stats = np.loadtxt(stats_file)

print(f'Using dataset statistics: {stats_file}')

# Crop size (no se utiliza directamente en este script)
data_root = DATA_DIR / args.dataset
crop_size_file = data_root / 'crop_size.txt'

if osp.isfile(crop_size_file):
  crop_size = tuple(np.loadtxt(crop_size_file).astype(int))
else:
  crop_size = None
  print('No se encontró crop_size.txt; se continuará sin él.')

# transformers
tforms = [transforms.Resize(256)]
if color_jitter > 0:
  assert color_jitter <= 1.0
  print('Using ColorJitter data augmentation')
  tforms.append(transforms.ColorJitter(brightness=color_jitter,
    contrast=color_jitter, saturation=color_jitter, hue=0.5))
tforms.append(transforms.ToTensor())
tforms.append(transforms.Normalize(mean=stats[0], std=np.sqrt(stats[1])))
data_transform = transforms.Compose(tforms)
target_transform = transforms.Lambda(lambda x: torch.from_numpy(x).float())

# datasets
# ============================================================
# Dataset loading
# ============================================================

# Dataset directory
if args.dataset == 'Bebop':
  data_dir = DEEPSLAM_DATA_DIR / 'Bebop' / 'bebop'
else:
  data_dir = DEEPSLAM_DATA_DIR / args.dataset

if not data_dir.is_dir():
  raise FileNotFoundError(
    f'Dataset directory not found: {data_dir}'
  )

kwargs = dict(
  scene=args.scene,
  data_path=str(data_dir),
  transform=data_transform,
  target_transform=target_transform,
  seed=seed
)

print(f'Using dataset directory: {data_dir}')

if args.model == 'posenet':
  if args.dataset == '7Scenes':
    from dataset_loaders.seven_scenes import SevenScenes
    train_set = SevenScenes(train=True, **kwargs)
    val_set = SevenScenes(train=False, **kwargs)
  elif args.dataset == 'RobotCar':
    from dataset_loaders.robotcar import RobotCar
    train_set = RobotCar(train=True, **kwargs)
    val_set = RobotCar(train=False, **kwargs)
  else:
    raise NotImplementedError
elif args.model.find('mapnet') >= 0:
  kwargs = dict(kwargs, dataset=args.dataset, skip=skip, steps=steps,
    variable_skip=variable_skip)
  if args.model.find('++') >= 0:
    train_set = MFOnline(vo_lib=vo_lib, gps_mode=(vo_lib=='gps'), **kwargs)
    val_set = None
  else:
    train_set = MF(train=True, real=real, **kwargs)
    val_set = MF(train=False, real=real, **kwargs)
else:
  raise NotImplementedError

# trainer
config_name = config_path.stem
experiment_name = '{:s}'.format(config_name)
experiment_name += args.suffix
# Build experiment output directory
dataset_folder = '7scenes' if args.dataset == '7Scenes' else args.dataset

experiment_dir = (
    REPO_ROOT
    / 'experiments'
    / dataset_folder
    / args.scene
    / experiment_name
)

experiment_dir.mkdir(parents=True, exist_ok=True)

print('Experiment output directory:', experiment_dir)
trainer = Trainer(model, optimizer, train_criterion, str(config_path),
                  experiment_name, train_set, val_set, device=args.device,
                  checkpoint_file=args.checkpoint,
                  resume_optim=args.resume_optim, val_criterion=val_criterion,
                  output_dir=str(experiment_dir))
lstm = args.model == 'vidloc'
trainer.train_val(lstm=lstm)
