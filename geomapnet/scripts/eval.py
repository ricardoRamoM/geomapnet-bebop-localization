"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
 
import set_paths
from models.posenet import PoseNet, MapNet
from common.train import load_state_dict, step_feedfwd
from common.pose_utils import optimize_poses, quaternion_angular_error, qexp,\
  calc_vos_safe_fc, calc_vos_safe
from dataset_loaders.composite import MF
import argparse
import os
import os.path as osp
import sys
import numpy as np
import matplotlib
DISPLAY = 'DISPLAY' in os.environ
if not DISPLAY:
  matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import configparser
import torch.cuda
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False
from torch.utils.data import DataLoader
from torchvision import transforms, models
import pickle
from pathlib import Path
import json
import csv

from common.evaluation_metrics import (
    compute_pose_errors,
    summarize_errors
)

# Resolve paths independently of the current working directory.
SCRIPT_DIR = Path(__file__).resolve().parent
GEOMAPNET_DIR = SCRIPT_DIR.parent
REPO_ROOT = GEOMAPNET_DIR.parent

DATA_DIR = GEOMAPNET_DIR / 'data'
DEEPSLAM_DATA_DIR = DATA_DIR / 'deepslam_data'

# config
parser = argparse.ArgumentParser(description='Evaluation script for PoseNet and'
                                             'MapNet variants')
parser.add_argument('--dataset', type=str, 
                    choices=('7Scenes', 'RobotCar', 'Bebop'),
                    help='Dataset')
parser.add_argument('--scene', type=str, help='Scene name')
parser.add_argument('--weights', type=str, help='trained weights to load')
parser.add_argument('--model', type=str,
  choices=('posenet', 'mapnet', 'mapnet++'), required=True,
  help='Model to use (mapnet includes both MapNet and MapNet++ since their'
       'evluation process is the same and they only differ in the input weights'
       'file')
parser.add_argument('--device', type=str, default='0', help='GPU device(s)')
parser.add_argument('--config_file', type=str, help='configuration file')
parser.add_argument('--val', action='store_true', help='Plot graph for val')
parser.add_argument(
  '--output_dir',
  type=str,
  default=None,
  help='Output directory for evaluation results. '
       'If omitted, results are saved under eval_results/<dataset>/<scene>/'
)
parser.add_argument(
    '--eval_name',
    type=str,
    default=None,
    help='Custom name for the evaluation output directory. '
         'Defaults to the checkpoint filename.'
)
parser.add_argument('--pose_graph', action='store_true',
  help='Turn on Pose Graph Optimization')
args = parser.parse_args()



if 'CUDA_VISIBLE_DEVICES' not in os.environ:
  os.environ['CUDA_VISIBLE_DEVICES'] = args.device

settings = configparser.ConfigParser()
with open(args.config_file, 'r') as f:
  settings.read_string(f.read())
seed = settings.getint('training', 'seed')
section = settings['hyperparameters']
dropout = section.getfloat('dropout')
if (args.model.find('mapnet') >= 0) or args.pose_graph:
  steps = section.getint('steps')
  skip = section.getint('skip')
  real = section.getboolean('real')
  variable_skip = section.getboolean('variable_skip')
  fc_vos = args.dataset == 'RobotCar'
  if args.pose_graph:
    vo_lib = section.get('vo_lib')
    sax = section.getfloat('s_abs_trans', 1)
    saq = section.getfloat('s_abs_rot', 1)
    srx = section.getfloat('s_rel_trans', 20)
    srq = section.getfloat('s_rel_rot', 20)

# model
feature_extractor = models.resnet34(weights=None)
posenet = PoseNet(feature_extractor, droprate=dropout, pretrained=False)
if (args.model.find('mapnet') >= 0) or args.pose_graph:
  model = MapNet(mapnet=posenet)
else:
  model = posenet
model.eval()

# loss functions
t_criterion = lambda t_pred, t_gt: np.linalg.norm(t_pred - t_gt)
q_criterion = quaternion_angular_error

# load weights
weights_filename = osp.expanduser(args.weights)
if osp.isfile(weights_filename):
  loc_func = lambda storage, loc: storage
  checkpoint = torch.load(weights_filename, map_location=loc_func, weights_only=False)
  load_state_dict(model, checkpoint['model_state_dict'])
  print('Loaded weights from {:s}'.format(weights_filename))
else:
  print('Could not load weights from {:s}'.format(weights_filename))
  sys.exit(-1)

# Dataset directory
if args.dataset == 'Bebop':
  data_dir = DEEPSLAM_DATA_DIR / 'Bebop' / 'bebop'
else:
  data_dir = DEEPSLAM_DATA_DIR / args.dataset

if not data_dir.is_dir():
  raise FileNotFoundError(
      'Dataset directory not found: {}'.format(data_dir)
  )

print('Using dataset directory:', data_dir)

# Dataset statistics
# 7Scenes stores statistics separately from the original image dataset.
if args.dataset == '7Scenes':
  stats_dir = DATA_DIR / '7Scenes' / args.scene
else:
  stats_dir = data_dir / args.scene

stats_filename = stats_dir / 'stats.txt'

if not stats_filename.is_file():
  raise FileNotFoundError(
      'Dataset statistics not found: {}'.format(stats_filename)
  )

stats = np.loadtxt(stats_filename)

print('Using dataset statistics:', stats_filename)

# transformer
data_transform = transforms.Compose([
  transforms.Resize(256),
  transforms.ToTensor(),
  transforms.Normalize(mean=stats[0], std=np.sqrt(stats[1]))])
target_transform = transforms.Lambda(lambda x: torch.from_numpy(x).float())

# Read mean and standard deviation for un-normalizing predictions
pose_stats_file = stats_dir / 'pose_stats.txt'

if not pose_stats_file.is_file():
  raise FileNotFoundError(
      'Pose statistics not found: {}'.format(pose_stats_file)
  )

pose_m, pose_s = np.loadtxt(pose_stats_file)  # mean and stdev

print('Using pose statistics:', pose_stats_file)

# dataset
train = not args.val
if train:
  print('Running {:s} on TRAIN data'.format(args.model))
else:
  print('Running {:s} on VAL data'.format(args.model))

kwargs = dict(scene=args.scene, data_path=data_dir, train=train,
  transform=data_transform, target_transform=target_transform, seed=seed)
if (args.model.find('mapnet') >= 0) or args.pose_graph:
  if args.pose_graph:
    assert real
    kwargs = dict(kwargs, vo_lib=vo_lib)
  vo_func = calc_vos_safe_fc if fc_vos else calc_vos_safe
  data_set = MF(dataset=args.dataset, steps=steps, skip=skip, real=real,
                variable_skip=variable_skip, include_vos=args.pose_graph,
                vo_func=vo_func, no_duplicates=False, **kwargs)
  L = len(data_set.dset)
elif args.dataset == '7Scenes':
  from dataset_loaders.seven_scenes import SevenScenes
  data_set = SevenScenes(**kwargs)
  L = len(data_set)
elif args.dataset == 'RobotCar':
  from dataset_loaders.robotcar import RobotCar
  data_set = RobotCar(**kwargs)
  L = len(data_set)
else:
  raise NotImplementedError

# loader (batch_size MUST be 1)
batch_size = 1
assert batch_size == 1
loader = DataLoader(data_set, batch_size=batch_size, shuffle=False,
                    num_workers=5, pin_memory=True)

# activate GPUs
CUDA = torch.cuda.is_available()
torch.manual_seed(seed)
if CUDA:
  torch.cuda.manual_seed(seed)
  model.cuda()

pred_poses = np.zeros((L, 7))  # store all predicted poses
targ_poses = np.zeros((L, 7))  # store all target poses

# Original image indices that receive a prediction
valid_indices = []

# inference loop
for batch_idx, (data, target) in enumerate(loader):
  if batch_idx % 200 == 0:
    print('Image {:d} / {:d}'.format(batch_idx, len(loader)))

  # indices into the global arrays storing poses
  if args.dataset == 'Bebop':
    # MapNet receives clips, so recover the original image indices.
    idx = data_set.get_indices(batch_idx)
    idx = int(idx[len(idx) // 2])
  elif (args.model.find('vid') >= 0) or args.pose_graph:
    idx = data_set.get_indices(batch_idx)
    idx = int(idx[len(idx) // 2])
  else:
    idx = batch_idx

  # output : 1 x 6 or 1 x STEPS x 6
  _, output = step_feedfwd(data, model, CUDA, train=False)
  s = output.size()
  output = output.cpu().data.numpy().reshape((-1, s[-1]))
  target = target.numpy().reshape((-1, s[-1]))
  
  # normalize the predicted quaternions
  q = [qexp(p[3:]) for p in output]
  output = np.hstack((output[:, :3], np.asarray(q)))
  q = [qexp(p[3:]) for p in target]
  target = np.hstack((target[:, :3], np.asarray(q)))

  if args.pose_graph:  # do pose graph optimization
    kwargs = {'sax': sax, 'saq': saq, 'srx': srx, 'srq': srq}
    # target includes both absolute poses and vos
    vos = target[len(output):]
    target = target[:len(output)]
    output = optimize_poses(pred_poses=output, vos=vos, fc_vos=fc_vos, **kwargs)

  # un-normalize the predicted and target translations
  output[:, :3] = (output[:, :3] * pose_s) + pose_m
  target[:, :3] = (target[:, :3] * pose_s) + pose_m

  # take the middle prediction
  pred_poses[idx, :] = output[len(output)//2]
  targ_poses[idx, :] = target[len(target)//2]

  # Keep track of images that were actually evaluated
  valid_indices.append(idx)

# calculate losses
# Calculate losses only for images that were actually evaluated
# -------------------------------------------------------------------------
# Calculate evaluation errors
# -------------------------------------------------------------------------

valid_indices = np.asarray(valid_indices, dtype=int)

pred_eval = pred_poses[valid_indices]
targ_eval = targ_poses[valid_indices]

# Single source of truth for translation and orientation errors.
t_loss, q_loss = compute_pose_errors(pred_eval, targ_eval)

# Summary statistics.
metrics = summarize_errors(t_loss, q_loss)

print('Evaluated images: {:d} / {:d}'.format(
    len(valid_indices), L
))

print(
    'Error in translation: median {:3.2f} m, mean {:3.2f} m\n'
    'Error in rotation: median {:3.2f} degrees, mean {:3.2f} degrees'.format(
        metrics['position']['median_m'],
        metrics['position']['mean_m'],
        metrics['orientation']['median_deg'],
        metrics['orientation']['mean_deg']
    )
)

#eval_func = np.mean if args.dataset == 'RobotCar' else np.median
#eval_str  = 'Mean' if args.dataset == 'RobotCar' else 'Median'
#t_loss = eval_func(t_loss)
#q_loss = eval_func(q_loss)
#print '{:s} error in translation = {:3.2f} m\n' \
#      '{:s} error in rotation    = {:3.2f} degrees'.format(eval_str, t_loss,


# -------------------------------------------------------------------------
# Create 3D trajectory visualization
# Original NVIDIA MapNet visualization, extended to Bebop.
# -------------------------------------------------------------------------

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

plt.subplots_adjust(left=0, bottom=0, right=1, top=1)

# Reduce the number of plotted points for large datasets.
ss = max(1, int(len(valid_indices) / 1000))

# Predicted and ground-truth positions.
x = np.vstack((
    pred_eval[::ss, 0],
    targ_eval[::ss, 0]
))

y = np.vstack((
    pred_eval[::ss, 1],
    targ_eval[::ss, 1]
))

z = np.vstack((
    pred_eval[::ss, 2],
    targ_eval[::ss, 2]
))

# Draw blue lines connecting each prediction to its ground truth.
for xx, yy, zz in zip(x.T, y.T, z.T):
    ax.plot(
        xx,
        yy,
        zs=zz,
        c='b'
    )

# Predicted positions (red).
ax.scatter(
    x[0, :],
    y[0, :],
    zs=z[0, :],
    c='r',
    depthshade=0
)

# Ground-truth positions (green).
ax.scatter(
    x[1, :],
    y[1, :],
    zs=z[1, :],
    c='g',
    depthshade=0
)

# Original NVIDIA viewing angle.
ax.view_init(azim=119, elev=13)



# -------------------------------------------------------------------------
# Save evaluation results
# -------------------------------------------------------------------------

# Determine evaluation name from the checkpoint filename.
if args.eval_name is not None:
    eval_name = args.eval_name
else:
    checkpoint_name = Path(args.weights).name

    # Remove common checkpoint extensions.
    for extension in ('.pth.tar', '.pt.tar', '.pth', '.pt'):
        if checkpoint_name.endswith(extension):
            checkpoint_name = checkpoint_name[:-len(extension)]
            break

    eval_name = checkpoint_name

# Dataset folder naming.
dataset_folder = '7scenes' if args.dataset == '7Scenes' else args.dataset

# Default output directory:
# eval_results/<dataset>/<scene>/<evaluation_name>/
if args.output_dir is None:
    output_dir = (
        REPO_ROOT
        / 'eval_results'
        / dataset_folder
        / args.scene
        / eval_name
    )
else:
    output_dir = Path(args.output_dir).expanduser().resolve()

# Create output directory if it does not exist.
output_dir.mkdir(parents=True, exist_ok=True)
print('Evaluation output directory:', output_dir)

# Generate result filenames.
model_name = args.model

if args.weights.find('++') >= 0:
    model_name += '++'

if args.pose_graph:
    model_name += '_pgo_{:s}'.format(vo_lib)

experiment_name = '{:s}_{:s}_{:s}'.format(
    args.dataset,
    args.scene,
    model_name
)

image_filename = output_dir / '{:s}.png'.format(experiment_name)
result_filename = output_dir / '{:s}.pkl'.format(experiment_name)

# Save figure.
fig.savefig(str(image_filename))
print('{} saved'.format(image_filename))

# Save predictions and ground truth.
with open(result_filename, 'wb') as f:
    pickle.dump(
        {
            'targ_poses': targ_eval,
            'pred_poses': pred_eval,
            'valid_indices': valid_indices
        },
        f
    )

print('{} written'.format(result_filename))

# -------------------------------------------------------------------------
# Save evaluation metrics
# -------------------------------------------------------------------------

# Evaluation metadata.
metrics['metadata'] = {
    'dataset': args.dataset,
    'scene': args.scene,
    'model': args.model,
    'checkpoint': str(Path(args.weights).expanduser().resolve()),
    'checkpoint_name': Path(args.weights).name,
    'split': 'val' if args.val else 'train',
    'pose_graph': bool(args.pose_graph),
    'num_dataset_images': int(L),
}

# Save summary metrics.
metrics_filename = output_dir / 'metrics.json'

with open(metrics_filename, 'w', encoding='utf-8') as f:
    json.dump(metrics, f, indent=4, ensure_ascii=False)

print('{} written'.format(metrics_filename))


# Save per-image errors.
errors_filename = output_dir / 'per_image_errors.csv'

with open(errors_filename, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            'valid_index',
            'translation_error_m',
            'orientation_error_deg'
        ]
    )

    writer.writeheader()

    for idx, translation_error, orientation_error in zip(
        valid_indices,
        t_loss,
        q_loss
    ):
        writer.writerow({
            'valid_index': int(idx),
            'translation_error_m': float(translation_error),
            'orientation_error_deg': float(orientation_error)
        })

print('{} written'.format(errors_filename))

# Display interactive Matplotlib window after saving the results.
if DISPLAY:
    plt.show(block=True)