import sys
sys.path.append('..')
import argparse

import torch
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

from torch.utils import tensorboard
import numpy as np 
from models.MAML import MAML
from kitti import DataGenerator
torch.autograd.set_detect_anomaly(True)

def main(args):

    if args.device == "gpu" and torch.backends.mps.is_available() and torch.backends.mps.is_built():

        DEVICE = "cpu"
    elif args.device == "gpu" and torch.cuda.is_available():
        DEVICE = "cuda"
    else:
        DEVICE = "cpu"

    print("Using device: ", DEVICE)

    log_dir = args.log_dir
    if log_dir is None:
        log_dir = f'./logs/maml/kitti.way_{args.num_way}.support_{args.num_support}.query_{args.num_query}.inner_steps_{args.num_inner_steps}.inner_lr_{args.inner_lr}.learn_inner_lrs_{args.learn_inner_lrs}.outer_lr_{args.outer_lr}.batch_size_{args.batch_size}.bbox_type_{args.bounding_box_mode}.depth_{args.depth_supervision}'
    print(f'log_dir: {log_dir}')
    
    writer = tensorboard.SummaryWriter(log_dir=log_dir)

    maml = MAML(
        model_name='DETR',
        num_classes=args.num_way,   #Same as Number of classes 
        num_inner_steps=args.num_inner_steps,
        inner_lr=args.inner_lr,
        learn_inner_lrs=args.learn_inner_lrs,
        outer_lr=args.outer_lr,
        outer_steps= args.num_train_iterations,
        log_dir=log_dir,
        device=DEVICE,
        writer = writer,
        bounding_box_mode = args.bounding_box_mode,
        depth_supervision = args.depth_supervision
    )

    if args.checkpoint_step > -1:
        maml.load(args.checkpoint_step)
    else:
        print('Checkpoint loading skipped.')

    if not args.test:
        num_training_tasks = args.batch_size * (args.num_train_iterations -
                                                args.checkpoint_step - 1)
        print(
            f'Training on {num_training_tasks} tasks with composition: '
            f'num_way={args.num_way}, '
            f'num_support={args.num_support}, '
            f'num_query={args.num_query}'
        )
        # This is dataloader here. Samples bs = 4. 4 tasks. 

        dataloader_meta_train = DataGenerator(num_classes=args.num_way,
                                            num_samples_per_class=args.num_support + args.num_query,
                                            meta_train=True,
                                            class_names=None,
                                            bounding_box_mode = args.bounding_box_mode
                                            )
        
       

        maml.train(
            dataloader_meta_train=dataloader_meta_train,
            writer=writer
        )
    else:
        dataloader_meta_test =  DataGenerator(num_classes=args.num_way,
                                        num_samples_per_class=args.num_support + args.num_query,
                                        meta_train=False,
                                        class_names=None,
                                        bounding_box_mode = args.bounding_box_mode
                                        )
        


        print(
            f'Testing on tasks with composition '
            f'num_way={args.num_way}, '
            f'num_support={args.num_support}, '
            f'num_query={args.num_query}'
        )
        maml.test(dataloader_meta_test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Train a MAML!')
    parser.add_argument('--log_dir', type=str, default=None,
                        help='directory to save to or load from')
    
    parser.add_argument('--num_way', type=int, default=3,
                        help='number of classes in a task')
    
    parser.add_argument('--num_support', type=int, default=3,
                        help='number of support examples per class in a task')
    
    parser.add_argument('--num_query', type=int, default=1,
                        help='number of query examples per class in a task')
    parser.add_argument('--num_inner_steps', type=int, default=50,
                        help='number of inner-loop updates')
    parser.add_argument('--inner_lr', type=float, default=.0001,
                        help='inner-loop learning rate initialization')
    parser.add_argument('--learn_inner_lrs', default=False, action='store_true',
                        help='whether to optimize inner-loop learning rates')
    parser.add_argument('--outer_lr', type=float, default=0.0001,
                        help='outer-loop learning rate')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='number of tasks per outer-loop update')
    parser.add_argument('--num_train_iterations', type=int, default=100,
                        help='number of outer-loop updates to train for')
    parser.add_argument('--test', default=False, action='store_true',
                        help='train or test')
    parser.add_argument('--checkpoint_step', type=int, default=-1,
                        help=('checkpoint iteration to load for resuming '
                              'training, or for evaluation (-1 is ignored)'))
    parser.add_argument('--num_workers', type=int, default = 8, 
                        help=('needed to specify omniglot dataloader'))
    parser.add_argument('--device', type=str, default='gpu')
    parser.add_argument('--bounding_box_mode', type=str,
                        help=("specify the bounding box mode: 2D_to_2D or 2D_to_3D or 3D_to_3D"),
                        default='2D_to_3D')
    parser.add_argument('--depth_supervision', type=bool,
                        help=("Set to True to Concatenate Depth Map to Input to Transformer"),
                        default=False)
    args = parser.parse_args()
    main(args)