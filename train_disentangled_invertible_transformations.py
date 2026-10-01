from utils.dataloader import *
from utils.models_util import *
from utils.train_utils import *

import torch.nn.functional as F
from itertools import islice
from tqdm import tqdm
import argparse
import logging

import wandb
wandb.init(project="my-project")

params = {'batch_size': 8,
          'shuffle': True,
          'num_workers': 0}

params_val = {'batch_size': 8,
          'shuffle': True,
          'num_workers': 0}



def define_args():
    parser = argparse.ArgumentParser(description='Encoding model')
    parser.add_argument('--transformation_type', default='invertible', type=str)
    parser.add_argument('--model_name', default='dino_vit_b', type=str)
    parser.add_argument('--data_type', default='vision', type=str)
    parser.add_argument('--hidden_state_level', default=None, type=int)
    parser.add_argument('--training_datasets',nargs='+',type=str,default=['mnist_stylised_shape_texture'])
    parser.add_argument('--test_datasets',nargs='+',type=str,default=['mscoco_stylized_shape_texture'])
    parser.add_argument('--token_processing_type', default='avg', type=str)
    parser.add_argument('--functional_dissociations', default='content_style', type=str)
    parser.add_argument('--syntactic_uints_percentage', default=0.4375, type=float)
    parser.add_argument('--semantic_uints_percentage', default=0.4375, type=float)
    parser.add_argument('--residual_uints_percentage', default=0.125, type=float)
    parser.add_argument('--alpha', default=0.01, type=float)
    parser.add_argument('--epochs', default=1000, type=int)
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--finetune', action='store_true')
    parser.add_argument('--evaluation_dir', default=None,type=str)
    args = parser.parse_args()
    return args



if __name__ == "__main__":
    args = define_args()
    args.residual_uints_percentage = 1.0 - args.semantic_uints_percentage - args.syntactic_uints_percentage
    

    dataset_name = '_'.join(args.training_datasets)
    dirname = args.data_type + '_outputs/' + args.transformation_type + '/' + dataset_name + '_' + args.functional_dissociations + '/' + args.model_name + '/' + str(args.hidden_state_level) + '_' + args.token_processing_type + '_' + str(args.semantic_uints_percentage) + '_' + str(args.syntactic_uints_percentage)
    print(dirname)
    if os.path.isdir(dirname):
        print('Warning: Directory already exists. Results will be overwritten.')
        # print(4/0)
    os.makedirs(dirname, exist_ok=True)
    
    logging.basicConfig(filename=dirname + '/validation.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("start training")
    print(f"Residual units percentage: {args.residual_uints_percentage}")

    if args.data_type == 'language':
        training_set = Dataset_language(mode = 'train', datasets=args.training_datasets)
        training_generator = torch.utils.data.DataLoader(training_set,  **params)
        val_set = Dataset_language(mode = 'val')
        val_generator = torch.utils.data.DataLoader(val_set,  **params_val)
        print('Data loaded')
        if args.transformation_type == 'permutation':
            print('Using Disentangled_Permutation_Language_Model')  
            model = Disentangled_Permutation_Language_Model(args.model_name, args.hidden_state_level, args.token_processing_type, args.alpha, args.transformation_type).cuda()
        else:
            print('Using Disentangled_Invertible_Language_Model')
            model = Disentangled_Invertible_Language_Model(args.model_name, args.hidden_state_level, args.token_processing_type, args.alpha, args.transformation_type).cuda()
        if args.finetune:
            model.load_state_dict(torch.load(dirname + '/last_best_model.pt'))
            logging.info("finetuned")
        train_invertible_language_transforms(args, model, training_generator, val_generator, dirname)
    elif args.data_type == 'vision':
        training_set = Dataset_vision(mode = 'train', datasets=args.training_datasets, functional_dissociations=args.functional_dissociations)
        training_generator = torch.utils.data.DataLoader(training_set,  **params)
        val_set = Dataset_vision(mode = 'val', datasets=args.training_datasets, functional_dissociations=args.functional_dissociations)
        val_generator = torch.utils.data.DataLoader(val_set,  **params_val)
        print('Data loaded')
        print('Transformation type:', args.transformation_type)
        if args.transformation_type == 'permutation':
            print('Using Disentangled_Permutation_Visual_Model')  
            model = Disentangled_Permutation_Visual_Model(args.model_name, args.transformation_type, args.hidden_state_level).cuda()
        else:
            print('Using Disentangled_Invertible_Visual_Model')
            model = Disentangled_Invertible_Visual_Model(args.model_name, args.transformation_type, args.hidden_state_level).cuda()
        train_invertible_visual_transforms(args, model, training_generator, val_generator, dirname)