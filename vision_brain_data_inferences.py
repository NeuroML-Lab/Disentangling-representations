from utils.inference_dataloaders import NSD_Dataset_visual
from utils.models_util import *
from scipy.stats import pearsonr
from sklearn.linear_model import RidgeCV
import pickle
import torch
import argparse
from tqdm import tqdm
import numpy as np
import logging
import os
import shutil

params = {'batch_size': 8,
          'shuffle': True,
          'num_workers': 0}

params_val = {'batch_size': 8,
          'shuffle': True,
          'num_workers': 0}

schedule = [1e-4]
accumulate_gradient= 4
patience = 20

def compute_scores(y, y_hat): ##true,pred
    corrs = []
    for i in range(y.shape[1]):
        val_idx = y[:,i] > -999.0
        corrs.append(pearsonr(y[val_idx, i], y_hat[val_idx, i])[0])
    return corrs

def masked_Correlation_loss(output, target):
    mask = target > -900.0
    output, target = output[mask], target[mask]

    mean_output = torch.mean(output)
    mean_target = torch.mean(target)

    covariance = torch.mean((output - mean_output) * (target - mean_target))
    var_output = torch.var(output)
    var_target = torch.var(target)

    numerator = 2 * covariance
    denominator = var_output + var_target + (mean_output - mean_target)**2
    ccc = numerator / denominator

    loss = 1 - ccc
    return loss

def masked_MSEloss(output, target):
    vec = (output - target)**2
    mask = target > -900.0
    loss = torch.sum(vec[mask])/torch.sum(mask) ##; pudb.set_trace()
    return loss

criterion_mse = masked_MSEloss
criterion_corr = masked_Correlation_loss

def compute_predictions_spatial(loader, model, reshape=True, stack=True, return_lag=False):
    y, y_hat = [], []
    for id, x_val, y_val in (loader):
        x_val = x_val.squeeze()
        y_mod = model(x_val.cuda().float()).data.cpu().numpy()
        y.append(y_val.numpy())
        y_hat.append(y_mod)
    if stack:
        y, y_hat = np.vstack(y), np.vstack(y_hat) 
    return y, y_hat ##true,pred

def save_checkpoint(state, is_best, checkpoint, model_str):
       """Saves model and training parameters at checkpoint + 'last.pth.tar'. If is_best==True, also saves
       checkpoint + 'best.pth.tar'
       Args:
           state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
           is_best: (bool) True if it is the best model seen till now
           checkpoint: (string) folder where parameters are to be saved
       """
       existing_files = os.listdir(checkpoint)
       for f in existing_files:
        if f.startswith('last'):
            os.remove(os.path.join(checkpoint, f))
       filepath = os.path.join(checkpoint, 'last_%s.pth.tar'% model_str)
       if not os.path.exists(checkpoint):
        #    print("Checkpoint Directory does not exist! Making directory {}".format(checkpoint))
           os.mkdir(checkpoint)
    #    else:
    #        print("Checkpoint Directory exists! ")
       torch.save(state, filepath)
       if is_best:
           shutil.copyfile(filepath, os.path.join(checkpoint, 'best.pth.tar'))

def define_args():
    parser = argparse.ArgumentParser(description='Encoding model')
    parser.add_argument('--brain_region', default='ventral_visual_data', type=str)
    parser.add_argument('--pretrained_model_name', default='swinb', type=str)
    parser.add_argument('--pretrained_transformation', default='invertible', type=str)
    parser.add_argument('--unit_ratio', default='0.4_0.4', type=str)
    parser.add_argument('--units_type', default='content', type=str)
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--splits', default='splits_2', type=str)
    args = parser.parse_args()
    return args


def train(model_dir, x_train_norm, y_train_norm):
    ridge_cv = RidgeCV(alphas=[10 ** x for x in range(-30, 30)])
    ridge_cv.fit(x_train_norm, y_train_norm)

    with open(model_dir + "/ridgecv_model.pkl", "wb") as f:
        pickle.dump(ridge_cv, f)
    print("training done")

def evaluate(model_dir, x_test_norm, y_test_norm, noise_ceiling):
    with open(model_dir + "/ridgecv_model.pkl", "rb") as f:
        ridge_cv = pickle.load(f)
    y_pred = ridge_cv.predict(x_test_norm)
    test_corr = compute_scores(y_test_norm, y_pred)
    test_corr_norm = test_corr / noise_ceiling
    print(np.nanmean(test_corr_norm))
    np.save(model_dir + "/test_corr_norm.npy", test_corr_norm)

if __name__ == "__main__":
    args = define_args()

    brain_region = args.brain_region
    training_set = NSD_Dataset_visual(mode = 'train', brain_region = brain_region, data_path='/home/khoslalab/Documents/response-optimised-models/data/'+brain_region+'/', splits=args.splits, model_name=args.pretrained_model_name, transformation_type=args.pretrained_transformation, unit_ratio=args.unit_ratio, units_type=args.units_type)
    training_generator = torch.utils.data.DataLoader(training_set,  **params)
    print('Train data loaded')

    test_set = NSD_Dataset_visual(mode = 'test', brain_region = brain_region, data_path='/home/khoslalab/Documents/response-optimised-models/data/'+brain_region+'/', splits=args.splits, model_name=args.pretrained_model_name, transformation_type=args.pretrained_transformation, unit_ratio=args.unit_ratio, units_type=args.units_type)
    test_generator = torch.utils.data.DataLoader(test_set,  **params_val)
    print('test data loaded')

    
    model_dir = 'inference_outputs/NSD/' + brain_region + '/' + args.pretrained_model_name + '_' + args.unit_ratio + '_' + args.pretrained_transformation + '/' + args.units_type + '/' + args.splits + '/'
    os.makedirs(model_dir, exist_ok=True)

    logging.basicConfig(filename=model_dir + '/logs.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("start training with %d %d", 0, 0)

    noise_ceiling = np.load('/home/khoslalab/Documents/response-optimised-models/data/'+args.brain_region+'/noise_ceiling_1257_filtered.npy')

    x_train, y_train = [], []
    x_test, y_test = [], []


    for img_ids, x_batch, y_batch in tqdm(training_generator):
        x_batch = x_batch.squeeze()
        x_train.append(x_batch)
        y_train.append(y_batch)

    for img_ids, x_batch, y_batch in tqdm(test_generator):
        x_batch = x_batch.squeeze()
        x_test.append(x_batch)
        y_test.append(y_batch)

    x_train, y_train = np.vstack(x_train), np.vstack(y_train)
    x_test, y_test = np.vstack(x_test), np.vstack(y_test)

    x_train, x_test = x_train.reshape(x_train.shape[0],-1), x_test.reshape(x_test.shape[0],-1)

    train_mean = np.mean(y_train, axis=0)
    train_std  = np.std(y_train, axis=0)
    y_train_norm = (y_train - train_mean) / train_std
    y_test_norm  = (y_test - train_mean) / train_std

    train_mean = np.mean(x_train, axis=0)
    train_std  = np.std(x_train, axis=0)
    x_train_norm = (x_train - train_mean) / train_std
    x_test_norm  = (x_test - train_mean) / train_std
    
    if args.evaluate:
        evaluate(model_dir, x_test_norm, y_test_norm, noise_ceiling)
    else:
        train(model_dir, x_train_norm, y_train_norm)
            