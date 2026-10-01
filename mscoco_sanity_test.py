from utils.dataloader import *
from utils.models_util import *
from utils.train_utils import *
import torch.nn.functional as F
from itertools import islice
from tqdm import tqdm
import argparse
import logging
import random
import os

def parse_args():
    parser = argparse.ArgumentParser(description='Run MSCOCO sanity tests for a given model.')
    parser.add_argument('--model_name', type=str, required=True,
                        help='Name of the model to use (e.g. swinb)')
    parser.add_argument('--transformation_type', type=str, required=True,
                        help='Name of the transformation type to use (e.g. swinb)')
    return parser.parse_args()


def main():
    args = parse_args()
    model_name = args.model_name

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    os.makedirs('inference_outputs/mscoco_sanity_tests/' + args.transformation_type + '/' + model_name + '/', exist_ok=True)

    if args.transformation_type == "permutation":
        print("permutation")
        model = Disentangled_Permutation_Visual_Model(model_name, args.transformation_type, None).cuda()
        model.load_state_dict(
            torch.load(
                f"vision_outputs/{args.transformation_type}/mscoco_stylized_shape_texture_{model_name}_0.01_None_avg_{args.transformation_type}_0.4375_0.4375/last_best_model.pt"
            )
        )
    else:
        model = Disentangled_Invertible_Visual_Model(model_name, args.transformation_type, None).cuda()
        model.load_state_dict(
            torch.load(
                f"vision_outputs/{args.transformation_type}/mscoco_stylized_shape_texture_{model_name}_0.01_None_avg_{args.transformation_type}_0.4375_0.4375/last_best_model.pt"
            )
        )

    content_splits = np.load('raw_training_data/mscoco_stylised/splits/content_test.npy', allow_pickle=True)
    style_splits = np.load('raw_training_data/mscoco_stylised/splits/style_test.npy', allow_pickle=True)
    image_path = 'raw_training_data/mscoco_stylised/images/'

    # --- Correlations over content splits ---
    corr_content_c = []
    corr_style_c = []
    for c in tqdm(content_splits, desc='Content splits'):
        X = []
        for s in style_splits:
            img_name = c[:-4] + '_' + s
            X.append(preprocess(Image.open(os.path.join(image_path, img_name))))
        X = torch.stack(X, dim=0)
        with torch.no_grad():
            if args.transformation_type == "permutation":
                outputs = model.inference_forward(X.cuda())
            else:
                outputs = model(X.cuda())
        outputs_content, outputs_style, outputs_residual = torch.split(
            outputs,
            [int(0.4375 * model.hidden_dim), int(0.4375 * model.hidden_dim), int(0.125 * model.hidden_dim)],
            dim=1
        )
        corr_content_c.append(torch.corrcoef(outputs_content).cpu().detach().numpy().mean())
        corr_style_c.append(torch.corrcoef(outputs_style).cpu().detach().numpy().mean())

    # --- Correlations over style splits ---
    corr_content_s = []
    corr_style_s = []
    for s in tqdm(style_splits, desc='Style splits'):
        X = []
        for c in content_splits:
            img_name = c[:-4] + '_' + s
            X.append(preprocess(Image.open(os.path.join(image_path, img_name))))
        X = torch.stack(X, dim=0)
        with torch.no_grad():
            if args.transformation_type == "permutation":
                outputs = model.inference_forward(X.cuda())
            else:
                outputs = model(X.cuda())
        outputs_content, outputs_style, outputs_residual = torch.split(
            outputs,
            [int(0.4375 * model.hidden_dim), int(0.4375 * model.hidden_dim), int(0.125 * model.hidden_dim)],
            dim=1
        )
        corr_content_s.append(torch.corrcoef(outputs_content).cpu().detach().numpy().mean())
        corr_style_s.append(torch.corrcoef(outputs_style).cpu().detach().numpy().mean())

    # --- Cast to float and save ---
    corr_content_c = [float(c) for c in corr_content_c]
    corr_style_c   = [float(c) for c in corr_style_c]
    corr_content_s = [float(c) for c in corr_content_s]
    corr_style_s   = [float(c) for c in corr_style_s]

    out_dir = 'inference_outputs/mscoco_sanity_tests/' + args.transformation_type + '/' + model_name + '/'
    np.save(out_dir + 'corr_content_c.npy', corr_content_c)
    np.save(out_dir + 'corr_style_c.npy',   corr_style_c)
    np.save(out_dir + 'corr_content_s.npy', corr_content_s)
    np.save(out_dir + 'corr_style_s.npy',   corr_style_s)

    print(f'Results saved to {out_dir}')


if __name__ == '__main__':
    main()