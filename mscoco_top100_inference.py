from utils.dataloader import *
from utils.models_util import *
from utils.train_utils import *
from utils.inference_dataloaders import *
import torch.nn.functional as F
from itertools import islice
from tqdm import tqdm
import argparse
import logging
import random
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Run top-100 retrieval evaluation.")
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Model name (e.g. 'swinb')"
    )
    parser.add_argument('--transformation_type', type=str, required=True,
                        help='Name of the transformation type to use (e.g. invertible)')
    return parser.parse_args()


def get_nearest_100_neighbors(X, k=99):
    N = X.shape[0]
    neighbors = np.zeros((N, k), dtype=np.int32)
    X = X.astype(np.float32)
    for i in tqdm(range(N)):
        diff = X[i] - X              # (N, D)
        mse = np.mean(diff * diff, axis=1)  # (N,)
        mse[i] = np.inf              # exclude self
        neighbors[i] = np.argpartition(mse, k)[:k]
        neighbors[i] = neighbors[i][np.argsort(mse[neighbors[i]])]
    return neighbors


def calculate_top100_acc(neighbors, content_labels, style_labels):
    pred_content = content_labels[neighbors]
    pred_style = style_labels[neighbors]
    content_acc = 0.0
    style_acc = 0.0

    for i in tqdm(range(len(pred_content))):
        content_acc += float(np.sum(pred_content[i] == str(content_labels[i]))) / 99
        style_acc += float(np.sum(pred_style[i] == str(style_labels[i]))) / 99

    return content_acc / 10000, style_acc / 10000


def main():
    args = parse_args()
    model_name = args.model_name

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load model
    if args.transformation_type == "permutation":
        print("permutation")
        model = Disentangled_Permutation_Visual_Model(model_name, args.transformation_type, None).cuda()
        model.load_state_dict(
            torch.load(
                f"vision_outputs/{args.transformation_type}/shuffled_mscoco_stylized_shape_texture/{model_name}/None_avg_0.4375_0.4375/last_best_model.pt"
            )
        )
    else:
        model = Disentangled_Invertible_Visual_Model(model_name, args.transformation_type, None).cuda()
        model.load_state_dict(
            torch.load(
                f"vision_outputs/{args.transformation_type}/shuffled_mscoco_stylized_shape_texture/{model_name}/None_avg_0.4375_0.4375/last_best_model.pt"
            )
        )

    # DataLoader
    params = {'batch_size': 1, 'shuffle': True, 'num_workers': 0}
    all_set = Stylised_COCO_Conflicting_Cues()
    all_generator = torch.utils.data.DataLoader(all_set, **params)

    # Extract representations
    content_reps = []
    content_labels = []
    style_reps = []
    org_reps = []
    style_labels = []
    res_reps = []
    images = []

    for X, content_label, style_label in tqdm(all_generator):
        with torch.no_grad():
            images.append(X.numpy())
            if args.transformation_type == "permutation":
                output = model.inference_forward(X.cuda())
            else:
                output = model(X.cuda())
            content_rep, style_rep, res_rep = torch.split(
                output,
                [
                    int(0.4375 * model.hidden_dim),
                    int(0.4375 * model.hidden_dim),
                    int(0.125 * model.hidden_dim)
                ],
                dim=1
            )
            # org_rep = model.reconstruct(X.cuda())
            content_reps.append(content_rep.cpu().detach().numpy())
            style_reps.append(style_rep.cpu().detach().numpy())
            res_reps.append(res_rep.cpu().detach().numpy())
            content_labels.append(list(content_label))
            style_labels.append(style_label)
            # org_reps.append(org_rep.cpu().detach().numpy())

    content_reps = np.squeeze(np.array(content_reps))
    style_reps = np.squeeze(np.array(style_reps))
    res_reps = np.squeeze(np.array(res_reps))
    # org_reps = np.squeeze(np.array(org_reps))
    content_labels = np.squeeze(np.array(content_labels))
    style_labels = np.squeeze(np.array(style_labels))

    # Compute nearest neighbors
    style_nearest_neighbors = get_nearest_100_neighbors(style_reps)
    content_nearest_neighbors = get_nearest_100_neighbors(content_reps)
    # org_nearest_neighbors = get_nearest_100_neighbors(org_reps)

    # Compute accuracies
    style_units_content_acc, style_units_style_acc = calculate_top100_acc(
        style_nearest_neighbors, content_labels, style_labels
    )
    content_units_content_acc, content_units_style_acc = calculate_top100_acc(
        content_nearest_neighbors, content_labels, style_labels
    )

    # org_units_content_acc, org_units_style_acc = calculate_top100_acc(
    #     org_nearest_neighbors, content_labels, style_labels
    # )

    Content_units = [content_units_style_acc, content_units_content_acc]
    Style_units = [style_units_style_acc, style_units_content_acc]
    # org_units = [org_units_style_acc, org_units_content_acc]

    # Save results
    out_dir = f"inference_outputs/shuffled_mscoco_top100_retrieval_results/{args.transformation_type}/{model_name}/"
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "Content_units.npy"), Content_units)
    np.save(os.path.join(out_dir, "Style_units.npy"), Style_units)
    # np.save(os.path.join(out_dir, "Org_units.npy"), org_units)

    print(f"Results saved to {out_dir}")
    print(f"Content units — style acc: {content_units_style_acc:.4f}, content acc: {content_units_content_acc:.4f}")
    print(f"Style units   — style acc: {style_units_style_acc:.4f},   content acc: {style_units_content_acc:.4f}")


if __name__ == "__main__":
    main()