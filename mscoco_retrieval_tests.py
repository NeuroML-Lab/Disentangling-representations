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
    parser = argparse.ArgumentParser(description="MSCOCO retrieval evaluation")
    parser.add_argument(
        "--model_name",
        type=str,
        default="swinb",
        help="Model name, e.g. swinb, resnet50, vit_b …",
    )
    parser.add_argument("--transformation_type", type=str, default="invertible", help="Transformation type")
    parser.add_argument('--functional_dissociations', default='content_class', type=str)
    parser.add_argument('--syntactic_uints_percentage', default='0.4375', type=str)
    parser.add_argument('--semantic_uints_percentage', default='0.4375', type=str)
    return parser.parse_args()


def get_nearest_10_neighbors(X, k=100):
    N = X.shape[0]
    neighbors = np.zeros((N, k), dtype=np.int32)
    X = X.astype(np.float32)
    for i in tqdm(range(N)):
        diff = X[i] - X
        mse = np.mean(diff * diff, axis=1)
        mse[i] = np.inf
        neighbors[i] = np.argpartition(mse, k)[:k]
        neighbors[i] = neighbors[i][np.argsort(mse[neighbors[i]])]
    return neighbors


def get_topk_accuracy(neighbors, k, content_labels, style_labels):
    top_k_content = 0.0
    top_k_style   = 0.0
    pred_content   = content_labels[neighbors]
    pred_style     = style_labels[neighbors]
    for i in tqdm(range(len(pred_content))):
        if content_labels[i] in pred_content[i][:k]:
            top_k_content += 1.0
        if style_labels[i] in pred_style[i][: k]:
            top_k_style += 1.0
    return top_k_content / 10000, top_k_style / 10000


def main():
    args = parse_args()
    model_name = args.model_name
    args.residual_uints_percentage = 1.0 - float(args.semantic_uints_percentage) - float(args.syntactic_uints_percentage)

    out_dir = f"inference_outputs/mnist_{args.functional_dissociations}_retrieval_results/{args.transformation_type}/{model_name}/{args.semantic_uints_percentage}_{args.syntactic_uints_percentage}"
    if os.path.isdir(out_dir):
        print('Warning: Directory already exists. Results will be overwritten.')
        # print(4/0)

    # ── Preprocessing ────────────────────────────────────────────────────────
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # ── Model ────────────────────────────────────────────────────────────────
    if args.transformation_type == "permutation":
        model = Disentangled_Permutation_Visual_Model(model_name, args.transformation_type, None).cuda()
        model.load_state_dict(
            torch.load(
                f"vision_outputs/{args.transformation_type}/mnist_stylised_shape_texture_{args.functional_dissociations}/{model_name}/None_avg_{args.semantic_uints_percentage}_{args.syntactic_uints_percentage}/last_best_model.pt"
            )
        )
    else:
        model = Disentangled_Invertible_Visual_Model(model_name, args.transformation_type, None).cuda()
        model.load_state_dict(
            torch.load(
                f"vision_outputs/{args.transformation_type}/mnist_stylised_shape_texture_{args.functional_dissociations}/{model_name}/None_avg_{args.semantic_uints_percentage}_{args.syntactic_uints_percentage}/last_best_model.pt"
            )
        )

    # ── Dataloader ───────────────────────────────────────────────────────────
    params = {"batch_size": 1, "shuffle": True, "num_workers": 0}
    all_set = Stylised_COCO_Conflicting_Cues()
    all_generator = torch.utils.data.DataLoader(all_set, **params)

    # ── Feature extraction ───────────────────────────────────────────────────
    content_reps, content_labels = [], []
    style_reps, style_labels = [], []
    res_reps, images = [], []
    org_reps = []

    for X, content_label, style_label in tqdm(all_generator):
        # print(content_label, style_label)
        with torch.no_grad():
            images.append(X.numpy())
            # output = get_vision_model_embeddings(model.model, model.model_name, images).reshape(images.shape[0],-1)
            if args.transformation_type == "permutation":
                output = model.inference_forward(X.cuda())
            else:
                output = model(X.cuda())
            content_rep, style_rep, res_rep = torch.split(
                output,
                [
                    int(float(args.semantic_uints_percentage) * model.hidden_dim),
                    int(float(args.syntactic_uints_percentage) * model.hidden_dim),
                    int(float(args.residual_uints_percentage) * model.hidden_dim),
                ],
                dim=1,
            )
            content_reps.append(content_rep.cpu().detach().numpy())
            style_reps.append(style_rep.cpu().detach().numpy())
            res_reps.append(res_rep.cpu().detach().numpy())
            org_rep = output
            org_reps.append(org_rep.cpu().detach().numpy())
            content_labels.append(list(content_label))
            style_labels.append(style_label)

    content_reps   = np.squeeze(np.array(content_reps))
    style_reps     = np.squeeze(np.array(style_reps))
    res_reps       = np.squeeze(np.array(res_reps))
    org_reps = np.squeeze(np.array(org_reps))
    content_labels = np.squeeze(np.array(content_labels))
    style_labels   = np.squeeze(np.array(style_labels))

    # ── Nearest-neighbour search ──────────────────────────────────────────────
    style_nearest_neighbors   = get_nearest_10_neighbors(style_reps)
    content_nearest_neighbors = get_nearest_10_neighbors(content_reps)
    res_nearest_neighbors     = get_nearest_10_neighbors(res_reps)
    org_nearest_neighbors     = get_nearest_10_neighbors(org_reps)

    # ── Accuracy computation ──────────────────────────────────────────────────
    results = {
        "style": {
            10: get_topk_accuracy(style_nearest_neighbors,   10, content_labels, style_labels),
            5:  get_topk_accuracy(style_nearest_neighbors,   5,  content_labels, style_labels),
            1:  get_topk_accuracy(style_nearest_neighbors,   1,  content_labels, style_labels),
        },
        "content": {
            10: get_topk_accuracy(content_nearest_neighbors, 10, content_labels, style_labels),
            5:  get_topk_accuracy(content_nearest_neighbors, 5,  content_labels, style_labels),
            1:  get_topk_accuracy(content_nearest_neighbors, 1,  content_labels, style_labels),
        },
        "res": {
            10: get_topk_accuracy(res_nearest_neighbors,     10, content_labels, style_labels),
            5:  get_topk_accuracy(res_nearest_neighbors,     5,  content_labels, style_labels),
            1:  get_topk_accuracy(res_nearest_neighbors,     1,  content_labels, style_labels),
        },
        # "org": {
        #     10: get_topk_accuracy(org_nearest_neighbors,     10, content_labels, style_labels),
        #     5:  get_topk_accuracy(org_nearest_neighbors,     5,  content_labels, style_labels),
        #     1:  get_topk_accuracy(org_nearest_neighbors,     1,  content_labels, style_labels),
        # },
    }


    # ── Save results ──────────────────────────────────────────────────────────
    
    os.makedirs(out_dir, exist_ok=True)

    for unit_type, k_dict in results.items():
        for k, (content_acc, style_acc) in k_dict.items():
            np.save(os.path.join(out_dir, f"{unit_type}_units_content_acc_{k}.npy"), content_acc)
            np.save(os.path.join(out_dir, f"{unit_type}_units_style_acc_{k}.npy"),   style_acc)

    print(f"\nDone. Results saved to {out_dir}")


if __name__ == "__main__":
    main()