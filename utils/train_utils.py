
from utils.loss_utils import *
from utils.metric_utils import *

import torch
import torch.nn.functional as F
from tqdm import tqdm
import logging
import wandb
import os

mse_loss = SupMSELoss()
contrastive_loss = SupConLoss()
accumulate_gradient= 4
iter_tracker = 0 
schedule = [1e-5]
patience = 60


def calculate_validation_loss(args, val_generator, model):
    val_loss = 0.0
    val_semantic_loss = 0.0
    val_syntactic_loss = 0.0
    samples = 0.0

    with torch.no_grad():
        for org, semantic, syntax, label in val_generator:
            all_inputs = syntax + semantic + org

            outputs = model(all_inputs)
            outputs_semantic, outputs_syntactic, outputs_residual = torch.split(outputs, [int(args.semantic_uints_percentage*model.hidden_dim), int(args.syntactic_uints_percentage*model.hidden_dim), int(args.residual_uints_percentage*model.hidden_dim)], dim=1)
            outputs_semantic = F.normalize(outputs_semantic, p=2, dim=1)
            outputs_syntactic = F.normalize(outputs_syntactic, p=2, dim=1)

            dummy_labels = torch.tensor([(-i-1) for i in range(label.shape[0])])
            semantic_labels = torch.concat((dummy_labels, label, label))
            syntactic_labels = torch.concat((label, dummy_labels, label))

            semantic_loss = contrastive_loss(outputs_semantic, semantic_labels)
            syntactic_loss = contrastive_loss(outputs_syntactic, syntactic_labels)
            loss = semantic_loss + syntactic_loss
            val_loss += loss.item()
            val_semantic_loss += semantic_loss.item()
            val_syntactic_loss += syntactic_loss.item()

            samples += outputs_syntactic.shape[0]
    return val_loss / samples, val_semantic_loss / samples, val_syntactic_loss / samples


def train_invertible_language_transforms(args, model, training_generator, val_generator, dirname):
    opt = torch.optim.Adam
    lr = schedule[0]
    print('Training with learning rate', lr)
    optimizer = opt(list(model.linear_invertible.parameters()), lr=lr)
    optimizer.zero_grad()
    iteration=0
    validation_loss = 999999999.99
    iter_tracker = 0

    for epoch in range(args.epochs):
        for org, semantic, syntax, label in tqdm(training_generator):
            
            all_inputs = syntax + semantic + org
            outputs = model(all_inputs)
            outputs_semantic, outputs_syntactic, outputs_residual = torch.split(outputs, [int(args.semantic_uints_percentage*model.hidden_dim), int(args.syntactic_uints_percentage*model.hidden_dim), int(args.residual_uints_percentage*model.hidden_dim)], dim=1)

            dummy_labels = torch.tensor([(-i-1) for i in range(label.shape[0])])
            semantic_labels = torch.concat((dummy_labels, label, label))
            syntactic_labels = torch.concat((label, dummy_labels, label))

            outputs_semantic = F.normalize(outputs_semantic, p=2, dim=1)
            outputs_syntactic = F.normalize(outputs_syntactic, p=2, dim=1)

            semantic_loss = contrastive_loss(outputs_semantic, semantic_labels) 
            syntactic_loss = contrastive_loss(outputs_syntactic, syntactic_labels) 
            loss = semantic_loss + syntactic_loss 
            loss.backward(retain_graph=True)

            iteration += 1
            # if iteration % accumulate_gradient == accumulate_gradient - 1:
            optimizer.step()
            optimizer.zero_grad()
            wandb.log({"loss": loss, "semantic_loss": semantic_loss, "syntactic_loss": syntactic_loss, "epoch": epoch, "iteration":iteration})
            if iteration % 500 == 0:
                val_loss, val_semantic_loss, val_syntactic_loss = calculate_validation_loss(args, val_generator, model)
                wandb.log({"val_loss": val_loss, "val_semantic_loss": val_semantic_loss, "val_syntactic_loss": val_syntactic_loss, "epoch": epoch, "iteration":iteration})
                logging.info(f"Epoch {epoch}, Iteration {iteration}, Loss: {loss.item()}, Semantic Loss: {semantic_loss.item()}, Syntactic Loss: {syntactic_loss.item()}, Val Loss: {val_loss}, Val Semantic Loss: {val_semantic_loss}, Val Syntactic Loss: {val_syntactic_loss}")
                if val_loss < validation_loss:
                    validation_loss = val_loss
                    torch.save(model.state_dict(), os.path.join(dirname, 'last_best_model.pt'))
                    logging.info(f"New best model saved")
                    iter_tracker = 0
                else:
                    iter_tracker += 1
                    if iter_tracker == patience:
                        logging.info(f"Early stopping triggered after {patience} iterations without improvement.")
                        print(1/0)  # Raise an error to stop training
            torch.cuda.empty_cache()


def calculate_visual_validation_loss(args, val_generator, model):
    val_loss = 0.0
    val_semantic_loss = 0.0
    val_syntactic_loss = 0.0
    samples = 0.0

    with torch.no_grad():
        for X, content_label, style_label in val_generator:
            X = X.view(-1, *X.shape[2:]).cuda()
            content_label = content_label.view(-1).cuda()
            style_label = style_label.view(-1).cuda()

            outputs = model(X)
            outputs_content, outputs_style, outputs_residual = torch.split(outputs, [int(args.semantic_uints_percentage*model.hidden_dim), int(args.syntactic_uints_percentage*model.hidden_dim), int(args.residual_uints_percentage*model.hidden_dim)], dim=1)
            outputs_content = F.normalize(outputs_content, p=2, dim=1)
            outputs_style = F.normalize(outputs_style, p=2, dim=1)
            semantic_loss = contrastive_loss(outputs_content, content_label) 
            syntactic_loss = contrastive_loss(outputs_style, style_label)
            loss = semantic_loss + syntactic_loss
            val_loss += loss.item()
            val_semantic_loss += semantic_loss.item()
            val_syntactic_loss += syntactic_loss.item()

            samples += outputs.shape[0]
    return val_loss / samples, val_semantic_loss / samples, val_syntactic_loss / samples


def train_invertible_visual_transforms(args, model, training_generator, val_generator, dirname):
    opt = torch.optim.Adam
    lr = schedule[0]
    print('Training with learning rate', lr)
    optimizer = opt(list(model.linear_invertible.parameters()), lr=lr)
    optimizer.zero_grad()
    iteration=0
    validation_loss = 999999999.99
    iter_tracker = 0

    for epoch in range(args.epochs):
        for X, content_label, style_label in tqdm(training_generator):
            X = X.view(-1, *X.shape[2:]).cuda()
            content_label = content_label.view(-1).cuda()
            style_label = style_label.view(-1).cuda()
            outputs = model(X)

            if args.residual_uints_percentage > 0:
                outputs_content, outputs_style, outputs_residual = torch.split(outputs, [int(args.semantic_uints_percentage*model.hidden_dim), int(args.syntactic_uints_percentage*model.hidden_dim), int(args.residual_uints_percentage*model.hidden_dim)], dim=1)
            else:
                outputs_content, outputs_style = torch.split(outputs, [int(args.semantic_uints_percentage*model.hidden_dim), int(args.syntactic_uints_percentage*model.hidden_dim)], dim=1)
            outputs_content = F.normalize(outputs_content, p=2, dim=1)
            outputs_style = F.normalize(outputs_style, p=2, dim=1)
            semantic_loss = contrastive_loss(outputs_content, content_label)
            syntactic_loss = contrastive_loss(outputs_style, style_label) 
            loss = semantic_loss + syntactic_loss 
            loss.backward(retain_graph=True)
            
            iteration += 1
            # if iteration % accumulate_gradient == accumulate_gradient - 1:
            optimizer.step()
            optimizer.zero_grad()
            wandb.log({"loss": loss, "semantic_loss": semantic_loss, "syntactic_loss": syntactic_loss, "epoch": epoch, "iteration":iteration})
        val_loss, val_semantic_loss, val_syntactic_loss = calculate_visual_validation_loss(args, val_generator, model)
        wandb.log({"val_loss": val_loss, "val_semantic_loss": val_semantic_loss, "val_syntactic_loss": val_syntactic_loss, "epoch": epoch, "iteration":iteration})
        logging.info(f"Epoch {epoch}, Iteration {iteration}, Loss: {loss.item()}, Semantic Loss: {semantic_loss.item()}, Syntactic Loss: {syntactic_loss.item()}, Val Loss: {val_loss}, Val Semantic Loss: {val_semantic_loss}, Val Syntactic Loss: {val_syntactic_loss}")
        if val_loss < validation_loss:
            validation_loss = val_loss
            torch.save(model.state_dict(), os.path.join(dirname, 'last_best_model.pt'))
            logging.info(f"New best model saved")
            iter_tracker = 0
        else:
            iter_tracker += 1
            if iter_tracker == patience:
                logging.info(f"Early stopping triggered after {patience} iterations without improvement.")
                print(1/0)  # Raise an error to stop training
            torch.cuda.empty_cache()