import torch.nn.functional as F
import torch

def syntax_sensitivity(syntax_outputs, original_outputs, semantic_outputs):
    corr_syntax = F.cosine_similarity(syntax_outputs - syntax_outputs.mean(1, keepdim=True), original_outputs - original_outputs.mean(1, keepdim=True), dim=1)
    corr_semantic = F.cosine_similarity(semantic_outputs - semantic_outputs.mean(1, keepdim=True), original_outputs - original_outputs.mean(1, keepdim=True), dim=1)
    return torch.sum(corr_syntax - corr_semantic)



def semantic_sensitivity(syntax_outputs, original_outputs, semantic_outputs):
    corr_syntax = F.cosine_similarity(syntax_outputs - syntax_outputs.mean(1, keepdim=True), original_outputs - original_outputs.mean(1, keepdim=True), dim=1)
    corr_semantic = F.cosine_similarity(semantic_outputs - semantic_outputs.mean(1, keepdim=True), original_outputs - original_outputs.mean(1, keepdim=True), dim=1)
    return torch.sum(corr_semantic - corr_syntax)