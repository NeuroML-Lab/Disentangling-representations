import torch
import torch.nn as nn


class SupMSELoss(nn.Module):
    def __init__(self):
        super(SupMSELoss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, features, labels):
        if labels is None:
            raise ValueError("SupMSELoss requires `labels`; got None.")

        device = features.device
        N = features.shape[0]

        # Upper triangle indices (excludes diagonal)
        i_idx, j_idx = torch.triu_indices(N, N, offset=1, device=device)

        # Filter to positive pairs only
        pos_mask = labels[i_idx] == labels[j_idx]
        i_pos, j_pos = i_idx[pos_mask], j_idx[pos_mask]

        if i_pos.numel() == 0:
            return torch.tensor(0.0, device=device, dtype=features.dtype)

        # nn.MSELoss over all positive pairs at once
        loss = self.mse_loss(features[i_pos], features[j_pos])
        return loss


class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07, contrast_mode='all', base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels):
        """Compute supervised contrastive loss.

        Args:
            features: Tensor of shape (N, D) where N is batch size and D is feature dim.
            labels:  Tensor of shape (N,) containing integer class labels.

        Returns:
            scalar loss (mean over anchors that have at least one positive example).
        """
        if labels is None:
            raise ValueError("SupConLoss requires `labels`; got None.")

        device = features.device
        batch_size = features.shape[0]

        # ensure labels are column vector on same device
        labels = labels.contiguous().view(-1, 1).to(device)

        # mask[i,j] = 1 if i and j have same label, 0 otherwise
        mask = torch.eq(labels, labels.T).float().to(device)

        # exclude self-comparisons from positives and denominators
        logits_mask = (~torch.eye(batch_size, dtype=torch.bool, device=device)).float()
        mask = mask * logits_mask

        # compute logits: dot product / temperature
        logits = torch.matmul(features, features.T) / self.temperature

        # numerical stability: subtract max per row
        logits = logits - logits.max(dim=1, keepdim=True)[0].detach() 

        # exponentiate only non-diagonal entries
        exp_logits = torch.exp(logits) * logits_mask

        # log-probabilities for each pair
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

        # mean log-prob of positive pairs for each anchor
        positive_sum = (mask * log_prob).sum(dim=1)
        positive_count = mask.sum(dim=1)

        # avoid division by zero; only keep anchors with at least one positive
        valid = positive_count > 0
        if valid.sum() == 0:
            # no positive pairs in batch; return zero loss
            return torch.tensor(0.0, device=device, dtype=features.dtype)

        mean_log_prob_pos = positive_sum[valid] / (positive_count[valid] + 1)

        loss = - mean_log_prob_pos.mean()
        return loss