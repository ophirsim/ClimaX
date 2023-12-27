import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F


class JaccardLoss(nn.Module):
    """
    A JaccardLoss class to calculate the Jaccard loss, or IoU (Intersection over Union) loss, between two "images."
    In particular, this implementation seeks to calculate the loss in a "soft" way, where the confidence in the predictions plays a role in the loss

    This implementation allows for the introduction of a weighting which chooses how the Jaccard loss is weighed according to the losses of the individual classes in the label
    For context, a lot of this code was taken from https://github.com/hannahg141/ClimateNet/blob/main/climatenet/utils/losses.py

    (N, C, H, W) corresponds to (Number of samples, Class size, Height, Width)
    
    """
    def __init__(self):
        super(JaccardLoss, self).__init__()

    def get_cardinality(self, predictions, targets):
        """
        Return the cardinality of the predictions and targets tensors

        Args:
            predictions: Tensor of shape (N, C, H, W) containing the unnormalized logits outputted by the network
            targets: Tensor of shape (N, H, W) containing the ground-truth labels

        Returns:
            a tensor of size (N, C) containing the cardinalities of the predictions and targets
        """
        # generate one-hot encodings for the target labels
        targets_1_hot = torch.eye(predictions.shape[1])[targets]
        targets_1_hot = targets_1_hot.permute(0, 3, 1, 2).float()
        targets_1_hot = targets_1_hot.type(predictions.type())

        # return the cardinality
        return torch.sum(predictions + targets_1_hot, (2, 3))

    def get_intersection(self, predictions, targets):
        """
        Return the size of the intersection between the predictions and targets tensors

        Args:
            predictions: Tensor of shape (N, C, H, W) containing the unnormalized logits outputted by the network
            targets: Tensor of shape (N, H, W) containing the ground-truth labels

        Returns:
            a tensor of size (N, C) containing the intersection sizes of the predictions and targets
        """
        # generate one-hot encodings for the target labels
        targets_1_hot = torch.eye(predictions.shape[1])[targets]
        targets_1_hot = targets_1_hot.permute(0, 3, 1, 2).float()
        targets_1_hot = targets_1_hot.type(predictions.type())

        # return the intersection
        return torch.sum(predictions * targets_1_hot, (2, 3))

    def get_union(self, predictions, targets):
        """
        Return the size of the union between the predictions and targets tensors

        Args:
            predictions: Tensor of shape (N, C, H, W) containing the unnormalized logits outputted by the network
            targets: Tensor of shape (N, H, W) containing the ground-truth labels

        Returns:
            a tensor of size (N, C) containing the union sizes of the predictions and targets
        """
        # return the intersection
        return self.get_cardinality(predictions, targets) - self.get_intersection(predictions, targets)

    def forward(self, predictions, targets, weights=None, eps=1e-7) -> Tensor:
        """
        Calculates the cumulative weighted Jaccard loss of all predictions and their corresponding targets

        Args:
            predictions: Tensor of shape (N, C, H, W) containing the unnormalized logits outputted by the network
            targets: Tensor of shape (N, H, W) containing the ground-truth labels
            weights: Tensor of shape (C) containing the weights for each class that define each class's constribution to the loss
            eps: Int used in the Jaccard Index calculation to ensure numeric stability

        Returns:
            a single number wrapped in a tensor describing the loss of the predictions
        """

        # ensure that predictions and targets tensor shapes are appropriate
        assert predictions.dim() == 4, f"Predictions tensor expected to have shape (N, C, H, W)"
        assert targets.dim() == 3, f"Targets tensor expected to have shape (N, H, W)"
        assert (predictions.shape[0],) + predictions.shape[2:] == targets.shape, f"Prediction and Target tensor dimensions do not match up: {predictions.shape} and {targets.shape}"

        # grab the dimensions of the predictions and targets
        N, C, H, W = predictions.shape

        # set default weight if none is provided
        if weights is None:
            weights = torch.ones(C)
        
        # ensure that weight tensor shape is appropriate and sums to 1
        assert weights.shape == (C,), f"Weight shape is expected to be ({C},), but shape {weights.shape} was received"
        assert torch.sum(weights) == C, f"Weight tensor expected to sum to C, but summed to {torch.sum(weights)}"


        # perform softmax on the predictions
        predictions = F.softmax(predictions, dim=1)


        # compute intersection and union of the predictions with the targets
        intersection = self.get_intersection(predictions, targets)
        union = self.get_union(predictions, targets)


        # compute Jaccard Index
        JaccardIndices = (intersection / (union + eps)) # (N, C) IoU table
        wightedJaccardIndices = torch.matmul(JaccardIndices, weights) / C

        # compute Jaccard Loss
        JaccardLosses = 1 - wightedJaccardIndices
        loss = torch.sum(JaccardLosses)

        return loss
    

class DiceLoss(nn.Module):
    """
    A DiceLoss class to calculate the Dice loss, or Sørensen-Dice loss, between two "images."
    In particular, this implementation seeks to calculate the loss in a "soft" way, where the confidence in the predictions plays a role in the loss

    This implementation allows for the introduction of a weighting which chooses how the Dice loss is weighed according to the losses of the individual classes in the label
    For context, a lot of this code was taken from https://github.com/hannahg141/ClimateNet/blob/main/climatenet/utils/losses.py

    (N, C, H, W) corresponds to (Number of samples, Class size, Height, Width)
    
    """
    def __init__(self):
        super(DiceLoss, self).__init__()

    def get_cardinality(self, predictions, targets):
        """
        Return the cardinality of the predictions and targets tensors

        Args:
            predictions: Tensor of shape (N, C, H, W) containing the unnormalized logits outputted by the network
            targets: Tensor of shape (N, H, W) containing the ground-truth labels

        Returns:
            a tensor of size (N, C) containing the cardinalities of the predictions and targets
        """
        # generate one-hot encodings for the target labels
        targets_1_hot = torch.eye(predictions.shape[1])[targets]
        targets_1_hot = targets_1_hot.permute(0, 3, 1, 2).float()
        targets_1_hot = targets_1_hot.type(predictions.type())

        # return the cardinality
        return torch.sum(predictions + targets_1_hot, (2, 3))

    def get_intersection(self, predictions, targets):
        """
        Return the size of the intersection between the predictions and targets tensors

        Args:
            predictions: Tensor of shape (N, C, H, W) containing the unnormalized logits outputted by the network
            targets: Tensor of shape (N, H, W) containing the ground-truth labels

        Returns:
            a tensor of size (N, C) containing the intersection sizes of the predictions and targets
        """
        # generate one-hot encodings for the target labels
        targets_1_hot = torch.eye(predictions.shape[1])[targets]
        targets_1_hot = targets_1_hot.permute(0, 3, 1, 2).float()
        targets_1_hot = targets_1_hot.type(predictions.type())

        # return the intersection
        return torch.sum(predictions * targets_1_hot, (2, 3))

    def get_union(self, predictions, targets):
        """
        Return the size of the union between the predictions and targets tensors

        Args:
            predictions: Tensor of shape (N, C, H, W) containing the unnormalized logits outputted by the network
            targets: Tensor of shape (N, H, W) containing the ground-truth labels

        Returns:
            a tensor of size (N, C) containing the union sizes of the predictions and targets
        """
        # return the intersection
        return self.get_cardinality(predictions, targets) - self.get_intersection(predictions, targets)

    def forward(self, predictions, targets, weights=None, eps=1e-7) -> Tensor:
        """
        Calculates the cumulative weighted Dice loss of all predictions and their corresponding targets

        Args:
            predictions: Tensor of shape (N, C, H, W) containing the unnormalized logits outputted by the network
            targets: Tensor of shape (N, H, W) containing the ground-truth labels
            weights: Tensor of shape (C) containing the weights for each class that define each class's constribution to the loss
            eps: Int used in the Dice Index calculation to ensure numeric stability

        Returns:
            a single number wrapped in a tensor describing the loss of the predictions
        """

        # ensure that predictions and targets tensor shapes are appropriate
        assert predictions.dim() == 4, f"Predictions tensor expected to have shape (N, C, H, W)"
        assert targets.dim() == 3, f"Targets tensor expected to have shape (N, H, W)"
        assert (predictions.shape[0],) + predictions.shape[2:] == targets.shape, f"Prediction and Target tensor dimensions do not match up: {predictions.shape} and {targets.shape}"

        # grab the dimensions of the predictions and targets
        N, C, H, W = predictions.shape

        # set default weight if none is provided
        if weights is None:
            weights = torch.ones(C)
        
        # ensure that weight tensor shape is appropriate and sums to 1
        assert weights.shape == (C,), f"Weight shape is expected to be ({C},), but shape {weights.shape} was received"
        assert torch.sum(weights) == C, f"Weight tensor expected to sum to C, but summed to {torch.sum(weights)}"


        # perform softmax on the predictions
        predictions = F.softmax(predictions, dim=1)


        # compute cardinality and intersection of the predictions with the targets
        cardinality = self.get_cardinality(predictions, targets)
        intersection = self.get_intersection(predictions, targets)


        # compute Dice Index
        DiceIndices = ((2 * intersection) / (cardinality + eps)) # (N, C) Dice index table
        weightedDiceIndices = torch.matmul(DiceIndices, weights) / C

        # compute Dice Loss
        DiceLosses = 1 - weightedDiceIndices
        loss = torch.sum(DiceLosses)

        return loss