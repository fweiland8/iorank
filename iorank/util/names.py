from torch.optim import Adam, SGD

from torchcsrank.losses import hinged_rank_loss

losses = {
    "hinged_rank_loss": hinged_rank_loss
}

optimizers = {
    "Adam": Adam,
    "SGD": SGD
}
