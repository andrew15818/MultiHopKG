import torch


class ContrastiveSoftReward(LFramework):
    """
    The initial formulation of contrastive learning is as follows:
        - Take the entity and predicted action (e1, r) as a "positive sample",
          and "negative pairs" (e1, r_n) where r_n is *not* in e1's action space.
        - Try to make the positive pair similarity greater than the negative triplets (e1, r_n, pred_e2) for n in N
    """

    def __init__(self):
        pass

    def forward_fact(self, examples):
        pass
