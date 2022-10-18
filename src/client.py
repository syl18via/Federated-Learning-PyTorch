import numpy as np
from update import LocalUpdate, test_inference

class Client:
    def __init__(self, args, train_dataset, user_groups,
            client_idx, logger, global_model):
        self.args = args
        self.local_model = LocalUpdate(
                args=args,
                dataset=train_dataset,
                idxs=user_groups[client_idx],
                logger=logger,
                model=global_model)

    def train_step(self, global_model, epoch):
        _weight, loss = self.local_model.update_weights(
            global_model, global_round=epoch)
        return _weight, loss


