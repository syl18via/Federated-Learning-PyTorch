import numpy as np
from update import LocalUpdate, test_inference
user2local_model = {}


class Client:
    def __init__(self, args, train_dataset, user_groups,
            client_idx, logger, global_model) -> None:
        self.args = args
        self.local_model = LocalUpdate(
                args=args,
                dataset=train_dataset,
                idxs=user_groups[client_idx],
                logger=logger,
                model=global_model)
        user2local_model[client_idx]= self.local_model
        
    def train_step(self, global_model, epoch):
        _weight, loss = self.local_model.update_weights(
            global_model, global_round=epoch)
        return _weight, loss

def get_local_model_fn(idx):
    if idx == "shap":
        return user2local_model["shap"]
    else:
        return user2local_model[idx]

