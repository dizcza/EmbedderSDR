from trainer.trainer import Trainer


class Test(Trainer):
    def train_batch(self, images, labels):
        return None, None
