from trainer.trainer import Trainer


class Test(Trainer):
    def train_batch(self, images, labels):
        return None, None

    def train_epoch(self, epoch):
        self.timer.batch_id += self.timer.batches_in_epoch
        return None
