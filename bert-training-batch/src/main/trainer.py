import datetime
import os

import torch

from config import Config
from logger import get_logger
from model import ArcFace, BERT, OptimizerCreater
from dataloader import DataLoaderCreater


NUM_EPOCHS = Config.NUM_EPOCHS
PARAMITER_DIR = Config.PARAMETER_DIR

logger = get_logger()


class Trainer(object):

    num_epochs = NUM_EPOCHS
    parameter_dir = PARAMITER_DIR

    @classmethod
    def train(cls) -> bool:
        
        try:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

            bert = BERT().to(device).train()
            metric_fc = ArcFace(device=device).to(device).train()
            criterion = torch.nn.CrossEntropyLoss()
            optimizer = OptimizerCreater.create_optimizer(bert, metric_fc)
            train_dataloader = DataLoaderCreater.create_dataloader(is_train=True)

            c = 0
            logger.info('Training start!')
            for epoch in range(NUM_EPOCHS):
                for ids, mask, label in train_dataloader:
                    c += 1
                    logger.info(f"Training progress: {c}")

                    ids.to(device)
                    mask.to(device)
                    label.to(device)
                    optimizer.zero_grad()

                    feature = bert(ids, mask)
                    output = metric_fc(feature, label)
                    loss = criterion(output, label)
                    
                    loss.backward()
                    optimizer.step()

            logger.info('Training completed!')
            torch.save(bert.state_dict(), os.path.join(cls.parameter_dir, f'bert-{datetime.date.today()}.pth'))
            torch.save(metric_fc.state_dict(), os.path.join(cls.parameter_dir, f'metric_fc-{datetime.date.today()}.pth'))
            return True
            
        except Exception as e:
            logger.error(f"Error: {e}")
            return False
    
    
def main():
    success = Trainer.train()
    logger.info(f"success: {success}")
    
    
if __name__ == '__main__':
    main()