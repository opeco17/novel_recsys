import datetime
import sys
sys.path.append('..')

import torch

from config import Config
from run import app
from train.model import ArcMarginProduct, BERT, OptimizerCreater
from train.dataloader import DataLoaderCreater


NUM_EPOCHS = Config.NUM_EPOCHS
PARAMITER_DIR = Config.PARAMETER_DIR


class Trainer(object):

    num_epochs = NUM_EPOCHS
    parameter_dir = PARAMITER_DIR

    @classmethod
    def train(cls) -> bool:
        try:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

            bert = BERT().to(device).train()
            metric_fc = ArcMarginProduct(device=device).to(device).train()
            criterion = torch.nn.CrossEntropyLoss()
            optimizer = OptimizerCreater.create_optimizer(bert, metric_fc)
            train_dataloader = DataLoaderCreater.create_dataloader(train=True)

            c = 0
            for epoch in range(NUM_EPOCHS):
                for ids, mask, label in train_dataloader:
                    c += 1
                    app.logger.info(c)

                    ids.to(device)
                    mask.to(device)
                    label.to(device)
                    optimizer.zero_grad()

                    feature = bert(ids, mask)
                    output = metric_fc(feature, label)
                    loss = criterion(output, label)
                    loss_list.append(loss.item())
                    
                    loss.backward()
                    optimizer.step()
                    
                    clear_output()

            app.logger.info('Training completed!')
            torch.save(bert.state_dict(), os.path.join(cls.parameter_dir, f'bert-{datetime.date.today()}.pth'))
            torch.save(metric_fc.state_dict(), os.path.join(cls.parameter_dir, f'metric_fc-{datetime.date.today()}.pth'))
            return True
            
        except Exception as e:
            app.logger.error(f"Error: {e}")
            return False
    