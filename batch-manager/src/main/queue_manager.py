import os

from config import Config
from logger import logger
from messenger import Messenger


class QueueManager(object):
    """
        queue_data_list内のデータをバッチ処理していく
        バッチ処理が失敗した場合はqueue_data_listにデータが再び追加される
        バッチ処理が成功した場合はprocessed_data_listにデータが追加される
        processed_data_list内に全データが格納されると正常終了する
    """

    def __init__(self, data_num: int) -> None:
        self.data_num = data_num
        self.queue_data_list = list(range(self.data_num))
        self.processed_data_list = []
        logger.info(f"QueueManager initialized with {self.data_num} completions!")
            
    def pop_queue_data(self) -> int:
        if self.queue_data_list:
            queue_data = self.queue_data_list.pop(0)
            logger.info(f"Pop queue data: {queue_data}")
        else:
            queue_data = None
            logger.info('queue data not exists.')
        return queue_data
    
    def fail_queue_data(self, queue_data: int) -> None:
        """失敗したデータをqueue_data_listに追加する"""
        self.queue_data_list.append(queue_data)
        logger.info(f"Fail queue data: {queue_data}")
    
    def success_queue_data(self, queue_data: int):
        """成功したデータをprocessed_data_listへ追加する"""
        self.processed_data_list.append(queue_data)
        logger.info(f"Success queue data: {queue_data}")
        if len(self.processed_data_list) == self.data_num:
            message = f"{Config.TARGET} completed!"
            logger.info(message)
            Messenger.send_message(username=Config.TARGET, text=message)
            os._exit(0)