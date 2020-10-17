import os

from config import Config
from messenger import Messenger
from run import app


class QueueManager(object):
    """
        queue_data_listの要素を循環させ、バッチ処理が完了する度に対応する要素を削除する
        処理中に障害が発生しない限り、全ての処理は同一時間で完了するという仮定に基づく簡易的実装
    """

    def __init__(self, data_num: int) -> None:
        self.data_num = data_num
        self.queue_data_list = list(range(self.data_num))
        app.logger.info(f"QueueManager initialized with {self.data_num} completions!")
            
    def pop_queue_data(self) -> int:
        # 先頭のデータを取り出し末尾に追加する
        queue_data = self.queue_data_list.pop(0)
        self.queue_data_list.append(queue_data)
        app.logger.info(f"Get queue data: {queue_data}")
        return queue_data
    
    def delete_queue_data(self, processed_data: int) -> None:
        app.logger.info(f"Processed queue data: {processed_data}")
        # 処理が完了したデータをリストから削除する
        self.queue_data_list.remove(processed_data)
        app.logger.info(f"Number of queue data is {len(self.queue_data_list)}")
        # 全てのバッチ処理が完了した際に正常終了する
        if len(self.queue_data_list) == 0:
            message = 'Batch processes completed!'
            app.logger.info(message)
            Messenger.send_message(username = 'Batch Manager', message=message)
            os._exit(0)