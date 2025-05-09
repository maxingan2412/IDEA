import logging
import os
import sys
import os.path as osp

import datetime

def setup_logger(name, save_dir, if_train):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_dir:
        if not osp.exists(save_dir):
            os.makedirs(save_dir)

        # 获取当前时间戳
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M')

        # 根据模式生成日志文件名
        if if_train:
            log_filename = f"train_log_{timestamp}.txt"
        else:
            log_filename = f"test_log_{timestamp}.txt"

        fh = logging.FileHandler(os.path.join(save_dir, log_filename), mode='w')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
