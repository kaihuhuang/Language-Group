import logging
import os
import datetime


def get_Mylogger(logger_name, base_path):
    logger = logging.getLogger(logger_name)  
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s', '%Y-%m-%d %H:%M:%S')
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)  
    console_handler.setFormatter(formatter)
    current_datetime = datetime.datetime.now().strftime('%Y-%m-%d')
    log_filename = f'{logger_name}_{current_datetime}.log'
    log_file_path = os.path.join(base_path, log_filename)
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger


if __name__ == '__main__':
    print("ok")
