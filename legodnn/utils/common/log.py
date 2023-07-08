import logging
import os
import sys

from .others import get_cur_time_str  # 时间戳变字符串
from .file import ensure_dir  # 各种文件保存路径和文件夹创建


logger = logging.getLogger('zedl')  # logger名为zedl
logger.setLevel(logging.DEBUG)  # 低于调试级别就会被忽略
logger.propagate = False  # logger之间不会互相传输信息

formatter = logging.Formatter("%(asctime)s - %(filename)s[%(lineno)d] - %(levelname)s: %(message)s")  # 日志信息样式
log_dir_path = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), './log')  # 日志文件路径：当前位置的log文件夹
if not os.path.exists(log_dir_path):
    os.mkdir(log_dir_path)

# file log
cur_time_str = get_cur_time_str()
log_file_path = os.path.join(log_dir_path, cur_time_str[0:8], cur_time_str[8:] + '.log')  # log/年月日/时分秒.log
ensure_dir(log_file_path)
file_handler = logging.FileHandler(log_file_path, mode='a')  # 写入模式（非覆写）
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)  # 将输出的日志信息也写入到日志文件中

# cmd log
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

logging.getLogger('nni').setLevel(logging.ERROR)

# copy file content to log file
with open(os.path.abspath(sys.argv[0]), 'r') as f:  # 将文件内容也复制到log中方便调试排错
    content = f.read()
    logger.debug('entry file content: ---------------------------------')
    logger.debug('\n' + content)
    logger.debug('entry file content: ---------------------------------')
