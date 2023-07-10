import time
import os
import json
from datetime import datetime, timedelta

class time_logger:
    """
    used for logging the time of each step & total time, and save the log to a json file

    """
    def __init__(self, log_dir, title=None):
        self.log_dir = log_dir
        self.title = title
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir, exist_ok=True)

    def start(self):
        # 获取当前时间
        begin_time = datetime.now()
        current_time = time.perf_counter()
        # 格式化为字符串，例如：20230708_11:30:34
        formatted_time = begin_time.strftime("%Y%m%d_%H:%M:%S")
        # 拼接日志文件名
        self.file_dir = os.path.join(self.log_dir, formatted_time + '.json')
        self.log_content = {'title': self.title,
                            'start_time': begin_time.strftime("%Y-%m-%d %H:%M:%S"),
                            'end_time': None,
                            'total_time': None,
                            'execution_time_list': [],
                            'time_node': [begin_time.strftime("%Y-%m-%d %H:%M:%S")],
                            'source_time_node': [current_time],
                            'execution_time_dict': {}}

    def perf2time(self, time_delta):
        milliseconds = int(time_delta * 1000)
        delta = timedelta(milliseconds=milliseconds)
        time_obj = datetime(1, 1, 1) + delta
        time_str = time_obj.strftime("%H:%M:%S.%f")[:]
        return time_str

    def lap(self, time_delta_name=None):
        """
        record the time of each step

        """
        current_time = time.perf_counter()
        time_delta = current_time - self.log_content['source_time_node'][-1]
        self.log_content['execution_time_list'].append(self.perf2time(time_delta))
        self.log_content['time_node'].append(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        self.log_content['source_time_node'].append(current_time)
        if time_delta_name is not None:
            self.log_content['execution_time_dict'][time_delta_name] = self.perf2time(time_delta)

    def save(self):
        with open(self.file_dir, 'w') as f:
            json.dump(self.log_content, f, indent=4)

    def end(self, time_delta_name=None):
        end_time = datetime.now()
        current_time = time.perf_counter()
        time_delta = current_time - self.log_content['source_time_node'][-1]
        total_time_delta = current_time - self.log_content['source_time_node'][0]
        self.log_content['end_time'] = end_time.strftime("%Y-%m-%d %H:%M:%S")
        self.log_content['total_time'] = self.perf2time(total_time_delta)
        self.log_content['execution_time_list'].append(self.perf2time(time_delta))
        self.log_content['time_node'].append(end_time.strftime("%Y-%m-%d %H:%M:%S"))
        self.log_content['source_time_node'].append(current_time)
        if time_delta_name is not None:
            self.log_content['execution_time_dict'][time_delta_name] = self.perf2time(time_delta)
        self.save()

