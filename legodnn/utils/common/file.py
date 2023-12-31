import os
import shutil  # 复制、移动、删除文件/创建压缩包

def ensure_dir(file_path: str):  # 容错性创建文件夹
    """Create it if the directory of :attr:`file_path` is not existed.

    Args:
        file_path (str): Target file path.
    """

    if not os.path.isdir(file_path):
        file_path = os.path.dirname(file_path)

    if not os.path.exists(file_path):
        print(file_path)
        os.makedirs(file_path)

def create_dir(dir_path: str):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def remove_dir(dir_path: str):
    if os.path.exists(dir_path):
        # os.rmdir(dir_path)
        shutil.rmtree(dir_path)

        # compressed_model_save_path = experiments_model_file_path('./', cv_task, dataset_name, model_name, method, 'compressed', sparsity)
        
def experiments_model_file_path(root_path, cv_task, dataset_name, model_name, compress_method, mode, model_sparsity): # 实验数据的保存路径生成（按格式）
    if isinstance(model_sparsity, float):
        if model_sparsity == 0:
            model_sparsity = 0.0
        s_name = str(model_sparsity).split('.')[-1]
    else:
        s_name = model_sparsity
    
    p = 'results/{}/{}_{}/{}/{}/s{}.pt'.format(cv_task, dataset_name, model_name, compress_method, str(mode), s_name)
    return os.path.join(root_path, p)  # result/任务/数据集_模型/压缩方法/运行模式/s{稀疏度}.pt 参数保存

# 一个baseline，一个legodnn
def compressed_model_file_path(root_path, cv_task, compress_method, dataset_name, model_name, model_sparsity):  # 和上方experiments_model_file_path一样
    if model_sparsity == 0:
        model_sparsity = 0.0
    p = './results/baselines/{}/{}/{}/{}/s{}.pt'.format(cv_task, compress_method, dataset_name, model_name, str(model_sparsity).split('.')[-1])
    return os.path.join(root_path, p)

def legodnn_blocks_dir_path(root_path, cv_task, compress_method, dataset_name, teacher_model_name, teacher_model_sparsity):
    if teacher_model_sparsity == 0:
        teacher_model_sparsity = 0.0
    p = './results/legodnn/{}/{}/{}/{}/s{}'.format(cv_task, dataset_name, teacher_model_name, compress_method, 
                                                     str(teacher_model_sparsity).split('.')[-1])
    return os.path.join(root_path, p)

# def compressed_model_file_path(root_path, cv_task, compress_method, dataset_name, model_name, model_sparsity):
#     if model_sparsity == 0:
#         model_sparsity = 0.0
#     p = './results/baselines/{}/{}/{}/{}/s{}.pt'.format(cv_task, compress_method, dataset_name, model_name, str(model_sparsity).split('.')[-1])
#     return os.path.join(root_path, p)

# def legodnn_blocks_dir_path(root_path, cv_task, compress_method, dataset_name, teacher_model_name, teacher_model_sparsity):
#     if teacher_model_sparsity == 0:
#         teacher_model_sparsity = 0.0
#     p = './results/legodnn/{}/{}/{}/{}/s{}'.format(cv_task, dataset_name, teacher_model_name, compress_method, 
#                                                      str(teacher_model_sparsity).split('.')[-1])
#     return os.path.join(root_path, p)