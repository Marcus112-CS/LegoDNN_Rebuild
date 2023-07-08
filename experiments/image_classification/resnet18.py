import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from legodnn.utils.dl.common.env import set_random_seed
set_random_seed(0)

import sys
sys.setrecursionlimit(100000) # 最大随机深度（与资源占用有关）
import torch

from legodnn import BlockExtractor, BlockTrainer, ServerBlockProfiler, EdgeBlockProfiler, OptimalRuntime
from legodnn.gen_series_legodnn_models import gen_series_legodnn_models
from legodnn.block_detection.model_topology_extraction import topology_extraction  # 构建模型拓扑结构
from legodnn.presets.auto_block_manager import AutoBlockManager
from legodnn.presets.common_detection_manager_1204_new import CommonDetectionManager
from legodnn.model_manager.common_model_manager import CommonModelManager  # AbstractModelManager的实现
from legodnn.utils.common.file import experiments_model_file_path
from legodnn.utils.dl.common.model import get_module, set_module, get_model_size

from cv_task.datasets.image_classification.cifar_dataloader import CIFAR10Dataloader, CIFAR100Dataloader
from cv_task.image_classification.cifar.models import resnet18
from cv_task.image_classification.cifar.legodnn_configs import get_cifar100_train_config_200e
dataset_root_dir = '/home/marcus/newspace/datasets'

if __name__ == '__main__':
    cv_task = 'image_classification'
    dataset_name = 'cifar100'
    model_name = 'resnet18'
    method = 'legodnn'
    device = 'cuda'
    compress_layer_max_ratio = 0.125  # 最大压缩比？？？
    model_input_size = (1, 3, 32, 32)
    
    block_sparsity = [0.0, 0.2, 0.4, 0.6, 0.8]
    root_path = os.path.join('results/legodnn', cv_task, model_name+'_'+dataset_name + '_' + str(compress_layer_max_ratio).replace('.', '-'))

    compressed_blocks_dir_path = root_path + '/compressed'  # 压缩后的block
    trained_blocks_dir_path = root_path + '/trained'  # 训练过的block
    descendant_models_dir_path = root_path + '/descendant'  # 后代block
    block_training_max_epoch = 65
    test_sample_num = 100
    
    checkpoint = '/home/marcus/newspace/LegoDNN_teacher_model/cifar100/resnet18/2023-06-28/13-03-18/resnet18.pth'
    teacher_model = resnet18(num_classes=100).to(device)
    teacher_model.load_state_dict(torch.load(checkpoint)['net'])  # 权重导入（初次训练没有）

    print('\033[1;36m-------------------------------->    BUILD LEGODNN GRAPH\033[0m')  # 构建拓扑结构
    model_graph = topology_extraction(teacher_model, model_input_size, device=device, mode='unpack')  # pack/unpack
    model_graph.print_ordered_node()  # 按顺序打印节点
    
    print('\033[1;36m-------------------------------->    START BLOCK DETECTION\033[0m')  # 根据拓扑结构探测block
    detection_manager = CommonDetectionManager(model_graph, max_ratio=compress_layer_max_ratio)
    detection_manager.detection_all_blocks()
    detection_manager.print_all_blocks()

    # modelmanager和blockmanager
    model_manager = CommonModelManager()
    block_manager = AutoBlockManager(block_sparsity, detection_manager, model_manager)
    
    print('\033[1;36m-------------------------------->    START BLOCK EXTRACTION\033[0m')  # block导出
    block_extractor = BlockExtractor(teacher_model, block_manager, compressed_blocks_dir_path, model_input_size, device)
    block_extractor.extract_all_blocks()  # 按稀疏度导出blocks

    print('\033[1;36m-------------------------------->    START BLOCK TRAIN\033[0m')
    # num_workers>=1 报错 ValueError: signal number 32 out of range
    train_loader, test_loader = CIFAR100Dataloader(root_dir=dataset_root_dir, num_workers=0, train_batch_size=128, test_batch_size=128)
    print("\033[32mDataloader done\033[0m")
    block_trainer = BlockTrainer(teacher_model, block_manager, model_manager, compressed_blocks_dir_path,
                                 trained_blocks_dir_path, block_training_max_epoch, train_loader, device=device)
    print("\033[32mDatatrainer initialized\033[0m")
    block_trainer.train_all_blocks()
    print("\033[32mBlock trained\033[0m")

    # memory,accuracy profiler (original blocks & compressed blocks)
    server_block_profiler = ServerBlockProfiler(teacher_model, block_manager, model_manager,
                                                trained_blocks_dir_path, test_loader, model_input_size, device)
    server_block_profiler.profile_all_blocks()

    # latency profiler (original blocks & compressed blocks)
    edge_block_profiler = EdgeBlockProfiler(block_manager, model_manager, trained_blocks_dir_path, 
                                            test_sample_num, model_input_size, device)
    edge_block_profiler.profile_all_blocks()

    optimal_runtime = OptimalRuntime(trained_blocks_dir_path, model_input_size,
                                     block_manager, model_manager, device)
    model_size_min = get_model_size(torch.load(os.path.join(compressed_blocks_dir_path, 'model_frame.pt')))/1024**2
    model_size_max = get_model_size(teacher_model)/1024**2 + 1
    gen_series_legodnn_models(deadline=100, model_size_search_range=[model_size_min, model_size_max], target_model_num=100, optimal_runtime=optimal_runtime, descendant_models_save_path=descendant_models_dir_path, device=device)