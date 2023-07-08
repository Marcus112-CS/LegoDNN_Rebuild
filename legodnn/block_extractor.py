from .abstract_block_manager import AbstractBlockManager
from .utils.common.log import logger
from .utils.common.file import ensure_dir
from .utils.dl.common.model import save_model, ModelSaveMethod

import copy
import os


class BlockExtractor:
    def __init__(self, model, block_manager: AbstractBlockManager,
                 blocks_saved_dir, model_input_size,
                 device):

        self._model = model
        self._block_manager = block_manager
        self._blocks_saved_dir = blocks_saved_dir
        self._dummy_input_size = model_input_size
        self._device = device

    def _save_compressed_block(self, model, block_id, block_sparsity):  # 保存裁剪后的块在指定路径
        compressed_block = self._block_manager.get_pruned_block(model, block_id, block_sparsity,  # 获取修剪过的block
                                                            self._dummy_input_size, self._device)
        # exit(0)
        pruned_block_file_path = os.path.join(self._blocks_saved_dir,
                                              self._block_manager.get_block_file_name(block_id, block_sparsity))
        self._block_manager.save_block_to_file(compressed_block, pruned_block_file_path)
        logger.info('save pruned block {} (sparsity {}) in {}'.format(block_id, block_sparsity, pruned_block_file_path))
        logger.debug(compressed_block)

    def _compress_single_block(self, block_id, block_sparsity):  # 对单个块进行裁剪和存储
        # 深拷贝模型供压缩用
        model = copy.deepcopy(self._model)
        model = model.to(self._device)
        self._save_compressed_block(model, block_id, block_sparsity)  # 上一函数_save_compressed_block
        
    def _save_model_frame(self):  # 保存模型中间文件
        # model frame
        empty_model = copy.deepcopy(self._model)
        # print(empty_model)
        for block_id in self._block_manager.get_blocks_id(): # 当前模型的abstractblockmanager的所有block_id list
            self._block_manager.empty_block_in_model(empty_model, block_id)  # 清空模型中已有的block的对应位置
        model_frame_path = os.path.join(self._blocks_saved_dir, 'model_frame.pt')
        ensure_dir(model_frame_path)
        # print(empty_model)
        # exit(0)
        save_model(empty_model, model_frame_path, ModelSaveMethod.FULL)  # 全模型保存

    def extract_all_blocks(self):
        self._save_model_frame()  # 导出所有blocks之前先保存模型frame
        # exit(0)
        for i, block_id in enumerate(self._block_manager.get_blocks_id()):
            for block_sparsity in self._block_manager.get_blocks_sparsity()[i]:
                # 压缩特定稀疏度的特定块 [块id, 稀疏度] , 并进行保存
                print('\033[1;32m', '--> extracting {}: {} in sparsity {}'.format(block_id, [self._block_manager.graph.order_to_node.get(
                    num).get_name() for num in self._block_manager.blocks[int(block_id.split('-')[-1])]], block_sparsity), '\033[0m')
                self._compress_single_block(block_id, block_sparsity)
