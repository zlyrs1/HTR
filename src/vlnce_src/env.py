import os
import json
import copy
import lmdb
import time
import tqdm
import torch
import random
import airsim
import threading
import open_clip
import numpy as np
import msgpack_numpy
from gym import spaces
from pathlib import Path
from fastdtw import fastdtw
from typing import Dict, List, Optional, Any

from src.common.param import args
from utils.logger import logger
from airsim_plugin.AirVLNSimulatorClientTool import AirVLNSimulatorClientTool
from airsim_plugin.airsim_settings import AirsimActions, AirsimActionSettings
from utils.env_utils import SimState, getPoseAfterMakeAction
from utils.env_vector import VectorEnvUtil
from utils.shorest_path_sensor import EuclideanDistance3


# # 加载json文件中的导航数据，原始版本，验证时使用
# # 该版本在split中缺少数字时自动选取整个数据集
def load_my_datasets(splits) -> List[Dict[str, Any]]:
    data = []
    old_state = random.getstate()
    for split in splits:
        components = split.split("@")
        number = -1
        if len(components) > 1:
            split, number = components[0], int(components[1])

        # 读取json文件到new_data
        with open(str(Path(args.project_prefix) / 'DATA/data/aerialvln-s/{}.json'.format(split)), 'r', encoding='utf-8') as f:
            new_data = json.load(f)
            new_data = new_data['episodes']

        # Partition
        if number > 0:
            random.seed(1)          # Make the data deterministic, additive
            random.shuffle(new_data)
            new_data = new_data[:number]

        # Join
        data += new_data
    random.setstate(old_state)      # Recover the state of the random generator
    return data


# 加载json文件中的导航数据:
# 1)既可以用于收集AirVLN02的预训练数据
# 2)也可以用于针对均匀分布小样本量的参数验证
# 其本质就是保证每一类导航场景能够均匀的采样，按照场景ID均匀的抽取导航样本
# def load_my_datasets(splits) -> List[Dict[str, Any]]:
#     data = []
#     old_state = random.getstate()
#
#     for split in splits:
#         components = split.split("@")
#         if len(components) > 1:
#             split, number = components[0], int(components[1])
#
#         # 读取json文件到new_data
#         with open(str(Path(args.project_prefix) / 'DATA/data/aerialvln-s/{}.json'.format(split)), 'r', encoding='utf-8') as f:
#             new_data = json.load(f)
#             new_data = new_data['episodes']
#
#         # 按照scene_id进行分组
#         scene_groups = {}
#         for item in new_data:
#             scene_id = str(item['scene_id'])  # 获取场景id
#             if scene_id not in scene_groups:
#                 scene_groups[scene_id] = []
#             scene_groups[scene_id].append(item)
#
#         # 对每个scene_id分组，随机选取前N个样本
#         sampled_data = []
#         for scene_id, scene_samples in scene_groups.items():
#             random.seed(1)  # 保证结果可重复
#             random.shuffle(scene_samples)  # 对该场景中的样本进行随机打乱
#             sampled_data += scene_samples[:number]  # 取前前N个样本
#
#         # 加入到总数据
#         data += sampled_data
#
#     random.setstate(old_state)
#     return data


def resize_sequence(data_list: List[Any], target_length: int, pad_value: Any = '') -> List[Any]:
    """
    Adjust a list to a desired length by either padding it with pad_value or truncating it.
    Args:
        data_list (List[Any]): 输入的原始列表。
        target_length (int): 最终列表的目标长度。
        pad_value (Any): 用于填充的值。

    Returns:
        List[Any]: 一个长度为 target_length 的新列表。
    """
    current_length = len(data_list)

    if current_length > target_length:
        print(f"警告：列表被截断。原始长度: {current_length}, 目标长度: {target_length}")
        return data_list[:target_length]
    elif current_length < target_length:
        padding = [pad_value] * (target_length - current_length)
        return data_list + padding
    else:
        return data_list[:]


class AirVLNENV:
    def __init__(self, batch_size=8, split='train', seed=1, dataset_group_by_scene=True):
        self.seed = seed
        self.split = split
        self.batch_size = batch_size
        self.dataset_group_by_scene = dataset_group_by_scene

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        # step1: 加载CLIP模型和分词器
        script_dir = Path(__file__).parent.resolve()
        clip_weights_path = script_dir / 'laion' / 'CLIP-ViT-B-32-laion2B-s34B-b79K.bin'
        self.CLIPEncoder, _, _ = open_clip.create_model_and_transforms(
            'ViT-B-32',
            pretrained=str(clip_weights_path),
            device = self.device
        )
        self.tokenizer = open_clip.get_tokenizer('ViT-B-32')
        # 冻结CLIP模型的参数
        for param in self.CLIPEncoder.parameters():
            param.requires_grad = False
        self.CLIPEncoder.eval()

        # step2: 通过CLIP文本编码器处理导航文本并保存在self.data的新字段instructionCLIP
        self.ori_raw_data = load_my_datasets([split]).copy()
        logger.info(f'Loaded with {len(self.ori_raw_data)} instructions, using split: {split}')

        self.index_data = 0
        self.data = []
        pbar = tqdm.tqdm(total=len(self.ori_raw_data))
        for i_item, item in enumerate(self.ori_raw_data):
            # 如下两个条件用于在不同训练模式下筛选特定场景的样本
            if args.collect_type in ['TF']:
                if len(list(args.TF_mode_load_scene)) > 0 and str(item['scene_id']) not in list(args.TF_mode_load_scene):
                    pbar.update()
                    continue

            if args.collect_type in ['dagger', 'SF']:
                if len(list(args.dagger_mode_load_scene)) > 0 and str(item['scene_id']) not in list(args.dagger_mode_load_scene):
                    pbar.update()
                    continue

            new_item = dict(item).copy()

            # 利用CLIP文本编码器处理导航指令的各个字段
            item_start_time = time.time()

            SUBTASK_LEN = 25
            AE_ELEMENT_LEN = 10
            LE_ELEMENT_LEN = 5

            padded_subtasks = resize_sequence(item['instruction']['Subtask'], SUBTASK_LEN)
            padded_nA = resize_sequence(item['instruction']['nA'], SUBTASK_LEN)
            padded_nL = resize_sequence(item['instruction']['nL'], SUBTASK_LEN)

            padded_Ae_inner = [resize_sequence(subtask_elements, AE_ELEMENT_LEN) for subtask_elements in
                               item['instruction']['Ae']]
            empty_subtask_pad = [''] * AE_ELEMENT_LEN
            padded_Ae_outer = resize_sequence(padded_Ae_inner, SUBTASK_LEN, pad_value=empty_subtask_pad)
            padded_Ae_flatten = [element for sublist in padded_Ae_outer for element in sublist]

            padded_Le_inner = [resize_sequence(subtask_elements, LE_ELEMENT_LEN) for subtask_elements in
                               item['instruction']['Le']]
            empty_subtask_pad = [''] * LE_ELEMENT_LEN
            padded_Le_outer = resize_sequence(padded_Le_inner, SUBTASK_LEN, pad_value=empty_subtask_pad)
            padded_Le_flatten = [element for sublist in padded_Le_outer for element in sublist]

            all_texts_to_encode = (
                    padded_subtasks +
                    padded_nA +
                    padded_nL +
                    padded_Ae_flatten +
                    padded_Le_flatten
            )
            all_tokens = self.tokenizer(all_texts_to_encode).to(self.device)
            with torch.no_grad():
                all_embeddings = self.CLIPEncoder.encode_text(all_tokens).cpu()

            subtask_start = 0
            na_start = subtask_start + SUBTASK_LEN
            nl_start = na_start + SUBTASK_LEN
            ae_start = nl_start + SUBTASK_LEN
            le_start = ae_start + (SUBTASK_LEN * AE_ELEMENT_LEN)
            le_end = le_start + (SUBTASK_LEN * LE_ELEMENT_LEN)

            # 统一维度为： 512, element, subtask
            Subtask_embedding = all_embeddings[subtask_start:na_start].permute(1, 0)
            nA_embedding = all_embeddings[na_start:nl_start].permute(1, 0)
            nL_embedding = all_embeddings[nl_start:ae_start].permute(1, 0)

            Ae_embedding_flatten = all_embeddings[ae_start:le_start]
            Ae_embedding = Ae_embedding_flatten.view(SUBTASK_LEN, AE_ELEMENT_LEN, -1).permute(2, 1, 0)

            Le_embedding_flatten = all_embeddings[le_start:le_end]
            Le_embedding = Le_embedding_flatten.view(SUBTASK_LEN, LE_ELEMENT_LEN, -1).permute(2, 1, 0)

            item_end_time = time.time()
            print(f"总耗时: {item_end_time - item_start_time:.4f} 秒")

            new_item['instructionCLIP'] = {
                'subtask_embedding': Subtask_embedding,
                'nA_embedding': nA_embedding,
                'nL_embedding': nL_embedding,
                'Ae_embedding': Ae_embedding,
                'Le_embedding': Le_embedding,
            }
            self.data.append(new_item)
            pbar.update()
        pbar.close()

        # 1.2 建立轨迹id和指令tokens之间的对应关系
        # 同一个轨迹id对应三个不同的导航指令，将指令信息和episode保存下来，在collect中会同时收集他们的数据以节省计算资源
        self.trajectory_id_2_instruction_tokens = {}
        self.trajectory_id_2_episode_ids = {}
        for i_item, item in enumerate(self.data):
            if item['trajectory_id'] not in self.trajectory_id_2_instruction_tokens.keys():
                self.trajectory_id_2_instruction_tokens[item['trajectory_id']] = []
                self.trajectory_id_2_instruction_tokens[item['trajectory_id']].append(item['instructionCLIP'])
            else:
                self.trajectory_id_2_instruction_tokens[item['trajectory_id']].append(item['instructionCLIP'])

            if item['trajectory_id'] not in self.trajectory_id_2_episode_ids.keys():
                self.trajectory_id_2_episode_ids[item['trajectory_id']] = []
                self.trajectory_id_2_episode_ids[item['trajectory_id']].append(item['episode_id'])
            else:
                self.trajectory_id_2_episode_ids[item['trajectory_id']].append(item['episode_id'])

        # 对self.data中的数据进行排序，尽量保证相同场景ID的数据会被放在一起
        random.shuffle(self.data)
        if args.EVAL_NUM != -1 and int(args.EVAL_NUM) > 0:
            [random.shuffle(self.data) for i in range(10)]
            self.data = self.data[:int(args.EVAL_NUM)].copy()
        if dataset_group_by_scene:
            self.data = self._group_scenes()
            logger.warning('dataset grouped by scene')

        # 收集训练数据集的场景id
        # 26个场景中有17个场景用于进行训练，而其他的用于unseen scene进行测试
        scenes = [item['scene_id'] for item in self.data]
        self.scenes = set(scenes)

        # 定义观测空间和动作空间
        self.observation_space = spaces.Dict({
            "rgb": spaces.Box(low=0, high=255, shape=(args.Image_Height_RGB, args.Image_Width_RGB, 3), dtype=np.uint8),
            "depth": spaces.Box(low=0, high=1, shape=(args.Image_Height_DEPTH, args.Image_Width_DEPTH, 1), dtype=np.float32),
            "instruction": spaces.Discrete(0),
            "progress": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "teacher_action": spaces.Box(low=0, high=100, shape=(1,)),
        })
        self.action_space = spaces.Discrete(int(len(AirsimActions)))

        self.sim_states: Optional[List[SimState], List[None]] = [None for _ in range(batch_size)]
        self.last_scene_id_list = []
        self.one_scene_could_use_num = 5000
        self.this_scene_used_cnt = 0

        # 初始化LMDB数据库，并预加载已存在的特征键用于后续的去重检查
        if args.collect_type in ['TF']:
            if args.run_type in ['collect']:
                self.lmdb_features_dir = str(Path(args.project_prefix) / 'DATA' / 'img_features' / str(args.run_type) / str(args.name) / str(split))
                self.lmdb_rgb_dir = str(Path(args.project_prefix) / 'DATA' / 'img_features' / str(args.run_type) / str(args.name) / (str(split) + '_rgb'))
                self.lmdb_depth_dir = str(Path(args.project_prefix) / 'DATA' / 'img_features' / str(args.run_type) / str(args.name) / (str(split) + '_depth'))

                if not os.path.exists(str(self.lmdb_features_dir)):
                    os.makedirs(str(self.lmdb_features_dir), exist_ok=True)
                if not os.path.exists(str(self.lmdb_rgb_dir)):
                    os.makedirs(str(self.lmdb_rgb_dir), exist_ok=True)
                if not os.path.exists(str(self.lmdb_depth_dir)):
                    os.makedirs(str(self.lmdb_depth_dir), exist_ok=True)

                lmdb_features_map_size = 5.0e12
                lmdb_rgb_map_size = 5.0e12
                lmdb_depth_map_size = 5.0e12

                try:
                    # 打开数据库、检查现有数据量、准备写入操作、创建线程锁确保安全
                    self.lmdb_features_env = lmdb.open(self.lmdb_features_dir, map_size=int(lmdb_features_map_size), readahead=False,)
                    self.lmdb_features_start_id = self.lmdb_features_env.stat()["entries"]
                    self.lmdb_features_txn = self.lmdb_features_env.begin(write=True)
                    self.threading_lock_lmdb_features_txn = threading.Lock()
                    logger.info('init lmdb of {}, {}, lmdb_start_id: {}'.format(split, 'features', self.lmdb_features_start_id))

                    # 将数据库中的keys存储在set中，以便后续进行快速的重复性检查
                    self.lmdb_collected_keys = set()
                    with tqdm.tqdm(total=int(self.lmdb_features_start_id), dynamic_ncols=True) as pbar:
                        for key in self.lmdb_features_txn.cursor().iternext(keys=True, values=False):
                            pbar.update()
                            self.lmdb_collected_keys.add(key.decode())

                    self.lmdb_rgb_env = lmdb.open(self.lmdb_rgb_dir, map_size=int(lmdb_rgb_map_size), readahead=False, )
                    self.lmdb_rgb_start_id = self.lmdb_rgb_env.stat()["entries"]
                    self.lmdb_rgb_txn = self.lmdb_rgb_env.begin(write=True)
                    self.threading_lock_lmdb_rgb_txn = threading.Lock()
                    logger.info('init lmdb of {}, {}, lmdb_start_id: {}'.format(split, 'rgb', self.lmdb_rgb_start_id))

                    self.lmdb_depth_env = lmdb.open(self.lmdb_depth_dir, map_size=int(lmdb_depth_map_size), readahead=False, )
                    self.lmdb_depth_start_id = self.lmdb_depth_env.stat()["entries"]
                    self.lmdb_depth_txn = self.lmdb_depth_env.begin(write=True)
                    self.threading_lock_lmdb_depth_txn = threading.Lock()
                    logger.info('init lmdb of {}, {}, lmdb_start_id: {}'.format(split, 'depth', self.lmdb_depth_start_id))

                except lmdb.Error as err:
                    logger.error(err)
                    raise err

            if args.run_type in ['eval']:
                self.lmdb_features_dir = str(Path(args.project_prefix) / 'DATA' / 'img_features' / str(args.run_type) / str(args.name) / '{}_{}'.format(str(split), args.make_dir_time))

                if not os.path.exists(str(self.lmdb_features_dir)):
                    os.makedirs(str(self.lmdb_features_dir), exist_ok=True)

                lmdb_features_map_size = 1.0e11  # 1.0e6  1M

                try:
                    self.lmdb_features_env = lmdb.open(self.lmdb_features_dir, map_size=int(lmdb_features_map_size), readahead=False,)
                    self.lmdb_features_start_id = self.lmdb_features_env.stat()["entries"]
                    self.lmdb_features_txn = self.lmdb_features_env.begin(write=True)
                    self.threading_lock_lmdb_features_txn = threading.Lock()
                    logger.info('init lmdb of {}, {}, lmdb_start_id: {}'.format(split, 'features', self.lmdb_features_start_id))

                    self.lmdb_collected_keys = set()
                    with tqdm.tqdm(total=int(self.lmdb_features_start_id), dynamic_ncols=True) as pbar:
                        for key in self.lmdb_features_txn.cursor().iternext(keys=True, values=False):
                            pbar.update()
                            self.lmdb_collected_keys.add(key.decode())

                except lmdb.Error as err:
                    logger.error(err)
                    raise err

        if args.collect_type in ['dagger', 'SF']:
            self.lmdb_features_dir = str(Path(args.project_prefix) / 'DATA' / 'img_features' / str(args.run_type) / str(args.name) / str(split))

            if not os.path.exists(str(self.lmdb_features_dir)):
                os.makedirs(str(self.lmdb_features_dir), exist_ok=True)

            lmdb_features_map_size = 20.0e8  # 1.0e11  100GB  初始值20.0e12

            try:
                self.lmdb_features_env = lmdb.open(self.lmdb_features_dir, map_size=int(lmdb_features_map_size), readahead=False,)
                self.lmdb_features_start_id = self.lmdb_features_env.stat()["entries"]
                self.lmdb_features_txn = self.lmdb_features_env.begin(write=True)
                self.threading_lock_lmdb_features_txn = threading.Lock()
                logger.info('init lmdb of {}, {}, lmdb_start_id: {}'.format(split, 'features', self.lmdb_features_start_id))

                self.lmdb_collected_keys = set()
                with tqdm.tqdm(
                    total=int(self.lmdb_features_start_id), dynamic_ncols=True
                ) as pbar:
                    for key in self.lmdb_features_txn.cursor().iternext(keys=True, values=False):
                        pbar.update()
                        if len(str(key.decode()).split('_')) <= 1:
                            self.lmdb_collected_keys.add(
                                '{}_0'.format(key.decode())
                            )
                        else:
                            self.lmdb_collected_keys.add(key.decode())

            except lmdb.Error as err:
                logger.error(err)
                raise err

        self.init_VectorEnvUtil()

    def _group_scenes(self):
        # 按照场景id对
        assert self.dataset_group_by_scene, 'error args param'

        scene_sort_keys: Dict[str, int] = {}
        for item in self.data:
            if str(item['scene_id']) not in scene_sort_keys:
                scene_sort_keys[str(item['scene_id'])] = len(scene_sort_keys)

        return sorted(self.data, key=lambda e: scene_sort_keys[str(e['scene_id'])])

    def init_VectorEnvUtil(self):
        self.delete_VectorEnvUtil()
        self.load_scenes = [int(_scene) for _scene in list(self.scenes)]
        self.VectorEnvUtil = VectorEnvUtil(self.load_scenes, self.batch_size)

    def delete_VectorEnvUtil(self):
        if hasattr(self, 'VectorEnvUtil'):
            del self.VectorEnvUtil
        # 通过垃圾回收模块gc的collect函数回收垃圾并释放内存
        import gc
        gc.collect()

    def next_minibatch(self, skip_scenes=[], data_it=0):
        # 能够从训练数据中取出新的batch进行训练
        batch = []
        while True:
            # 这里考虑的是最后一个batch的情况
            if self.index_data >= len(self.data)-1:
                # 对于最后一个batch不够的情况，差多少补多少
                random.shuffle(self.data)
                logger.warning('random shuffle data')
                if self.dataset_group_by_scene:
                    self.data = self._group_scenes()
                    logger.warning('dataset grouped by scene')

                if len(batch) == 0:
                    self.index_data = 0
                    self.batch = None
                    return

                # 虽然index_data变小了，但是不影响退出循环
                self.index_data = self.batch_size - len(batch)
                batch += self.data[:self.index_data]
                break

            new_episode = self.data[self.index_data]

            if new_episode['scene_id'] in skip_scenes:
                self.index_data += 1
                continue

            if args.run_type in ['collect', 'train'] and args.collect_type in ['TF']:
                lmdb_key = '{}'.format(new_episode['episode_id'])
                if lmdb_key in self.lmdb_collected_keys:
                    self.index_data += 1
                    continue
                else:
                    batch.append(new_episode)
                    self.index_data += 1
            elif args.run_type in ['collect', 'train'] and args.collect_type in ['dagger', 'SF']:
                lmdb_key = '{}_{}'.format(new_episode['episode_id'], data_it)
                if lmdb_key in self.lmdb_collected_keys:
                    self.index_data += 1
                    continue
                else:
                    batch.append(new_episode)
                    self.index_data += 1
            else:
                batch.append(new_episode)
                self.index_data += 1

            # 取满当前batch的样本后退出循环
            if len(batch) == self.batch_size:
                break

        self.batch = copy.copy(batch)
        assert len(self.batch) == self.batch_size, 'next_minibatch error'

        self.VectorEnvUtil.set_batch(self.batch)

    def changeToNewEpisodes(self):
        # 构建导航场景以及创建无人机对应的airsim客户端
        self._changeEnv(need_change=False)

        # 设置无人机起始时刻的位姿以及将数据存储在self.sim_states中
        self._setEpisodes()

        # 更新指标数据
        self.update_measurements()

    def _changeEnv(self, need_change: bool = True):
        # 获取当前batch中样本对应的场景id
        scene_id_list = [item['scene_id'] for item in self.batch]
        assert len(scene_id_list) == self.batch_size, '错误'

        # 保证所有仿真客户端的最大场景数量之和不大于batch size
        machines_info_template = copy.deepcopy(args.machines_info)
        total_max_scene_num = 0
        for item in machines_info_template:
            total_max_scene_num += item['MAX_SCENE_NUM']
        assert self.batch_size <= total_max_scene_num, 'error args param: batch_size'

        # 构造机器信息 TODO
        # 确定当前batch要打开的场景并更新到machines_info中的open_scenes中
        # machines_info为所有仿真客户端的信息，目前只有一个仿真客户端
        machines_info = []
        ix = 0
        for index, item in enumerate(machines_info_template):
            machines_info.append(item)
            delta = min(self.batch_size, item['MAX_SCENE_NUM'], len(scene_id_list)-ix)
            machines_info[index]['open_scenes'] = scene_id_list[ix: ix + delta]
            ix += delta

        # 校验所有仿真客户端构建的导航场景数总和要等于batch size
        cnt = 0
        for item in machines_info:
            cnt += len(item['open_scenes'])
        assert self.batch_size == cnt, 'error create machines_info'

        # 确定是否要主机端去更换导航场景
        # 例如上轮batch的16个样本都是场景12,这次batch的样本也都是12,则可以直接沿用上一轮batch的
        if self.this_scene_used_cnt < self.one_scene_could_use_num and \
                len(set(scene_id_list)) == 1 and len(set(self.last_scene_id_list)) == 1 and \
                scene_id_list[0] is not None and self.last_scene_id_list[0] is not None and scene_id_list[0] == self.last_scene_id_list[0] and \
                need_change == False:
            self.this_scene_used_cnt += 1
            logger.warning('no need to change env: {}'.format(scene_id_list))
            return
        else:
            logger.warning('to change env: {}'.format(scene_id_list))

        # 只有成功拉起仿真客户端才会跳出这个循环
        while True:
            try:
                self.machines_info = copy.deepcopy(machines_info)
                if (not args.ablate_rgb or not args.ablate_depth):
                    self.simulator_tool = AirVLNSimulatorClientTool(machines_info=self.machines_info)
                    self.simulator_tool.run_call()
                    logger.info("当前batch样本对应的导航场景和无人机airsim客户端创建成功！！！")
                break
            except Exception as e:
                logger.error("启动场景失败 {}".format(e))
                time.sleep(3)
            except:
                logger.error('启动场景失败')
                time.sleep(3)

        self.last_scene_id_list = scene_id_list.copy()
        self.this_scene_used_cnt = 1

    def _setEpisodes(self):
        # self.simulator_tool客户端读取样本的起始位姿对无人机在仿真场景中的位置进行设置
        start_position_list = [item['start_position'] for item in self.batch]
        start_rotation_list = [item['start_rotation'] for item in self.batch]

        poses = []
        cnt = 0
        for index_1, item in enumerate(self.machines_info):
            poses.append([])
            for index_2, _ in enumerate(item['open_scenes']):
                pose = airsim.Pose(
                    position_val=airsim.Vector3r(
                        x_val=start_position_list[cnt][0],
                        y_val=start_position_list[cnt][1],
                        z_val=start_position_list[cnt][2],
                    ),
                    orientation_val=airsim.Quaternionr(
                        x_val=start_rotation_list[cnt][1],
                        y_val=start_rotation_list[cnt][2],
                        z_val=start_rotation_list[cnt][3],
                        w_val=start_rotation_list[cnt][0],
                    ),
                )
                poses[index_1].append(pose)
                cnt += 1

        if not args.ablate_rgb or not args.ablate_depth:
            result = self.simulator_tool.setPoses(poses=poses)
            if not result:
                logger.error('设置位置失败')
                self.reset_to_this_pose(poses)

        # 收集所有无人机step=0时的状态数据self.sim_states
        cnt = 0
        for index_1, item in enumerate(self.machines_info):
            for index_2, _ in enumerate(item['open_scenes']):
                pose = airsim.Pose(
                    position_val=airsim.Vector3r(
                        x_val=start_position_list[cnt][0],
                        y_val=start_position_list[cnt][1],
                        z_val=start_position_list[cnt][2],
                    ),
                    orientation_val=airsim.Quaternionr(
                        x_val=start_rotation_list[cnt][1],
                        y_val=start_rotation_list[cnt][2],
                        z_val=start_rotation_list[cnt][3],
                        w_val=start_rotation_list[cnt][0],
                    ),
                )
                self.sim_states[cnt] = SimState(index=cnt, step=0, episode_info=self.batch[cnt], pose=pose)
                self.sim_states[cnt].trajectory = [[
                    pose.position.x_val, pose.position.y_val, pose.position.z_val,  # xyz
                    pose.orientation.x_val, pose.orientation.y_val, pose.orientation.z_val, pose.orientation.w_val,  # xyzw
                ]]
                cnt += 1

    def get_obs(self):
        # 获取无人机当前的观测(前视图像)obs和状态量的结合体
        obs_states = self._getStates()

        obs, states = self.VectorEnvUtil.get_obs(obs_states)
        self.sim_states = states

        return obs

    def _getStates(self):
        while True:
            if not args.ablate_rgb or not args.ablate_depth:
                responses = self.simulator_tool.getImageResponses(get_rgb=not bool(args.ablate_rgb),
                                                                  get_depth=not bool(args.ablate_depth))
            else:
                responses = [[(None, None) for j in range(self.batch_size)] for i in range(len(self.machines_info))]

            if responses is None:
                poses = self._get_current_pose()
                self.reset_to_this_pose(poses)
                time.sleep(3)
            else:
                break

        # 校验是否获取了所有batch样本的前视图像
        cnt = 0
        for item in responses:
            cnt += len(item)
        assert len(responses) == len(self.machines_info), 'error'
        assert cnt == self.batch_size, 'error'

        # 只有验证脚本文件符合条件
        # 后期这个逻辑需要修改，collect是否真的需要验证"将要发生碰撞"
        if args.run_type in ['eval'] or (args.run_type in ['collect'] and args.collect_type in ['dagger']):
            cnt = 0
            for index_1, item in enumerate(self.machines_info):
                for index_2 in range(len(item['open_scenes'])):
                    depth_image = responses[index_1][index_2][1]
                    collision_sensor_result = (np.array(depth_image) < 0.004).sum() / np.array(depth_image).flatten().shape[0]
                    if collision_sensor_result > 0.1:
                        self.sim_states[cnt].is_collisioned = True
                        self.sim_states[cnt].is_end = True
                        logger.warning('collisioned: {}'.format(cnt))
                    cnt += 1

        #
        states = [None for _ in range(self.batch_size)]
        cnt = 0
        for index_1, item in enumerate(self.machines_info):
            for index_2 in range(len(item['open_scenes'])):
                rgb_image = responses[index_1][index_2][0]
                if rgb_image is not None:
                    _rgb_image = np.array(rgb_image)
                else:
                    _rgb_image = None

                depth_image = responses[index_1][index_2][1]
                if depth_image is not None:
                    _depth_image = np.array(depth_image)
                else:
                    _depth_image = None

                state = self.sim_states[cnt]
                states[cnt] = (_rgb_image, _depth_image, state)
                cnt += 1

                # 针对收集数据阶段
                # 以当前节点的步数和轨迹id作为标识KEY存储对应的rgb和深度图像
                if self.split in ['train'] and args.run_type in ['collect'] and args.collect_type in ['TF']:
                    trajectory_id = state.episode_info['trajectory_id']
                    step = state.step
                    lmdb_rgb_key = '{}_{}_rgb'.format(trajectory_id, step)
                    lmdb_depth_key = '{}_{}_depth'.format(trajectory_id, step)

                    if rgb_image is not None:
                        self.threading_lock_lmdb_rgb_txn.acquire()
                        self.lmdb_rgb_txn.put(
                            lmdb_rgb_key.encode(),
                            msgpack_numpy.packb(
                                rgb_image, use_bin_type=True
                            ),
                        )
                        self.threading_lock_lmdb_rgb_txn.release()

                    if depth_image is not None:
                        self.threading_lock_lmdb_depth_txn.acquire()
                        self.lmdb_depth_txn.put(
                            lmdb_depth_key.encode(),
                            msgpack_numpy.packb(
                                depth_image, use_bin_type=True
                            ),
                        )
                        self.threading_lock_lmdb_depth_txn.release()

        return states

    def _get_current_pose(self) -> list:
        # 获取当前所有无人机的位姿信息
        # self.sim_states应该每个时刻都被更新
        poses = []
        cnt = 0
        for index_1, item in enumerate(self.machines_info):
            poses.append([])
            for index_2, _ in enumerate(item['open_scenes']):
                poses[index_1].append(self.sim_states[cnt].pose)
                cnt += 1
        return poses

    def reset(self):
        self.changeToNewEpisodes()
        return self.get_obs()

    def reset_to_this_pose(self, poses):
        # 强行设置所有无人机的位置为poses，如果失败的话会打印失败信息并刷新场景后重试

        self._changeEnv(need_change=True)
        if not args.ablate_rgb or not args.ablate_depth:
            result = self.simulator_tool.setPoses(poses=poses)
            if not result:
                logger.error('重置到此位置失败')
                self.reset_to_this_pose(poses)

    def makeActions(self, action_list):
        # 根据反馈的action_list更新当前batch样本的pose
        poses = []
        for index, action in enumerate(action_list):
            if self.sim_states[index].is_end == True:
                action = AirsimActions.STOP
                # continue

            if action == AirsimActions.STOP or self.sim_states[index].step >= int(args.maxAction):
                self.sim_states[index].is_end = True

            state = self.sim_states[index]

            pose = copy.deepcopy(state.pose)
            new_pose = getPoseAfterMakeAction(pose, action)
            poses.append(new_pose)

        # 新pose对应仿真客户端下的格式
        poses_formatted = []
        cnt = 0
        for index_1, item in enumerate(self.machines_info):
            poses_formatted.append([])
            for index_2, _ in enumerate(item['open_scenes']):
                poses_formatted[index_1].append(poses[cnt])
                cnt += 1

        # 将pose信息反馈给i无人机失败后
        if not args.ablate_rgb or not args.ablate_depth:
            result = self.simulator_tool.setPoses(poses=poses_formatted)
            if not result:
                logger.error('设置位置失败')
                self.reset_to_this_pose(poses_formatted)

        #
        for index, action in enumerate(action_list):
            if self.sim_states[index].is_end == True:
                continue

            if action == AirsimActions.STOP or self.sim_states[index].step >= int(args.maxAction):
                self.sim_states[index].is_end = True

            self.sim_states[index].step += 1
            self.sim_states[index].pose = poses[index]
            self.sim_states[index].trajectory.append([
                poses[index].position.x_val, poses[index].position.y_val, poses[index].position.z_val, # xyz
                poses[index].orientation.x_val, poses[index].orientation.y_val, poses[index].orientation.z_val, poses[index].orientation.w_val, # xyzw
            ])
            self.sim_states[index].pre_action = action

        # 更新评价指标的值
        if args.run_type not in ['collect']:
            self.update_measurements()

    def update_measurements(self):
        # 后续所有的代码都是在更新评价指标的值
        self._update_DistanceToGoal()
        self._updata_Success()
        self._updata_NDTW()
        self._updata_SDTW()
        self._update_PathLength()
        self._update_OracleSuccess()
        self._update_StepsTaken()

    def _update_DistanceToGoal(self):
        for i, state in enumerate(self.sim_states):

            current_position = np.array([
                state.pose.position.x_val,
                state.pose.position.y_val,
                state.pose.position.z_val
            ])

            if self.sim_states[i].DistanceToGoal['_previous_position'] is None or \
                not np.allclose(self.sim_states[i].DistanceToGoal['_previous_position'], current_position, atol=1):
                distance_to_target = EuclideanDistance3(
                    np.array(current_position)[0:2],
                    np.array(state.episode_info['goals'][0]['position'])[0:2]
                )
                self.sim_states[i].DistanceToGoal['_previous_position'] = current_position
                self.sim_states[i].DistanceToGoal['_metric'] = distance_to_target

    def _updata_Success(self):
        for i, state in enumerate(self.sim_states):
            distance_to_target = self.sim_states[i].DistanceToGoal['_metric']
            if (
                self.sim_states[i].is_end
                and distance_to_target <= self.sim_states[i].SUCCESS_DISTANCE
            ):
                self.sim_states[i].Success['_metric'] = 1.0
            else:
                self.sim_states[i].Success['_metric'] = 0.0

    def _updata_NDTW(self):
        def euclidean_distance(
                position_a,
                position_b,
        ) -> float:
            return np.linalg.norm(
                np.array(position_b) - np.array(position_a), ord=2
            )

        for i, state in enumerate(self.sim_states):

            current_position = np.array([
                state.pose.position.x_val,
                state.pose.position.y_val,
                state.pose.position.z_val
            ])

            if len(state.NDTW['locations']) == 0:
                self.sim_states[i].NDTW['locations'].append(current_position)
            else:
                if current_position.tolist() == state.NDTW['locations'][-1].tolist():
                    continue
                self.sim_states[i].NDTW['locations'].append(current_position)

            dtw_distance = fastdtw(
                self.sim_states[i].NDTW['locations'], self.sim_states[i].NDTW['gt_locations'], dist=euclidean_distance
            )[0]

            nDTW = np.exp(
                -dtw_distance / (len(self.sim_states[i].NDTW['gt_locations']) * self.sim_states[i].SUCCESS_DISTANCE)
            )
            self.sim_states[i].NDTW['_metric'] = nDTW

    def _updata_SDTW(self):
        for i, state in enumerate(self.sim_states):
            ep_success = self.sim_states[i].Success['_metric']
            nDTW = self.sim_states[i].NDTW['_metric']
            self.sim_states[i].SDTW['_metric'] = ep_success * nDTW

    def _update_PathLength(self):
        for i, state in enumerate(self.sim_states):

            current_position = np.array([
                state.pose.position.x_val,
                state.pose.position.y_val,
                state.pose.position.z_val
            ])

            if state.PathLength['_previous_position'] is None:
                self.sim_states[i].PathLength['_previous_position'] = current_position

            self.sim_states[i].PathLength['_metric'] += EuclideanDistance3(
                current_position, self.sim_states[i].PathLength['_previous_position']
            )
            self.sim_states[i].PathLength['_previous_position'] = current_position

    def _update_OracleSuccess(self):
        for i, state in enumerate(self.sim_states):
            d = self.sim_states[i].DistanceToGoal['_metric']
            self.sim_states[i].OracleSuccess['_metric'] = float(
                self.sim_states[i].OracleSuccess['_metric'] or d <= self.sim_states[i].SUCCESS_DISTANCE
            )

    def _update_StepsTaken(self):
        for i, state in enumerate(self.sim_states):
            self.sim_states[i].StepsTaken['_metric'] = self.sim_states[i].step
