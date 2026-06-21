import os
import gc
import sys
import cv2
import copy
import time
import lmdb
import tqdm
import math
import random
import json
import numpy as np
import msgpack_numpy
from collections import defaultdict
from pathlib import Path
import torch
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from typing import List, Optional, DefaultDict
from PIL import Image, ImageFont, ImageDraw
from gym import spaces
from airsim_plugin.airsim_settings import AirsimActions

from utils.logger import logger
from utils.utils import get_rank, is_dist_avail_and_initialized, is_main_process, init_distributed_mode
from Model.il_trainer import VLNCETrainer
from Model.utils.common import get_checkpoint_id
from Model.utils.tensor_dict import DictTree, TensorDict
from Model.aux_losses import AuxLosses
from Model.utils.tensorboard_utils import TensorboardWriter
from Model.utils.common import observations_to_image, append_text_to_image, generate_video
from src.common.param import args
from src.vlnce_src.env import AirVLNENV

# 将标准输出和标准错误设置为无缓冲模式
sys.stdout.reconfigure(write_through=True)
sys.stderr.reconfigure(write_through=True)

def setup():
    init_distributed_mode()

    seed = 100 + get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = False

# [train] 将收集的lmdb数据构造为Dataset
# 并行：DDPIWTrajectoryDataset
# 串行：IWTrajectoryDataset
def _block_shuffle(lst, block_size):
    blocks = [lst[i: i + block_size] for i in range(0, len(lst), block_size)]
    random.shuffle(blocks)
    return [ele for block in blocks for ele in block]

class DDPIWTrajectoryDataset(torch.utils.data.IterableDataset):
    def __init__(self, lmdb_features_dir, use_iw=True, inflection_weight_coef=1.0, batch_size=1):
        super().__init__()
        self.keys = []
        self._preload = []
        self.batch_size = batch_size
        self.preload_size = batch_size * 100
        self.lmdb_features_dir = lmdb_features_dir

        if use_iw:
            self.inflec_weights = torch.tensor([1.0, inflection_weight_coef])
        else:
            self.inflec_weights = torch.tensor([1.0, 1.0])

        with lmdb.open(
                self.lmdb_features_dir,
                readonly=True,
                lock=False,
                readahead=False,
        ) as lmdb_env, tqdm.tqdm(
            total=int(lmdb_env.stat()["entries"]), dynamic_ncols=True
        ) as pbar, lmdb_env.begin() as txn:
            for key in txn.cursor().iternext(keys=True, values=False):
                pbar.update()
                self.keys.append(key.decode())

        self.length = len(self.keys)
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        self.start = 0
        self.end = self.length

        self.per_worker = int(math.floor((self.end - self.start) / float(self.world_size)))
        self.iter_start = 0 + self.rank * self.per_worker
        self.iter_end = min(self.iter_start + self.per_worker, self.end)
        logger.warning(
            "END init DDP-Dataset \t rank: {} \t start({}) - end({})".format(self.rank, self.iter_start, self.iter_end))

    def _load_next(self):
        if len(self._preload) == 0:
            if len(self.load_ordering) == 0:
                raise StopIteration

            new_preload = []
            lengths = []
            with lmdb.open(self.lmdb_features_dir, readonly=True, lock=False,
            ) as lmdb_env, lmdb_env.begin(buffers=True) as txn:
                for i in range(self.preload_size):
                    if len(self.load_ordering) == 0:
                        break

                    if (i + 1) % 10 == 0:
                        logger.warning("rank: {} \t lmdb load: {} / {}".format(self.rank, i + 1, self.preload_size))

                    new_preload.append(
                        msgpack_numpy.unpackb(
                            txn.get(str(self.keys[self.load_ordering.pop()]).encode()),
                            raw=False,
                        )
                    )

                    lengths.append(len(new_preload[-1][0]))

            sort_priority = list(range(len(lengths)))
            random.shuffle(sort_priority)

            sorted_ordering = list(range(len(lengths)))
            sorted_ordering.sort(key=lambda k: (lengths[k], sort_priority[k]))

            for idx in _block_shuffle(sorted_ordering, self.batch_size):
                self._preload.append(new_preload[idx])

            del new_preload, lengths

        return self._preload.pop()

    def __next__(self):
        obs_without_instruction, prev_actions, oracle_actions, instruction_embeddings = self._load_next()

        for k, v in obs_without_instruction.items():
            obs_without_instruction[k] = torch.from_numpy(np.copy(v))

        for k, v in instruction_embeddings.items():
            instruction_embeddings[k] = torch.from_numpy(np.copy(v))

        trajectory_length = obs_without_instruction['progress'].shape[0]
        for key, embedding_tensor in instruction_embeddings.items():
            expanded_dims = [-1] * embedding_tensor.ndim
            instruction_embeddings[key] = embedding_tensor.unsqueeze(0).expand(trajectory_length, *expanded_dims)

        obs = obs_without_instruction
        obs.update(instruction_embeddings)

        prev_actions = torch.from_numpy(np.copy(prev_actions))
        oracle_actions = torch.from_numpy(np.copy(oracle_actions))

        inflections = torch.cat(
            [
                torch.tensor([1], dtype=torch.long),
                (oracle_actions[1:] != oracle_actions[:-1]).long(),
            ]
        )

        return (
            obs,
            prev_actions,
            oracle_actions,
            self.inflec_weights[inflections],
        )

    def __iter__(self):
        self.load_ordering = list(
            reversed(
                _block_shuffle(list(range(self.iter_start, self.iter_end)), self.preload_size)
            )
        )

        return self

class IWTrajectoryDataset(torch.utils.data.IterableDataset):
    def __init__(self, lmdb_features_dir, use_iw=True, inflection_weight_coef=1.0, batch_size=1):
        super().__init__()
        self.lmdb_features_dir = lmdb_features_dir
        self.preload_size = batch_size * 100
        self.keys = []
        self._preload = []
        self.batch_size = batch_size

        if use_iw:
            self.inflec_weights = torch.tensor([1.0, inflection_weight_coef])
        else:
            self.inflec_weights = torch.tensor([1.0, 1.0])

        with lmdb.open(
                self.lmdb_features_dir,
                readonly=True,
                lock=False,
                readahead=False,
        ) as lmdb_env, tqdm.tqdm(
            total=int(lmdb_env.stat()["entries"]), dynamic_ncols=True
        ) as pbar, lmdb_env.begin() as txn:
            for key in txn.cursor().iternext(keys=True, values=False):
                pbar.update()
                self.keys.append(key.decode())

        self.length = len(self.keys)

        self.iter_start = 0
        self.iter_end = self.length
        logger.warning("END init Dataset \t start({}) - end({})".format(self.iter_start, self.iter_end))

    def _load_next(self):
        if len(self._preload) == 0:
            if len(self.load_ordering) == 0:
                raise StopIteration

            new_preload = []
            lengths = []
            with lmdb.open(self.lmdb_features_dir, readonly=True, lock=False,
            ) as lmdb_env, lmdb_env.begin(buffers=True) as txn:
                for i in range(self.preload_size):
                    if len(self.load_ordering) == 0:
                        break

                    if (i + 1) % 10 == 0:
                        if self.worker_info is not None:
                            logger.info("{} lmdb load: {} / {}".format(self.worker_info.id, i + 1, self.preload_size))
                        else:
                            logger.info("{} lmdb load: {} / {}".format(0, i + 1, self.preload_size))

                    new_preload.append(
                        msgpack_numpy.unpackb(
                            txn.get(str(self.keys[self.load_ordering.pop()]).encode()),
                            raw=False,
                        )
                    )

                    lengths.append(len(new_preload[-1][0]))

            sort_priority = list(range(len(lengths)))
            random.shuffle(sort_priority)

            sorted_ordering = list(range(len(lengths)))
            sorted_ordering.sort(key=lambda k: (lengths[k], sort_priority[k]))

            for idx in _block_shuffle(sorted_ordering, self.batch_size):
                self._preload.append(new_preload[idx])

            del new_preload, lengths

        return self._preload.pop()

    def __next__(self):
        obs_without_instruction, prev_actions, oracle_actions, instruction_embeddings = self._load_next()

        for k, v in obs_without_instruction.items():
            obs_without_instruction[k] = torch.from_numpy(np.copy(v))

        for k, v in instruction_embeddings.items():
            instruction_embeddings[k] = torch.from_numpy(np.copy(v))

        trajectory_length = obs_without_instruction['progress'].shape[0]
        for key, embedding_tensor in instruction_embeddings.items():
            expanded_dims = [-1] * embedding_tensor.ndim
            instruction_embeddings[key] = embedding_tensor.unsqueeze(0).expand(trajectory_length, *expanded_dims)

        obs = obs_without_instruction
        obs.update(instruction_embeddings)

        prev_actions = torch.from_numpy(np.copy(prev_actions))
        oracle_actions = torch.from_numpy(np.copy(oracle_actions))

        inflections = torch.cat(
            [
                torch.tensor([1], dtype=torch.long),
                (oracle_actions[1:] != oracle_actions[:-1]).long(),
            ]
        )

        return (
            obs,
            prev_actions,
            oracle_actions,
            self.inflec_weights[inflections],
        )

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        self.worker_info = worker_info
        if worker_info is None:
            start = 0
            end = self.length
        else:
            per_worker = int(np.ceil(self.length / worker_info.num_workers))

            start = per_worker * worker_info.id
            end = min(start + per_worker, self.length)

        self.load_ordering = list(
            reversed(
                _block_shuffle(list(range(start, end)), self.preload_size)
            )
        )

        return self

# [train] 从batch中提取各个观测量
# ObservationsDict: 对于字典类型的数据定义pin_memory方法用于将数据从cpu加载到Gpu显存中
# collate_fn: 从batch中提取各个观测
class ObservationsDict(dict):
    def pin_memory(self):
        for k, v in self.items():
            self[k] = v.pin_memory()
        return self

def collate_fn(batch):
    """Each sample in batch: (obs, prev_actions, oracle_actions, inflec_weight)"""
    def _pad_helper(t, max_len, fill_val=0):
        pad_amount = max_len - t.size(0)
        if pad_amount == 0:
            return t
        pad = torch.full_like(t[0:1], fill_val).expand(
            pad_amount, *t.size()[1:]
        )
        return torch.cat([t, pad], dim=0)

    transposed = list(zip(*batch))

    observations_batch = list(transposed[0])
    prev_actions_batch = list(transposed[1])
    corrected_actions_batch = list(transposed[2])
    weights_batch = list(transposed[3])
    B = len(prev_actions_batch) # batchSize大小

    new_observations_batch = defaultdict(list)
    for sensor in observations_batch[0]:
        for bid in range(B):
            new_observations_batch[sensor].append(
                observations_batch[bid][sensor]
            )

    observations_batch = new_observations_batch

    # max_traj_len = max(ele.size(0) for ele in prev_actions_batch)
    max_traj_len = 500
    for bid in range(B):
        for sensor in observations_batch:
            observations_batch[sensor][bid] = _pad_helper(
                observations_batch[sensor][bid][:max_traj_len, ...], max_traj_len, fill_val=1.0
            )

        prev_actions_batch[bid] = _pad_helper(
            prev_actions_batch[bid][:max_traj_len, ...], max_traj_len
        )
        corrected_actions_batch[bid] = _pad_helper(
            corrected_actions_batch[bid][:max_traj_len, ...], max_traj_len
        )
        weights_batch[bid] = _pad_helper(weights_batch[bid][:max_traj_len, ...], max_traj_len)

    for sensor in observations_batch:
        observations_batch[sensor] = torch.stack(observations_batch[sensor], dim=1)
        observations_batch[sensor] = observations_batch[sensor].view(-1, *observations_batch[sensor].size()[2:])

    prev_actions_batch = torch.stack(prev_actions_batch, dim=1)
    corrected_actions_batch = torch.stack(corrected_actions_batch, dim=1)
    weights_batch = torch.stack(weights_batch, dim=1)
    not_done_masks = torch.ones_like(corrected_actions_batch, dtype=torch.uint8)
    not_done_masks[0] = 0

    observations_batch = ObservationsDict(observations_batch)

    return (
        observations_batch,
        prev_actions_batch.view(-1, 1),
        not_done_masks.view(-1, 1),
        corrected_actions_batch,
        weights_batch
    )


# collect以及eval阶段时从batch中提取各个观测量
@torch.no_grad()
def batch_obs(observations: List[DictTree], device: Optional[torch.device] = None) -> TensorDict:
    r"""Transpose a batch of observation dicts to a dict of batched observations.

    Args:
        observations:  list of dicts of observations.
        device: The torch.device to put the resulting tensors on.

    Returns:
        dict of observations(torch.Tensor)
    """
    batch: DefaultDict[str, List] = defaultdict(list)

    for obs in observations:
        for sensor, value in obs.items():
            if sensor == 'instruction':
                batch['subtask_embedding'].append(value['subtask_embedding'])
                batch['nA_embedding'].append(value['nA_embedding'])
                batch['nL_embedding'].append(value['nL_embedding'])
                batch['Ae_embedding'].append(value['Ae_embedding'])
                batch['Le_embedding'].append(value['Le_embedding'])
            else:
                batch[sensor].append(torch.as_tensor(value))

    batch_t: TensorDict = TensorDict()

    for sensor in batch:
        batch_t[sensor] = torch.stack(batch[sensor], dim=0)

    return batch_t.map(lambda v: v.to(device))


def initialize_trainer():
    # 无人机观测空间
    # VLN模型视觉编码器的输入参数
    observation_space = spaces.Dict({
        "rgb": spaces.Box(low=0, high=255, shape=(512, 512, 3), dtype=np.uint8),
        "depth": spaces.Box(low=0, high=1, shape=(256, 256, 1), dtype=np.float32),
        "instruction": spaces.Discrete(0),
        "progress": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
        "teacher_action": spaces.Box(low=0, high=100, shape=(1,)),
    })
    # 无人机的行动空间
    action_space = spaces.Discrete(int(len(AirsimActions)))

    # VLN模型
    trainer = VLNCETrainer(load_from_ckpt=False, observation_space=observation_space, action_space=action_space)

    logger.info('initialize_trainer over')
    return trainer


def collect_data(data_it=0):
    logger.info(args)

    train_env = AirVLNENV(batch_size=args.batchSize, split='train')

    trainer = initialize_trainer()

    if torch.cuda.is_available():
        with torch.cuda.device(trainer.device):
            torch.cuda.empty_cache()

    # 定义追踪rgb和depth图像特征的钩子
    def hook_builder(tgt_tensor):
        def hook(m, i, o):
            tgt_tensor.set_(o.cpu())
        return hook

    rgb_features = torch.zeros((1,), device="cpu")
    if not args.ablate_rgb:
        rgb_hook = trainer.policy.net.rgb_encoder.layer_extract.register_forward_hook(hook_builder(rgb_features))
    else:
        rgb_hook = None

    depth_features = torch.zeros((1,), device="cpu")
    if not args.ablate_depth:
        depth_hook = trainer.policy.net.depth_encoder.visual_encoder.register_forward_hook(hook_builder(depth_features))
    else:
        depth_hook = None

    beta = 1.0
    with torch.no_grad():
        pbar = None
        pbar_pre_index = 0  # 当前batch的起始index
        end_iter = len(train_env.data)  # 最后一个样本的index

        while train_env.index_data < end_iter:
            if pbar_pre_index + train_env.batch_size >= end_iter:
                break

            pbar_pre_index = train_env.index_data
            train_env.next_minibatch()  # 获取导航样本train_env.batch
            if train_env.batch is None:
                logger.warning('train_env.batch is None, going to break and stop collect')
                break
            if pbar is None:
                pbar = tqdm.tqdm(total=end_iter)
                pbar.update(train_env.index_data)
            else:
                pbar.update(n=train_env.index_data - pbar_pre_index)

            # 关键变量初始化
            if args.policy_type in ['seq2seq', 'cma']:
                rnn_states = torch.zeros(
                    train_env.batch_size,
                    trainer.policy.net.num_recurrent_layers,
                    trainer.policy.net.state_encoder.hidden_size,
                    device=trainer.device,
                )
                prev_actions = torch.zeros(
                    train_env.batch_size,
                    1,
                    dtype=torch.long,
                    device=trainer.device,
                )
                not_done_masks = torch.zeros(
                    train_env.batch_size,
                    1,
                    dtype=torch.uint8,
                    device=trainer.device,
                )
            else:
                raise NotImplementedError

            # 收集数据所用变量
            episodes = [[] for _ in range(train_env.batch_size)]
            dones = [False for _ in range(train_env.batch_size)]
            skips = [False for _ in range(train_env.batch_size)]
            envs_to_pause = []

            # 初始化导航环境并获取初始状态
            outputs = train_env.reset()
            observations, _, _, _ = [list(x) for x in zip(*outputs)]
            batch = batch_obs(observations, trainer.device)
            ended = False  # 结束收集的总标志

            # 遍历batch中所有样本args.maxAction次，迭代次数少的样本会提前结束
            for t in range(int(args.maxAction) + 1):
                logger.info('{} - {} / {}'.format(int(train_env.index_data) - int(train_env.batch_size), t, end_iter))

                # episodes的数据收集完毕后保存为lmdb
                # dones:False skips:False (保存前，收集数据) -> dones:True skips:False (开始保存！！！) -> dones:True skips:True (保存跳过)
                for i in range(train_env.batch_size):
                    if dones[i] and not skips[i]:
                        _episodes = episodes[i].copy()
                        current_info = copy.deepcopy(infos[i])

                        for step_data in _episodes:
                            if 'instruction' in step_data[0]:
                                del step_data[0]['instruction']

                        traj_obs = batch_obs([step[0] for step in _episodes], device=torch.device("cpu"))
                        del traj_obs['teacher_action']
                        for k, v in traj_obs.items():
                            traj_obs[k] = v.numpy()

                        for _i, _j in enumerate(train_env.trajectory_id_2_instruction_tokens[current_info['trajectory_id']]):
                            instruction_embeddings = {
                                key: tensor.numpy()
                                for key, tensor in _j.items()
                            }

                            transposed_ep = [
                                traj_obs,
                                np.array([step[1] for step in _episodes], dtype=np.int64),
                                np.array([step[2] for step in _episodes], dtype=np.int64),
                                instruction_embeddings,
                            ]

                            # 保存收集到的数据transposed_ep到lmdb数据库
                            train_env.threading_lock_lmdb_features_txn.acquire()
                            lmdb_key = str(train_env.trajectory_id_2_episode_ids[current_info['trajectory_id']][_i])
                            train_env.lmdb_features_txn.put(
                                lmdb_key.encode(),
                                msgpack_numpy.packb(transposed_ep, use_bin_type=True)
                            )
                            train_env.lmdb_features_txn.commit()
                            train_env.lmdb_features_start_id = train_env.lmdb_features_env.stat()["entries"]
                            train_env.lmdb_features_txn = train_env.lmdb_features_env.begin(write=True)
                            train_env.threading_lock_lmdb_features_txn.release()
                            logger.info('lmdb of {}, lmdb_start_id: {}'.format(train_env.split, train_env.lmdb_features_start_id))

                        episodes[i] = []
                        _episodes = []
                        envs_to_pause.append(i)
                        skips[i] = True
                    if np.array(dones).all():
                        ended = True

                if ended:
                    break

                # 收集过程：
                # batch内每个样本: t(新位置，新状态) -> 将图像替换为图像特征 -> 保存t时刻episodes -> 执行真值action -> t+1(新位置，新状态)
                # batch内每个样本返回dones以后将episodes的信息在第一个for循环中保存为lmdb
                # batch内所有样本dones以后ended变为true结束当前batch

                # --- 正常collect模式 ---
                actions, rnn_states, _ = trainer.policy.act(
                    batch,
                    rnn_states,
                    prev_actions,
                    not_done_masks,
                    deterministic=False,
                )
                # 实际collect的过程都是选择了teacher_action
                actions = torch.where(
                    torch.rand_like(actions, dtype=torch.float) < beta,
                    batch['teacher_action'].long(),
                    actions,
                )

                for i in range(train_env.batch_size):
                    if not args.ablate_rgb and rgb_features is not None:
                        observations[i]["rgb_features"] = rgb_features[i]
                        del observations[i]["rgb"]

                    if not args.ablate_depth and depth_features is not None:
                        observations[i]["depth_features"] = depth_features[i]
                        del observations[i]["depth"]

                    if i in envs_to_pause:
                        continue

                    episodes[i].append(
                        (
                            observations[i],
                            prev_actions[i].item(),
                            batch['teacher_action'][i].item(),
                        )
                    )

                # 执行真值的行动决策actions，并作为下一时刻的prev_actions
                prev_actions.copy_(actions)
                actions = [temp[0] for temp in actions.cpu().numpy()]
                train_env.makeActions(actions)

                # 无人机的新状态
                outputs = train_env.get_obs()
                observations, _, dones, infos = [list(x) for x in zip(*outputs)]
                batch = batch_obs(observations, trainer.device)

                # # -----------收集高分辨率的图像 ------------
                # # 关键参数：每个场景采样15个导航样本，每个样本间隔Gap帧采样RGBD图像对
                # # 保存位置及格式：'/home/code/AirVLN02/RGBD-CL/sceneID_episode_id_RGB/Depth.jpg'
                # Gap = 50
                # if t % Gap == 0:
                #     RGBFilename = "{}_{}_{}_RGB".format(infos[0]['scene_id'], infos[0]['episode_id'], str(t))
                #     folder_path = "/home/code/AirVLN03/high_image/SampleGap{}".format(Gap)
                #     os.makedirs(folder_path, exist_ok=True)
                #
                #     rgb_image = observations[0]["rgb"]
                #     draw_image(RGBFilename, rgb_image, folder_path)
                # # --------------------------------------------------------

                logger.info('action: {}'.format(actions))
                not_done_masks = torch.tensor(
                    [[0] if done else [1] for done in dones],
                    dtype=torch.uint8,
                    device=trainer.device,
                )

            # 冗余代码，保险手段
            # 此时dones[i]必定为True，因此考虑的是如果真的有样本的step超过了500时保存溢出的数据
            for i in range(train_env.batch_size):
                if dones[i] and not t >= int(args.maxAction):
                    continue

                if args.collect_type in ['TF']:
                    _episodes = episodes[i].copy()
                    current_info = copy.deepcopy(infos[i])

                    for step_data in _episodes:
                        if 'instruction' in step_data[0]:
                            del step_data[0]['instruction']

                    traj_obs = batch_obs([step[0] for step in _episodes], device=torch.device("cpu"))
                    del traj_obs['teacher_action']
                    for k, v in traj_obs.items():
                        traj_obs[k] = v.numpy()

                    for _i, _j in enumerate(train_env.trajectory_id_2_instruction_tokens[current_info['trajectory_id']]):
                        instruction_embeddings = {
                            key: tensor.numpy()
                            for key, tensor in _j.items()
                        }

                        transposed_ep = [
                            traj_obs,
                            np.array([step[1] for step in _episodes], dtype=np.int64),
                            np.array([step[2] for step in _episodes], dtype=np.int64),
                            instruction_embeddings,
                        ]

                        train_env.threading_lock_lmdb_features_txn.acquire()
                        lmdb_key = str(train_env.trajectory_id_2_episode_ids[current_info['trajectory_id']][_i])
                        train_env.lmdb_features_txn.put(
                            lmdb_key.encode(),
                            msgpack_numpy.packb(transposed_ep, use_bin_type=True)
                        )
                        train_env.lmdb_features_txn.commit()
                        train_env.lmdb_features_start_id = train_env.lmdb_features_env.stat()["entries"]
                        train_env.lmdb_features_txn = train_env.lmdb_features_env.begin(write=True)
                        train_env.threading_lock_lmdb_features_txn.release()
                        logger.info(
                            'lmdb of {}, lmdb_start_id: {}'.format(train_env.split, train_env.lmdb_features_start_id))

                    episodes[i] = []
                    _episodes = []
                    envs_to_pause.append(i)
                    skips[i] = True

                else:
                    ep = episodes[i]
                    if len(ep) <= 0:
                        continue
                    traj_obs = batch_obs(
                        [step[0] for step in ep],
                        device=torch.device("cpu"),
                    )
                    del traj_obs['teacher_action']
                    for k, v in traj_obs.items():
                        traj_obs[k] = v.numpy()

                    transposed_ep = [
                        traj_obs,
                        np.array([step[1] for step in ep], dtype=np.int64),
                        np.array([step[2] for step in ep], dtype=np.int64),
                    ]

                    train_env.threading_lock_lmdb_features_txn.acquire()
                    lmdb_key = str(infos[i]['episode_id'])
                    train_env.lmdb_features_txn.put(
                        lmdb_key.encode(),
                        msgpack_numpy.packb(
                            transposed_ep, use_bin_type=True
                        ),
                    )
                    train_env.lmdb_features_txn.commit()
                    train_env.lmdb_features_start_id = train_env.lmdb_features_env.stat()["entries"]
                    train_env.lmdb_features_txn = train_env.lmdb_features_env.begin(write=True)
                    train_env.lmdb_collected_keys.add(lmdb_key)
                    train_env.threading_lock_lmdb_features_txn.release()
                    logger.info('lmdb of {}, lmdb_start_id: {}'.format(train_env.split, train_env.lmdb_features_start_id))

                    episodes[i] = []
                    envs_to_pause.append(i)
                    skips[i] = True

            gc.collect()

    try:
        pbar.close()
    except:
        pass

    if rgb_hook is not None:
        rgb_hook.remove()
    if depth_hook is not None:
        depth_hook.remove()

    try:
        train_env.simulator_tool.closeScenes()
    except:
        pass
    logger.info('END data_it: {}'.format(data_it))


def train_vlnce():
    logger.info(args)  # 记录配置

    # 初始化 Tensorboard 日志记录器
    if get_rank() == 0:
        log_dir = Path(args.project_prefix) / f"DATA/output/{args.name}/train/TensorBoard/{args.make_dir_time}"
        writer = SummaryWriter(log_dir=str(log_dir))
    else:
        writer = None

    trainer = initialize_trainer()

    for dagger_it in range(int(args.dagger_it)):
        step_id = 0

        # 清理内存和显存
        if torch.cuda.is_available():
            with torch.cuda.device(trainer.device):
                torch.cuda.empty_cache()
        gc.collect()

        # 加载训练使用的lmdb
        lmdb_features_dir = str(Path(args.project_prefix) / 'DATA' / 'img_features' / 'collect' / str(args.name) / 'train')
        assert os.path.exists(str(lmdb_features_dir))

        if args.DistributedDataParallel:
            dataset = DDPIWTrajectoryDataset(
                lmdb_features_dir,
                use_iw=True,
                inflection_weight_coef=float(args.inflection_weight_coef),
                batch_size=args.batchSize,
            )
        else:
            dataset = IWTrajectoryDataset(
                lmdb_features_dir,
                use_iw=True,
                inflection_weight_coef=float(args.inflection_weight_coef),
                batch_size=args.batchSize,
            )

        diter = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batchSize,
            shuffle=False,
            collate_fn=collate_fn,
            pin_memory=False,
            drop_last=True,
            num_workers=2,
        )

        AuxLosses.activate()
        for epoch in tqdm.trange(int(args.epochs), dynamic_ncols=True):
            batch_cnt = 0
            total = (dataset.length // dataset.batch_size
                    if not args.DistributedDataParallel
                    else (dataset.iter_end - dataset.iter_start) // dataset.batch_size)

            for batch in tqdm.tqdm(diter, total=total, leave=False, dynamic_ncols=True):
                (
                    observations_batch,
                    prev_actions_batch,
                    not_done_masks,
                    corrected_actions_batch,
                    weights_batch,
                ) = batch

                observations_batch = {
                    k: v.to(
                        device=trainer.device,
                        dtype=torch.float32,
                        non_blocking=True,
                    )
                    for k, v in observations_batch.items()
                }

                loss, action_loss, aux_loss = trainer._update_agent(
                    observations_batch,
                    prev_actions_batch.to(device=trainer.device, non_blocking=True),
                    not_done_masks.to(device=trainer.device, non_blocking=True),
                    corrected_actions_batch.to(device=trainer.device, non_blocking=True),
                    weights_batch.to(device=trainer.device, non_blocking=True),
                )

                if get_rank() == 0:
                    logger.warning(
                        'dagger_it: {} / {} \t epoch: {} / {} \t batch: {} / {}'.format(
                            dagger_it, args.dagger_it,
                            epoch, args.epochs,
                            batch_cnt, dataset.length // dataset.batch_size)
                    )

                    logger.info(f"train_loss: {loss}")
                    logger.info(f"train_action_loss: {action_loss}")
                    logger.info(f"train_aux_loss: {aux_loss}")
                    logger.info(f"Batches processed: {step_id}.")
                    logger.info(f"On DAgger iter {dagger_it}, Epoch {epoch}.")
                    logger.info('\n')

                    writer.add_scalar(f"train_loss_iter_{dagger_it}", loss, step_id)
                    writer.add_scalar(f"train_action_loss_iter_{dagger_it}", action_loss, step_id)
                    writer.add_scalar(f"train_aux_loss_iter_{dagger_it}", aux_loss, step_id)

                step_id += 1
                batch_cnt += 1

            if is_main_process():
                if ((dagger_it * args.epochs + epoch) + 1) % 5 == 0:
                    trainer.save_checkpoint(f"ckpt.{dagger_it * args.epochs + epoch}.pth", dagger_it, epoch)

            if is_dist_avail_and_initialized() == 1:
                dist.barrier()

        if is_main_process():
            trainer.save_checkpoint(f"ckpt.LAST.pth", dagger_it, epoch)
        AuxLosses.deactivate()


def eval_vlnce():
    logger.info(args)

    writer = TensorboardWriter(
        str(Path(args.project_prefix) / 'DATA/output/{}/eval/TensorBoard/{}'.format(args.name, args.make_dir_time)),
        flush_secs=30,
    )

    assert os.path.exists(args.EVAL_CKPT_PATH_DIR), '评估文件(夹)不存在'

    proposed_index = get_checkpoint_id(args.EVAL_CKPT_PATH_DIR)
    if proposed_index is not None:
        ckpt_idx = proposed_index
    else:
        ckpt_idx = 100000

    _eval_checkpoint(checkpoint_path=args.EVAL_CKPT_PATH_DIR, writer=writer, checkpoint_index=ckpt_idx)
    logger.info("END evaluate")

    if writer is not None:
        try:
            writer.writer.close()
            del writer
        except Exception as e:
            logger.error(e)
    logger.info("END evaluate")


def _eval_checkpoint(checkpoint_path: str, writer, checkpoint_index: int = 0, ) -> None:
    logger.info(f"checkpoint_path: {checkpoint_path}")

    # 加载要验证的数据集，同时也可以设置要验证的样本数目
    if args.EVAL_DATASET == 'train':
        train_env = AirVLNENV(batch_size=args.batchSize, split='train')
    elif args.EVAL_DATASET == 'val_seen':
        train_env = AirVLNENV(batch_size=args.batchSize, split='val_seen')
    elif args.EVAL_DATASET == 'val_unseen':
        train_env = AirVLNENV(batch_size=args.batchSize, split='val_unseen')
    elif args.EVAL_DATASET == 'test':
        train_env = AirVLNENV(batch_size=args.batchSize, split='test')
    else:
        raise KeyError

    EVAL_RESULTS_DIR = Path(args.project_prefix) / 'DATA/output/{}/eval/results/{}'.format(args.name, args.make_dir_time)
    fname = os.path.join(EVAL_RESULTS_DIR, f"stats_ckpt_{checkpoint_index}_{train_env.split}.json")
    if os.path.exists(fname):
        print("skipping -- evaluation exists.")
        return

    trainer = VLNCETrainer(
        load_from_ckpt=True,
        observation_space=train_env.observation_space,
        action_space=train_env.action_space,
        ckpt_path=checkpoint_path,
    )
    trainer.policy.eval()

    # 初始化 Slot 可视化器
    if torch.cuda.is_available():
        with torch.cuda.device(trainer.device):
            torch.cuda.empty_cache()
    gc.collect()

    # 开始验证每一个导航样本
    stats_episodes = {}
    episodes_to_eval = len(train_env.data)
    pbar = tqdm.tqdm(total=episodes_to_eval, dynamic_ncols=True)

    # ---------------- 新增代码 ----------------
    # 初始化一个列表，用于存储所有episodes的动作生成准确率
    # 具体分析的话，其实对比真值action和实际action之间的准确率没有什么意义，因为在发生一步错步步错的情况以后，哪怕后面action一致了，
    # 但是这也不代表有用，更合理的指标是考虑到路径的相似度。例如我们的sdtw，但是这个指标从定义上和基金的第一章验收需求很相似
    action_accuracies = []
    
    # 模型效率统计
    inference_latencies = []  # 推理延迟列表
    peak_gpu_memory = 0  # 峰值 GPU 内存
    # ----------------

    with torch.no_grad():
        start_iter = 0
        end_iter = len(train_env.data)
        cnt = 0
        for idx in range(start_iter, end_iter, train_env.batch_size):
            if args.EVAL_NUM != -1 and cnt * train_env.batch_size >= args.EVAL_NUM:
                break
            cnt += 1

            train_env.next_minibatch()
            if train_env.batch is None:
                logger.warning('train_env.batch is None, going to break and stop collect')
                break

            # ## ---------- 针对scene_list的样例分析 ----------
            # 定位具体的某个导航样本
            # if train_env.batch[0]['episode_id'] not in scene_list:
            #     continue

            if args.policy_type in ['seq2seq', 'cma']:
                rnn_states = torch.zeros(
                    train_env.batch_size,
                    trainer.policy.net.num_recurrent_layers,
                    trainer.policy.net.state_encoder.hidden_size,
                    device=trainer.device,
                )
                prev_actions = torch.zeros(
                    train_env.batch_size,
                    1,
                    dtype=torch.long,
                    device=trainer.device,
                )
                not_done_masks = torch.zeros(
                    train_env.batch_size,
                    1,
                    dtype=torch.uint8,
                    device=trainer.device,
                )
            else:
                raise NotImplementedError

            vis_frames = [[] for _ in range(train_env.batch_size)]
            rgb_images = [[] for _ in range(train_env.batch_size)]
            filenames = [[] for _ in range(train_env.batch_size)]

            episodes = [[] for _ in range(train_env.batch_size)]
            skips = [False for _ in range(train_env.batch_size)]
            dones = [False for _ in range(train_env.batch_size)]
            envs_to_pause = []

            # --- 新增代码 ---
            # 初始化用于累积当前batch每个episode正确动作数和总步数的列表
            correct_action_counts = [0] * train_env.batch_size
            total_steps_counts = [0] * train_env.batch_size
            # ----------------

            outputs = train_env.reset()
            observations, _, _, _ = [list(x) for x in zip(*outputs)]
            batch = batch_obs(observations, trainer.device)

            # policy输出导航行为，并将最后的状态保存在infos[i]
            for t in range(int(args.maxAction)):
                logger.info('checkpoint_index:{} \t {} - {} / {} \t {}'.format(checkpoint_index, idx, t, end_iter,
                                                                               not_done_masks.cpu().numpy().reshape(
                                                                                   (-1,)).tolist()))
                
                # --- 测量推理延迟和 GPU 内存 ---
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    start_time = time.time()
                
                actions, rnn_states, subtask_index = trainer.policy.act(
                    batch,
                    rnn_states,
                    prev_actions,
                    not_done_masks,
                    deterministic=True,
                    step=t,
                )
                
                # 记录推理延迟和 GPU 内存
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    inference_time = time.time() - start_time
                    inference_latencies.append(inference_time)
                    
                    current_gpu_memory = torch.cuda.max_memory_allocated(trainer.device) / (1024 ** 3)  # GB
                    peak_gpu_memory = max(peak_gpu_memory, current_gpu_memory)
                
                prev_actions.copy_(actions)
                # 2. 获取真值动作 (Teacher Action)
                #    它存储在 `batch` 字典中
                teacher_actions_tensor = batch['teacher_action'].long()
                correct_predictions = (actions.squeeze(1) == teacher_actions_tensor)
                for i in range(train_env.batch_size):
                    # 只在 episode 尚未结束时进行计数
                    if not dones[i]:
                        if correct_predictions[i].item():
                            correct_action_counts[i] += 1
                        total_steps_counts[i] += 1
                # ----------------------------------------

                # Make action and get the new state
                actions = [temp[0] for temp in actions.cpu().numpy()]
                train_env.makeActions(actions)

                outputs = train_env.get_obs()
                observations, _, dones, infos = [list(x) for x in zip(*outputs)]
                batch = batch_obs(observations, trainer.device)

                logger.info('action: {}'.format(actions))

                not_done_masks = torch.tensor(
                    [[0] if done else [1] for done in dones],
                    dtype=torch.uint8,
                    device=trainer.device,
                )

                # 保存可视化的结果到vis_frames，最后保存为视频
                for i in range(train_env.batch_size):
                    if args.EVAL_GENERATE_VIDEO:
                        frame = observations_to_image(observations[i], infos[i])
                        frame = append_text_to_image(frame, train_env.batch[i]['instruction']['instruction_text'])
                        vis_frames[i].append(frame)

                        rgb_images[i].append(observations[i]['rgb'])
                        action_str = action_dict[actions[i].item()]
                        subtask_str = str(subtask_index[i].item())
                        filenames[i].append("{}_{}_{}".format(str(t), action_str, subtask_str))

                    if not dones[i] or skips[i]:
                        continue

                    skips[i] = True
                    pbar.update()

                if np.array(dones).all():
                    break

            # --- 新增代码 ---
            # 计算并保存当前batch所有episodes的动作准确率
            for i in range(train_env.batch_size):
                if total_steps_counts[i] > 0:
                    accuracy = correct_action_counts[i] / total_steps_counts[i]
                    action_accuracies.append(accuracy)
                else:
                    # 如果一个episode在第一步就结束了，可以记为100%或0%，或者直接忽略
                    # 这里我们选择忽略，因为它没有做出任何有效决策
                    pass
            # ----------------

            # 保存当前当前样本的指标数据infos[t]
            for t in range(int(train_env.batch_size)):
                stats_episodes[str(train_env.batch[t]['episode_id'])] = infos[t]

                EVAL_SAVE_EVERY_RESULTS_DIR = Path(
                    args.project_prefix) / 'DATA/output/{}/eval/intermediate_results_every/{}'.format(args.name,
                                                                                                      args.make_dir_time)
                if not os.path.exists(str(EVAL_SAVE_EVERY_RESULTS_DIR / str(checkpoint_index))):
                    os.makedirs(str(EVAL_SAVE_EVERY_RESULTS_DIR / str(checkpoint_index)), exist_ok=True)

                f_intermediate_result_name = os.path.join(
                    str(EVAL_SAVE_EVERY_RESULTS_DIR / str(checkpoint_index)),
                    f"{train_env.batch[t]['episode_id']}.json",
                )
                f_intermediate_trajectory = {**infos[t]}
                with open(f_intermediate_result_name, "w") as f:
                    json.dump(f_intermediate_trajectory, f)

                # ------ 实验数据可视化 ------
                if args.EVAL_GENERATE_VIDEO and infos[t]['success'] > 0.8:
                    EVAL_GENERATE_VIDEO_DIR = Path(args.project_prefix) / 'DATA/output/{}/eval/videos/{}'.format(
                        args.name, args.make_dir_time)
                    generate_video(
                        video_option=["disk"],
                        video_dir=str(EVAL_GENERATE_VIDEO_DIR),
                        images=vis_frames[t],
                        episode_id=train_env.batch[t]['episode_id'],
                        checkpoint_idx=checkpoint_index,
                        metrics={
                            # "spl": infos[t]['spl'],
                            "ndtw": infos[t]['ndtw'],
                        },
                        tb_writer=writer,
                    )

                    # 保存符合可视化条件的导航样本的导航过程图像
                    folder_path = str(
                        Path(args.project_prefix) / "DATA" / "output" / args.name / "eval" / "visual" /
                        infos[t]['episode_id']
                    )
                    os.makedirs(folder_path, exist_ok=True)
                    for index, rgb in enumerate(rgb_images[t]):
                        draw_image(filenames[t][index], rgb, folder_path)

                logger.info((
                        'result-{} \t' +
                        'distance_to_goal: {} \t' +
                        'success: {} \t' +
                        'ndtw: {} \t' +
                        'sdtw: {} \t' +
                        'path_length: {} \t' +
                        'oracle_success: {} \t' +
                        'steps_taken: {}'
                ).format(
                    t,
                    infos[t]['distance_to_goal'],
                    infos[t]['success'],
                    infos[t]['ndtw'],
                    infos[t]['sdtw'],
                    infos[t]['path_length'],
                    infos[t]['oracle_success'],
                    infos[t]['steps_taken']
                ))
    pbar.close()

    # --- 新增代码 ---
    # 计算并记录最终的平均动作准确率
    if len(action_accuracies) > 0:
        avg_action_accuracy = np.mean(action_accuracies)
        logger.info(f"Average Action Accuracy: {avg_action_accuracy:.6f}")
    
    # 计算并输出模型效率指标
    if len(inference_latencies) > 0:
        avg_inference_latency = np.mean(inference_latencies)
        logger.info("=" * 80)
        logger.info("Model Efficiency Metrics:")
        logger.info(f"  Peak GPU Memory: ~{peak_gpu_memory:.1f} GB")
        logger.info(f"  Average Inference Latency: {avg_inference_latency:.3f} s")
        logger.info(f"  Total Inference Steps: {len(inference_latencies)}")
        logger.info("=" * 80)
    # ----------------

    # 保存所有导航样本的测试结果到一个json文件“intermediate_results”
    EVAL_INTERMEDIATE_RESULTS_DIR = Path(args.project_prefix) / 'DATA/output/{}/eval/intermediate_results/{}'.format(
        args.name, args.make_dir_time)
    f_intermediate_name = os.path.join(
        EVAL_INTERMEDIATE_RESULTS_DIR,
        f"stats_ckpt_{checkpoint_index}_{train_env.split}.json",
    )
    if not os.path.exists(EVAL_INTERMEDIATE_RESULTS_DIR):
        os.makedirs(EVAL_INTERMEDIATE_RESULTS_DIR, exist_ok=True)
    with open(f_intermediate_name, "w") as f:
        json.dump(stats_episodes, f)

    # 分开保存每个导航样本的测试结果“intermediate_results_every”
    new_stats_episodes = {}
    for i, j in stats_episodes.items():
        temp_1 = {}
        temp_1 = j.copy()

        temp_2 = temp_1.copy()
        for _i, _j in temp_2.items():
            if type(_j) == str or type(_j) == list or type(_j) == dict:
                del temp_1[_i]

        new_stats_episodes[i] = temp_1.copy()
    stats_episodes = new_stats_episodes.copy()

    aggregated_stats = {}
    num_episodes = len(stats_episodes)
    for stat_key in next(iter(stats_episodes.values())).keys():
        aggregated_stats[stat_key] = (
                sum(v[stat_key] for v in stats_episodes.values())
                / num_episodes
        )
    
    # 添加模型效率指标到 aggregated_stats
    if len(inference_latencies) > 0:
        aggregated_stats['peak_gpu_memory_gb'] = round(peak_gpu_memory, 2)
        aggregated_stats['avg_inference_latency_s'] = round(np.mean(inference_latencies), 3)
        aggregated_stats['total_inference_steps'] = len(inference_latencies)

    # 保存所有导航样本的平均指标数据到"result"
    fname = os.path.join(EVAL_RESULTS_DIR, f"stats_ckpt_{checkpoint_index}_{train_env.split}.json")
    if not os.path.exists(EVAL_RESULTS_DIR):
        os.makedirs(EVAL_RESULTS_DIR, exist_ok=True)
    with open(fname, "w") as f:
        json.dump(aggregated_stats, f, indent=4)

    logger.info(f"Episodes evaluated: {num_episodes}")
    checkpoint_num = checkpoint_index + 1
    for k, v in aggregated_stats.items():
        logger.info(f"Average episode {k}: {v:.6f}")
        writer.add_scalar(f"eval_{train_env.split}_{k}", v, checkpoint_num)

    try:
        train_env.simulator_tool.closeScenes()
    except Exception as e:
        logger.error(f"An unexpected error occurred while closing AirSim scenes: {e}", exc_info=True)
        pass


def draw_image(file_name, image, folder_path):
    full_name = folder_path + "/{}.png".format(file_name)
    cv2.imwrite(full_name, image)


if __name__ == "__main__":
    # 可视化相关设置
    text_color = (255, 255, 255)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)

    setup()

    action_dict = {
        0: "Stop",
        1: "Move Forward",
        2: "Turn Left",
        3: "Turn Right",
        4: "Ascend",
        5: "Descend",
        6: "Move Left",
        7: "Move Right"
    }

    if args.run_type == 'collect':
        collect_data()
    elif args.run_type == 'train':
        train_vlnce()
    elif args.run_type == 'eval':
        eval_vlnce()
    else:
        raise NotImplementedError

