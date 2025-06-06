import math
import os
import random

import numpy as np
import torch
from torch.utils.data import Sampler


class RatioSampler(Sampler):
    def __init__(
        self,
        data_source,
        scales,
        first_bs=512,
        fix_bs=True,
        divided_factor=[8, 16],
        is_training=True,
        max_ratio=10,
        max_bs=1024,
        seed=None,
        # ---------------- 下面是新增的可选参数，用于“增强版” MSR ---------------
        advanced_msr=False,
        shuffle_scale=True,
        scale_choice_prob=None,
        # ----------------------------------------------------------------------
    ):
        """
        multi scale sampler (MSR)
        Args:
            data_source (Dataset): 数据集
            scales (list): 多尺度配置
                - 若元素为 int，则表示所有宽高都同一个数值
                - 若元素为 [w, h]，则可分别指定宽高
            first_bs (int): 在 scales[0] 对应的尺度下，batch 的基准大小
            fix_bs (bool): 是否固定 batch 大小（True 表示在不同尺度下 batch 大小不变）
            divided_factor (list[w_factor, h_factor]): 对图像进行下采样，确保宽高均是 factor 的整数倍
            is_training (bool): 是否在训练模式下（控制是否 shuffle）
            max_ratio (int/float): 最长/最宽的比例阈值
            max_bs (int): 最大可用 batch_size
            seed (int): 随机种子
            ------------------------------------------------------------------
            advanced_msr (bool): 是否开启“增强版”多尺度策略 (默认为 False 保持兼容)
            shuffle_scale (bool): 多尺度下是否随机打乱 scales 的顺序
            scale_choice_prob (list/None): 若指定，为各个 scale 的采样概率（长度需与 scales 一致）
                                           若不指定，则默认均匀采样。
        """
        self.data_source = data_source
        self.ds_width = data_source.ds_width
        self.seed = data_source.seed
        if self.ds_width:
            self.wh_ratio = data_source.wh_ratio
            self.wh_ratio_sort = data_source.wh_ratio_sort
        self.n_data_samples = len(self.data_source)
        self.max_ratio = max_ratio
        self.max_bs = max_bs

        # 根据输入的 scales 分别解析宽和高
        if isinstance(scales[0], list) or isinstance(scales[0], tuple):
            width_dims = [s[0] for s in scales]
            height_dims = [s[1] for s in scales]
        else:  # 如果scales元素是int
            width_dims = scales
            height_dims = scales

        # 基准尺度与基准batch
        base_im_w = width_dims[0]
        base_im_h = height_dims[0]
        base_batch_size = first_bs
        # 参考原代码中的  "base_elements = base_im_w * base_im_h * base_batch_size"
        base_elements = base_im_w * base_im_h * base_batch_size
        self.base_elements = base_elements
        self.base_batch_size = base_batch_size
        self.base_im_h = base_im_h
        self.base_im_w = base_im_w

        # 获得分布式信息（如果非分布式训练，也可简单视作 num_replicas=1, rank=0）
        num_replicas = torch.cuda.device_count() if torch.cuda.is_available() else 1
        rank = int(os.environ['LOCAL_RANK']) if 'LOCAL_RANK' in os.environ else 0

        # 计算单卡需要采样的样本数
        num_samples_per_replica = int(math.ceil(self.n_data_samples * 1.0 / num_replicas))

        img_indices = [idx for idx in range(self.n_data_samples)]

        self.shuffle = False
        # ------------------- 增强的多尺度初始化逻辑 -------------------
        # 原本这里在训练模式会对 scales 做一次“对齐到 factor 的整数倍”以及后续组合
        # 在增强版中，我们允许对 scales 做额外随机操作（shuffle_scale）或按指定概率选。
        self.advanced_msr = advanced_msr
        self.shuffle_scale = shuffle_scale
        self.scale_choice_prob = scale_choice_prob

        if is_training:
            # 确保宽高对齐 factor
            width_dims = [int((w // divided_factor[0]) * divided_factor[0]) for w in width_dims]
            height_dims = [int((h // divided_factor[1]) * divided_factor[1]) for h in height_dims]

            # 如果指定了随机打乱 scale 的顺序:
            if self.shuffle_scale and len(width_dims) > 1:
                # 这里可在每次 epoch 重置时做 shuffle
                # 也可只在 init 做一次，这里示范仅在 init
                # (若希望每个 epoch 都变化，可把它放在 __iter__ 开头)
                zipped_scales = list(zip(width_dims, height_dims))
                random.shuffle(zipped_scales)
                width_dims, height_dims = zip(*zipped_scales)

            # 组装 (w, h, batch_size)
            img_batch_pairs = []
            for idx, (w, h) in enumerate(zip(width_dims, height_dims)):
                if fix_bs:
                    batch_size = base_batch_size
                else:
                    # 动态计算 batch_size: base_elements / (h*w)
                    # 并且受 max_bs 限制
                    batch_size = int(max(1, (base_elements / (h * w))))
                    if batch_size > self.max_bs:
                        batch_size = self.max_bs

                img_batch_pairs.append((w, h, batch_size))

            # 如果指定了抽样概率 scale_choice_prob，需要保证其长度与 scales 一致
            # 并根据概率对 (w,h,bs) 做重复/扩展/或其它策略
            if self.advanced_msr and self.scale_choice_prob is not None:
                if len(self.scale_choice_prob) != len(img_batch_pairs):
                    raise ValueError("len(scale_choice_prob) must match len(scales).")
                # 这里仅示例：若概率大就重复多份
                # (也可改写成别的策略，比如按概率选下标)
                new_img_batch_pairs = []
                for (pair, p) in zip(img_batch_pairs, self.scale_choice_prob):
                    replicate_times = max(1, int(p * 10))  # 简单示例：放大 p 倍
                    new_img_batch_pairs.extend([pair] * replicate_times)
                # 打乱
                random.shuffle(new_img_batch_pairs)
                img_batch_pairs = new_img_batch_pairs

            self.img_batch_pairs = img_batch_pairs
            self.shuffle = True
            np.random.seed(seed)
            random.seed(seed)
        else:
            # 测试/验证模式下只使用 (base_im_w, base_im_h, base_batch_size)
            self.img_batch_pairs = [(base_im_w, base_im_h, base_batch_size)]

        self.img_indices = img_indices
        self.n_samples_per_replica = num_samples_per_replica
        self.epoch = 0
        self.rank = rank
        self.num_replicas = num_replicas

        # 兼容后续逻辑
        self.current = 0
        self.is_training = is_training
        if is_training:
            # 取出当前 rank 对应的样本
            indices_rank_i = self.img_indices[self.rank:len(self.img_indices):self.num_replicas]
        else:
            indices_rank_i = self.img_indices

        # 对 ratio 进行记录、分组
        self.indices_rank_i_ori = np.array(self.wh_ratio_sort[indices_rank_i])
        self.indices_rank_i_ratio = self.wh_ratio[self.indices_rank_i_ori]
        indices_rank_i_ratio_unique = np.unique(self.indices_rank_i_ratio)
        self.indices_rank_i_ratio_unique = indices_rank_i_ratio_unique.tolist()

        # 批次列表
        self.batch_list = self.create_batch()
        self.length = len(self.batch_list)
        self.batchs_in_one_epoch_id = [i for i in range(self.length)]

    def create_batch(self):
        """
        这里是“核心”增强点之一：
          - 原逻辑先根据 ratio 的不同做分组；
          - 然后 shuffle 再构造 batch；
          - batch_size_ratio 可能根据 ratio 动态调整。
        在“增强”场景下，我们可以在这里做更多的策略，比如：
          1. 对 ratio 分桶后，优先抽取极端宽高比或极端分辨率
          2. 对 batch_size 做更细粒度控制
          3. 加入一些容错策略，自动平衡各 ratio 的采样数
        以下示例保持与原来类似的流程，只在其中演示如何“插入”更多的策略点。
        """
        batch_list = []
        # 若在 advanced_msr 模式下，可以先对 ratio 做随机打乱
        if self.advanced_msr:
            random.shuffle(self.indices_rank_i_ratio_unique)

        for ratio in self.indices_rank_i_ratio_unique:
            ratio_ids = np.where(self.indices_rank_i_ratio == ratio)[0]
            ratio_ids = self.indices_rank_i_ori[ratio_ids]

            if self.shuffle:
                random.shuffle(ratio_ids)

            num_ratio = ratio_ids.shape[0]
            # 当 ratio < 5 时，采用默认 base_batch_size，否则动态计算
            if ratio < 5:
                batch_size_ratio = self.base_batch_size
            else:
                # 动态 batch_size
                # 这里可再次加 max_bs 限制
                batch_size_ratio = min(
                    self.max_bs,
                    int(
                        max(1,
                            (self.base_elements / (self.base_im_h * ratio * self.base_im_h))))
                )

            if num_ratio > batch_size_ratio:
                batch_num_ratio = num_ratio // batch_size_ratio
                # 打印日志以便观察
                print(self.rank, "Ratio:", ratio,
                      "num_ratio:", num_ratio,
                      "batch_num_ratio:", batch_num_ratio,
                      "batch_size_ratio:", batch_size_ratio)

                ratio_ids_full = ratio_ids[:batch_num_ratio * batch_size_ratio]
                ratio_ids_full = ratio_ids_full.reshape(batch_num_ratio, batch_size_ratio, 1)
                # 组装 [w, h, idx, ratio]
                w = np.full_like(ratio_ids_full, ratio * self.base_im_h)
                h = np.full_like(ratio_ids_full, self.base_im_h)
                ra_wh = np.full_like(ratio_ids_full, ratio)
                ratio_ids_full = np.concatenate([w, h, ratio_ids_full, ra_wh], axis=-1)
                batch_ratio = ratio_ids_full.tolist()

                remain = num_ratio - batch_num_ratio * batch_size_ratio
                if remain > 0:
                    drop = ratio_ids[-remain:]
                    if self.is_training:
                        # 这里也可在 advanced_msr 模式做更多操作，比如把剩余的合并到其他 batch
                        # 下方仍是和原版逻辑相同的处理：用开头若干数据来凑齐一个整批
                        drop_full = ratio_ids[:batch_size_ratio - remain]
                        drop = np.append(drop_full, drop)
                    drop = drop.reshape(-1, 1)
                    w = np.full_like(drop, ratio * self.base_im_h)
                    h = np.full_like(drop, self.base_im_h)
                    ra_wh = np.full_like(drop, ratio)
                    drop = np.concatenate([w, h, drop, ra_wh], axis=-1)
                    batch_ratio.append(drop.tolist())
                batch_list += batch_ratio
            else:
                print(self.rank, "Ratio:", ratio,
                      "num_ratio:", num_ratio,
                      "batch_size_ratio:", batch_size_ratio)
                ratio_ids = ratio_ids.reshape(-1, 1)
                w = np.full_like(ratio_ids, ratio * self.base_im_h)
                h = np.full_like(ratio_ids, self.base_im_h)
                ra_wh = np.full_like(ratio_ids, ratio)
                ratio_ids = np.concatenate([w, h, ratio_ids, ra_wh], axis=-1)
                batch_list.append(ratio_ids.tolist())

        return batch_list

    def __iter__(self):
        # 每个 epoch 开始前
        if self.shuffle or self.is_training:
            random.seed(self.epoch)
            self.epoch += 1
            # 重新创建 batch_list
            self.batch_list = self.create_batch()
            # 打乱 batch 的顺序
            random.shuffle(self.batchs_in_one_epoch_id)

        for batch_tuple_id in self.batchs_in_one_epoch_id:
            yield self.batch_list[batch_tuple_id]

    def set_epoch(self, epoch: int):
        """
        DDP 等分布式情况下，需要使用 sampler.set_epoch(epoch) 来控制 shuffle。
        """
        self.epoch = epoch

    def __len__(self):
        return self.length
