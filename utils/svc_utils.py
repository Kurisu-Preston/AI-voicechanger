import glob
import importlib
import os

import matplotlib
import numpy as np
import torch
import torch.distributions
import torch.optim
import torch.optim
import torch.utils.data

from preprocessing.process_pipeline import File2Batch
from utils.hparams import hparams
from utils.indexed_datasets import IndexedDataset
from utils.pitch_utils import norm_interp_f0

matplotlib.use('Agg')


class SvcDataset(torch.utils.data.Dataset):
    def __init__(self, prefix, shuffle=False):
        super().__init__()
        self.hparams = hparams
        self.shuffle = shuffle
        self.sort_by_len = hparams['sort_by_len']
        self.sizes = None
        self.data_dir = hparams['binary_data_dir']
        self.prefix = prefix
        self.sizes = np.load(f'{self.data_dir}/{self.prefix}_lengths.npy')
        self.indexed_ds = None
        # self.name2spk_id={}

        # pitch stats
        f0_stats_fn = f'{self.data_dir}/train_f0s_mean_std.npy'
        if os.path.exists(f0_stats_fn):
            hparams['f0_mean'], hparams['f0_std'] = self.f0_mean, self.f0_std = np.load(f0_stats_fn)
            hparams['f0_mean'] = float(hparams['f0_mean'])
            hparams['f0_std'] = float(hparams['f0_std'])
        else:
            hparams['f0_mean'], hparams['f0_std'] = self.f0_mean, self.f0_std = None, None

        if prefix == 'test':
            if hparams['test_input_dir'] != '':
                self.indexed_ds, self.sizes = self.load_test_inputs(hparams['test_input_dir'])
            else:
                if hparams['num_test_samples'] > 0:
                    self.avail_idxs = list(range(hparams['num_test_samples'])) + hparams['test_ids']
                    self.sizes = [self.sizes[i] for i in self.avail_idxs]

    @property
    def _sizes(self):
        return self.sizes

    def _get_item(self, index):
        if hasattr(self, 'avail_idxs') and self.avail_idxs is not None:
            index = self.avail_idxs[index]
        if self.indexed_ds is None:
            self.indexed_ds = IndexedDataset(f'{self.data_dir}/{self.prefix}')
        return self.indexed_ds[index]

    def __getitem__(self, index):
        item = self._get_item(index)
        max_frames = hparams['max_frames']
        spec = torch.Tensor(item['mel'])[:max_frames]
        # energy = (spec.exp() ** 2).sum(-1).sqrt()
        mel2ph = torch.LongTensor(item['mel2ph'])[:max_frames] if 'mel2ph' in item else None
        f0, uv = norm_interp_f0(item["f0"][:max_frames], hparams)
        hubert = torch.Tensor(item['hubert'][:hparams['max_input_tokens']])
        pitch = torch.LongTensor(item.get("pitch"))[:max_frames]
        sample = {
            "id": index,
            "item_name": item['item_name'],
            "hubert": hubert,
            "mel": spec,
            "pitch": pitch,
            "f0": f0,
            "uv": uv,
            "mel2ph": mel2ph,
            "mel_nonpadding": spec.abs().sum(-1) > 0,
        }
        if hparams['use_energy_embed']:
            sample['energy'] = item['energy']
        if hparams['use_spk_embed']:
            sample["spk_embed"] = torch.Tensor(item['spk_embed'])
        if hparams['use_spk_id']:
            sample["spk_id"] = item['spk_id']
        return sample

    @staticmethod
    def collater(samples):
        return File2Batch.processed_input2batch(samples)

    @staticmethod
    def load_test_inputs(test_input_dir):
        inp_wav_paths = glob.glob(f'{test_input_dir}/*.wav') + glob.glob(f'{test_input_dir}/*.mp3')
        sizes = []
        items = []

        binarizer_cls = hparams.get("binarizer_cls", 'basics.base_binarizer.BaseBinarizer')
        pkg = ".".join(binarizer_cls.split(".")[:-1])
        cls_name = binarizer_cls.split(".")[-1]
        binarizer_cls = getattr(importlib.import_module(pkg), cls_name)
        from preprocessing.hubertinfer import HubertEncoder
        for wav_fn in inp_wav_paths:
            item_name = os.path.basename(wav_fn)
            wav_fn = wav_fn
            encoder = HubertEncoder(hparams['hubert_path'])
            item = binarizer_cls.process_item(item_name, {'wav_fn': wav_fn}, encoder)
            print(item)
            items.append(item)
            sizes.append(item['len'])
        return items, sizes

    def __len__(self):
        return len(self._sizes)

    def num_tokens(self, index):
        return self.size(index)

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        size = min(self._sizes[index], hparams['max_frames'])
        return size

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            indices = np.random.permutation(len(self))
            if self.sort_by_len:
                indices = indices[np.argsort(np.array(self._sizes)[indices], kind='mergesort')]
                # 先random, 然后稳定排序, 保证排序后同长度的数据顺序是依照random permutation的 (被其随机打乱).
        else:
            indices = np.arange(len(self))
        return indices

    @property
    def num_workers(self):
        return int(os.getenv('NUM_WORKERS', hparams['ds_workers']))
