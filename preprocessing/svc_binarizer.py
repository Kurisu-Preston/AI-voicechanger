import json
import logging
import os
import random
from copy import deepcopy

import numpy as np
import yaml
from resemblyzer import VoiceEncoder
from tqdm import tqdm

from infer_tools.f0_static import static_f0_time
from modules.vocoders.nsf_hifigan import NsfHifiGAN
from preprocessing.hubertinfer import HubertEncoder
from preprocessing.process_pipeline import File2Batch
from preprocessing.process_pipeline import get_pitch_parselmouth, get_pitch_crepe
from utils.hparams import hparams
from utils.hparams import set_hparams
from utils.indexed_datasets import IndexedDatasetBuilder

os.environ["OMP_NUM_THREADS"] = "1"
BASE_ITEM_ATTRIBUTES = ['wav_fn', 'spk_id']


class SvcBinarizer:
    '''
        Base class for data processing.
        1. *process* and *process_data_split*:
            process entire data, generate the train-test split (support parallel processing);
        2. *process_item*:
            process singe piece of data;
        3. *get_pitch*:
            infer the pitch using some algorithm;
        4. *get_align*:
            get the alignment using 'mel2ph' format (see https://arxiv.org/abs/1905.09263).
        5. phoneme encoder, voice encoder, etc.

        Subclasses should define:
        1. *load_metadata*:
            how to read multiple datasets from files;
        2. *train_item_names*, *valid_item_names*, *test_item_names*:
            how to split the dataset;
        3. load_ph_set:
            the phoneme set.
    '''

    def __init__(self, data_dir=None, item_attributes=None):
        self.spk_map = None
        self.vocoder = NsfHifiGAN()
        self.phone_encoder = HubertEncoder(pt_path=hparams['hubert_path'])
        if item_attributes is None:
            item_attributes = BASE_ITEM_ATTRIBUTES
        if data_dir is None:
            data_dir = hparams['raw_data_dir']
        if 'speakers' not in hparams:
            speakers = hparams['datasets']
            hparams['speakers'] = hparams['datasets']
        else:
            speakers = hparams['speakers']
        assert isinstance(speakers, list), 'Speakers must be a list'
        assert len(speakers) == len(set(speakers)), 'Speakers cannot contain duplicate names'

        self.raw_data_dirs = data_dir if isinstance(data_dir, list) else [data_dir]
        assert len(speakers) == len(self.raw_data_dirs), \
            'Number of raw data dirs must equal number of speaker names!'
        self.speakers = speakers
        self.binarization_args = hparams['binarization_args']

        self.items = {}
        # every item in self.items has some attributes
        self.item_attributes = item_attributes

        # load each dataset
        for ds_id, data_dir in enumerate(self.raw_data_dirs):
            self.load_meta_data(data_dir, ds_id)
            if ds_id == 0:
                # check program correctness
                assert all([attr in self.item_attributes for attr in list(self.items.values())[0].keys()])
        self.item_names = sorted(list(self.items.keys()))

        if self.binarization_args['shuffle']:
            random.seed(hparams['seed'])
            random.shuffle(self.item_names)

        # set default get_pitch algorithm
        if hparams['use_crepe']:
            self.get_pitch_algorithm = get_pitch_crepe
        else:
            self.get_pitch_algorithm = get_pitch_parselmouth
        print('spkers: ', set(self.speakers))
        self._train_item_names, self._test_item_names = self.split_train_test_set(self.item_names)

    @staticmethod
    def split_train_test_set(item_names):
        auto_test = item_names[-5:]
        item_names = set(deepcopy(item_names))
        if hparams['choose_test_manually']:
            prefixes = set([str(pr) for pr in hparams['test_prefixes']])
            test_item_names = set()
            # Add prefixes that specified speaker index and matches exactly item name to test set
            for prefix in deepcopy(prefixes):
                if prefix in item_names:
                    test_item_names.add(prefix)
                    prefixes.remove(prefix)
            # Add prefixes that exactly matches item name without speaker id to test set
            for prefix in deepcopy(prefixes):
                for name in item_names:
                    if name.split(':')[-1] == prefix:
                        test_item_names.add(name)
                        prefixes.remove(prefix)
            # Add names with one of the remaining prefixes to test set
            for prefix in deepcopy(prefixes):
                for name in item_names:
                    if name.startswith(prefix):
                        test_item_names.add(name)
                        prefixes.remove(prefix)
            for prefix in prefixes:
                for name in item_names:
                    if name.split(':')[-1].startswith(prefix):
                        test_item_names.add(name)
            test_item_names = sorted(list(test_item_names))
        else:
            test_item_names = auto_test
        train_item_names = [x for x in item_names if x not in set(test_item_names)]
        logging.info("train {}".format(len(train_item_names)))
        logging.info("test {}".format(len(test_item_names)))
        return train_item_names, test_item_names

    @property
    def train_item_names(self):
        return self._train_item_names

    @property
    def valid_item_names(self):
        return self._test_item_names

    @property
    def test_item_names(self):
        return self._test_item_names

    def load_meta_data(self, raw_data_dir, ds_id):
        self.items.update(File2Batch.file2temporary_dict(raw_data_dir, ds_id))

    @staticmethod
    def build_spk_map():
        spk_map = {x: i for i, x in enumerate(hparams['speakers'])}
        assert len(spk_map) <= hparams['num_spk'], 'Actual number of speakers should be smaller than num_spk!'
        return spk_map

    def item_name2spk_id(self, item_name):
        return self.spk_map[self.items[item_name]['spk_id']]

    def meta_data_iterator(self, prefix):
        if prefix == 'valid':
            item_names = self.valid_item_names
        elif prefix == 'test':
            item_names = self.test_item_names
        else:
            item_names = self.train_item_names
        for item_name in item_names:
            meta_data = self.items[item_name]
            yield item_name, meta_data

    def process(self):
        os.makedirs(hparams['binary_data_dir'], exist_ok=True)
        self.spk_map = self.build_spk_map()
        print("| spk_map: ", self.spk_map)
        spk_map_fn = f"{hparams['binary_data_dir']}/spk_map.json"
        json.dump(self.spk_map, open(spk_map_fn, 'w', encoding='utf-8'))
        self.process_data_split('valid')
        self.process_data_split('test')
        self.process_data_split('train')

    def process_data_split(self, prefix):
        data_dir = hparams['binary_data_dir']
        args = []
        builder = IndexedDatasetBuilder(f'{data_dir}/{prefix}')
        lengths = []
        total_sec = 0
        if self.binarization_args['with_spk_embed']:
            voice_encoder = VoiceEncoder().cuda()
        for item_name, meta_data in self.meta_data_iterator(prefix):
            args.append([item_name, meta_data, self.binarization_args])
        spec_min = []
        spec_max = []
        f0_dict = {}
        # code for single cpu processing
        for i in tqdm(reversed(range(len(args))), total=len(args)):
            a = args[i]
            item = self.process_item(*a)
            if item is None:
                continue
            item['spk_embed'] = voice_encoder.embed_utterance(item['wav']) \
                if self.binarization_args['with_spk_embed'] else None
            spec_min.append(item['spec_min'])
            spec_max.append(item['spec_max'])
            f0_dict[item['wav_fn']] = item['f0']
            builder.add_item(item)
            lengths.append(item['len'])
            total_sec += item['sec']
        if prefix == 'train':
            spec_max = np.max(spec_max, 0)
            spec_min = np.min(spec_min, 0)
            pitch_time = static_f0_time(f0_dict)
            with open(hparams['config_path'], encoding='utf-8') as f:
                _hparams = yaml.safe_load(f)
                _hparams['spec_max'] = spec_max.tolist()
                _hparams['spec_min'] = spec_min.tolist()
                if self.speakers == 1:
                    _hparams['f0_static'] = json.dumps(pitch_time)
            with open(hparams['config_path'], 'w', encoding='utf-8') as f:
                yaml.safe_dump(_hparams, f)
        builder.finalize()
        np.save(f'{data_dir}/{prefix}_lengths.npy', lengths)
        print(f"| {prefix} total duration: {total_sec:.3f}s")

    def process_item(self, item_name, meta_data, binarization_args):
        from preprocessing.process_pipeline import File2Batch
        return File2Batch.temporary_dict2processed_input(item_name, meta_data, self.phone_encoder)


if __name__ == "__main__":
    set_hparams()
    SvcBinarizer().process()
