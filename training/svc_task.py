import os
from multiprocessing.pool import Pool

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import torch.distributions
import torch.nn.functional as F
import torch.optim
import torch.utils.data
from tqdm import tqdm

import utils
from modules.commons.ssim import ssim
from modules.diff.diffusion import GaussianDiffusion
from modules.diff.net import DiffNet
from modules.vocoders.nsf_hifigan import NsfHifiGAN, nsf_hifigan
from preprocessing.hubertinfer import HubertEncoder
from preprocessing.process_pipeline import get_pitch_parselmouth
from training.base_task import BaseTask
from utils import audio
from utils.hparams import hparams
from utils.pitch_utils import denorm_f0
from utils.pl_utils import data_loader
from utils.plot import spec_to_figure, f0_to_figure
from utils.svc_utils import SvcDataset

matplotlib.use('Agg')
DIFF_DECODERS = {
    'wavenet': lambda hp: DiffNet(hp['audio_num_mel_bins'])
}


class SvcTask(BaseTask):
    def __init__(self):
        super(SvcTask, self).__init__()
        self.vocoder = NsfHifiGAN()
        self.phone_encoder = HubertEncoder(hparams['hubert_path'])
        self.saving_result_pool = None
        self.saving_results_futures = None
        self.stats = {}
        self.dataset_cls = SvcDataset
        self.mse_loss_fn = torch.nn.MSELoss()
        mel_losses = hparams['mel_loss'].split("|")
        self.loss_and_lambda = {}
        for i, l in enumerate(mel_losses):
            if l == '':
                continue
            if ':' in l:
                l, lbd = l.split(":")
                lbd = float(lbd)
            else:
                lbd = 1.0
            self.loss_and_lambda[l] = lbd
        print("| Mel losses:", self.loss_and_lambda)

    def build_dataloader(self, dataset, shuffle, max_tokens=None, max_sentences=None,
                         required_batch_size_multiple=-1, endless=False, batch_by_size=True):
        devices_cnt = torch.cuda.device_count()
        if devices_cnt == 0:
            devices_cnt = 1
        if required_batch_size_multiple == -1:
            required_batch_size_multiple = devices_cnt

        def shuffle_batches(batches):
            np.random.shuffle(batches)
            return batches

        if max_tokens is not None:
            max_tokens *= devices_cnt
        if max_sentences is not None:
            max_sentences *= devices_cnt
        indices = dataset.ordered_indices()
        if batch_by_size:
            batch_sampler = utils.batch_by_size(
                indices, dataset.num_tokens, max_tokens=max_tokens, max_sentences=max_sentences,
                required_batch_size_multiple=required_batch_size_multiple,
            )
        else:
            batch_sampler = []
            for i in range(0, len(indices), max_sentences):
                batch_sampler.append(indices[i:i + max_sentences])

        if shuffle:
            batches = shuffle_batches(list(batch_sampler))
            if endless:
                batches = [b for _ in range(1000) for b in shuffle_batches(list(batch_sampler))]
        else:
            batches = batch_sampler
            if endless:
                batches = [b for _ in range(1000) for b in batches]
        num_workers = dataset.num_workers
        if self.trainer.use_ddp:
            num_replicas = dist.get_world_size()
            rank = dist.get_rank()
            batches = [x[rank::num_replicas] for x in batches if len(x) % num_replicas == 0]
        return torch.utils.data.DataLoader(dataset,
                                           collate_fn=dataset.collater,
                                           batch_sampler=batches,
                                           num_workers=num_workers,
                                           pin_memory=False)

    def test_start(self):
        self.saving_result_pool = Pool(8)
        self.saving_results_futures = []
        self.vocoder = nsf_hifigan

    def test_end(self, outputs):
        self.saving_result_pool.close()
        [f.get() for f in tqdm(self.saving_results_futures)]
        self.saving_result_pool.join()
        return {}

    @data_loader
    def train_dataloader(self):
        train_dataset = self.dataset_cls(hparams['train_set_name'], shuffle=True)
        return self.build_dataloader(train_dataset, True, self.max_tokens, self.max_sentences,
                                     endless=hparams['endless_ds'])

    @data_loader
    def val_dataloader(self):
        valid_dataset = self.dataset_cls(hparams['valid_set_name'], shuffle=False)
        return self.build_dataloader(valid_dataset, False, self.max_eval_tokens, self.max_eval_sentences)

    @data_loader
    def test_dataloader(self):
        test_dataset = self.dataset_cls(hparams['test_set_name'], shuffle=False)
        return self.build_dataloader(test_dataset, False, self.max_eval_tokens,
                                     self.max_eval_sentences, batch_by_size=False)

    def build_model(self):
        self.build_tts_model()
        if hparams['load_ckpt'] != '':
            self.load_ckpt(hparams['load_ckpt'], strict=True)
        utils.print_arch(self.model)
        return self.model

    def build_tts_model(self):
        mel_bins = hparams['audio_num_mel_bins']
        self.model = GaussianDiffusion(
            phone_encoder=self.phone_encoder,
            out_dims=mel_bins, denoise_fn=DIFF_DECODERS[hparams['diff_decoder_type']](hparams),
            timesteps=hparams['timesteps'],
            K_step=hparams['K_step'],
            loss_type=hparams['diff_loss_type'],
            spec_min=hparams['spec_min'], spec_max=hparams['spec_max'],
        )

    def build_optimizer(self, model):
        self.optimizer = optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=hparams['lr'],
            betas=(hparams['optimizer_adam_beta1'], hparams['optimizer_adam_beta2']),
            weight_decay=hparams['weight_decay'])
        return optimizer

    @staticmethod
    def run_model(model, sample, return_output=False, infer=False):
        '''
            steps:
            1. run the full model, calc the main loss
            2. calculate loss for dur_predictor, pitch_predictor, energy_predictor
        '''
        hubert = sample['hubert']  # [B, T_t,H]
        target = sample['mels']  # [B, T_s, 80]
        mel2ph = sample['mel2ph']  # [B, T_s]
        f0 = sample['f0']
        uv = sample['uv']
        energy = sample.get('energy')

        spk_embed = sample.get('spk_embed') if not hparams['use_spk_id'] else sample.get('spk_ids')
        output = model(hubert, mel2ph=mel2ph, spk_embed=spk_embed, ref_mels=target, f0=f0, uv=uv, energy=energy, infer=infer)

        losses = {}
        if 'diff_loss' in output:
            losses['mel'] = output['diff_loss']
        if not return_output:
            return losses
        else:
            return losses, output

    def build_scheduler(self, optimizer):
        return torch.optim.lr_scheduler.StepLR(optimizer, hparams['decay_steps'], gamma=0.5)

    def _training_step(self, sample, batch_idx, _):
        log_outputs = self.run_model(self.model, sample)
        total_loss = sum([v for v in log_outputs.values() if isinstance(v, torch.Tensor) and v.requires_grad])
        log_outputs['batch_size'] = sample['hubert'].size()[0]
        log_outputs['lr'] = self.scheduler.get_lr()[0]
        return total_loss, log_outputs

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx):
        if optimizer is None:
            return
        optimizer.step()
        optimizer.zero_grad()
        if self.scheduler is not None:
            self.scheduler.step(self.global_step // hparams['accumulate_grad_batches'])

    def validation_step(self, sample, batch_idx):
        outputs = {}
        hubert = sample['hubert']  # [B, T_t]
        energy = sample.get('energy')
        spk_embed = sample.get('spk_embed') if not hparams['use_spk_id'] else sample.get('spk_ids')
        mel2ph = sample['mel2ph']

        outputs['losses'] = {}

        outputs['losses'], model_out = self.run_model(self.model, sample, return_output=True, infer=False)

        outputs['total_loss'] = sum(outputs['losses'].values())
        outputs['nsamples'] = sample['nsamples']
        outputs = utils.tensors_to_scalars(outputs)
        if batch_idx < hparams['num_valid_plots']:
            model_out = self.model(
                hubert, spk_embed=spk_embed, mel2ph=mel2ph, f0=sample['f0'], uv=sample['uv'], energy=energy,
                ref_mels=None, infer=True
            )

            gt_f0 = denorm_f0(sample['f0'], sample['uv'], hparams)
            pred_f0 = model_out.get('f0_denorm')
            self.plot_wav(batch_idx, sample['mels'], model_out['mel_out'], is_mel=True, gt_f0=gt_f0, f0=pred_f0)
            self.plot_mel(batch_idx, sample['mels'], model_out['mel_out'], name=f'diffmel_{batch_idx}')
            if hparams['use_pitch_embed']:
                self.plot_pitch(batch_idx, sample, model_out)
        return outputs

    def _validation_end(self, outputs):
        all_losses_meter = {
            'total_loss': utils.AvgrageMeter(),
        }
        for output in outputs:
            n = output['nsamples']
            for k, v in output['losses'].items():
                if k not in all_losses_meter:
                    all_losses_meter[k] = utils.AvgrageMeter()
                all_losses_meter[k].update(v, n)
            all_losses_meter['total_loss'].update(output['total_loss'], n)
        return {k: round(v.avg, 4) for k, v in all_losses_meter.items()}

    ############
    # losses
    ############
    def add_mel_loss(self, mel_out, target, losses, postfix='', mel_mix_loss=None):
        if mel_mix_loss is None:
            for loss_name, lbd in self.loss_and_lambda.items():
                if 'l1' == loss_name:
                    l = self.l1_loss(mel_out, target)
                elif 'mse' == loss_name:
                    raise NotImplementedError
                elif 'ssim' == loss_name:
                    l = self.ssim_loss(mel_out, target)
                elif 'gdl' == loss_name:
                    raise NotImplementedError
                losses[f'{loss_name}{postfix}'] = l * lbd
        else:
            raise NotImplementedError

    def l1_loss(self, decoder_output, target):
        # decoder_output : B x T x n_mel
        # target : B x T x n_mel
        l1_loss = F.l1_loss(decoder_output, target, reduction='none')
        weights = self.weights_nonzero_speech(target)
        l1_loss = (l1_loss * weights).sum() / weights.sum()
        return l1_loss

    def ssim_loss(self, decoder_output, target, bias=6.0):
        # decoder_output : B x T x n_mel
        # target : B x T x n_mel
        assert decoder_output.shape == target.shape
        weights = self.weights_nonzero_speech(target)
        decoder_output = decoder_output[:, None] + bias
        target = target[:, None] + bias
        ssim_loss = 1 - ssim(decoder_output, target, size_average=False)
        ssim_loss = (ssim_loss * weights).sum() / weights.sum()
        return ssim_loss

    def add_pitch_loss(self, output, sample, losses):
        if hparams['pitch_type'] == 'ph':
            nonpadding = (sample['txt_tokens'] != 0).float()
            pitch_loss_fn = F.l1_loss if hparams['pitch_loss'] == 'l1' else F.mse_loss
            losses['f0'] = (pitch_loss_fn(output['pitch_pred'][:, :, 0], sample['f0'],
                                          reduction='none') * nonpadding).sum() \
                           / nonpadding.sum() * hparams['lambda_f0']
            return
        mel2ph = sample['mel2ph']  # [B, T_s]
        f0 = sample['f0']
        uv = sample['uv']
        nonpadding = (mel2ph != 0).float()
        if hparams['pitch_type'] == 'frame':
            self.add_f0_loss(output['pitch_pred'], f0, uv, losses, nonpadding=nonpadding)

    @staticmethod
    def add_f0_loss(p_pred, f0, uv, losses, nonpadding):
        assert p_pred[..., 0].shape == f0.shape
        if hparams['use_uv']:
            assert p_pred[..., 1].shape == uv.shape
            losses['uv'] = (F.binary_cross_entropy_with_logits(
                p_pred[:, :, 1], uv, reduction='none') * nonpadding).sum() \
                           / nonpadding.sum() * hparams['lambda_uv']
            nonpadding = nonpadding * (uv == 0).float()

        f0_pred = p_pred[:, :, 0]
        if hparams['pitch_loss'] in ['l1', 'l2']:
            pitch_loss_fn = F.l1_loss if hparams['pitch_loss'] == 'l1' else F.mse_loss
            losses['f0'] = (pitch_loss_fn(f0_pred, f0, reduction='none') * nonpadding).sum() \
                           / nonpadding.sum() * hparams['lambda_f0']
        elif hparams['pitch_loss'] == 'ssim':
            return NotImplementedError

    @staticmethod
    def add_energy_loss(energy_pred, energy, losses):
        nonpadding = (energy != 0).float()
        loss = (F.mse_loss(energy_pred, energy, reduction='none') * nonpadding).sum() / nonpadding.sum()
        loss = loss * hparams['lambda_energy']
        losses['e'] = loss

    ############
    # validation plots
    ############
    def plot_mel(self, batch_idx, spec, spec_out, name=None):
        spec_cat = torch.cat([spec, spec_out], -1)
        name = f'mel_{batch_idx}' if name is None else name
        vmin = hparams['mel_vmin']
        vmax = hparams['mel_vmax']
        self.logger.experiment.add_figure(name, spec_to_figure(spec_cat[0], vmin, vmax), self.global_step)

    def plot_pitch(self, batch_idx, sample, model_out):
        f0 = sample['f0']
        if hparams['pitch_type'] == 'ph':
            mel2ph = sample['mel2ph']
            f0 = self.expand_f0_ph(f0, mel2ph)
            f0_pred = self.expand_f0_ph(model_out['pitch_pred'][:, :, 0], mel2ph)
            self.logger.experiment.add_figure(
                f'f0_{batch_idx}', f0_to_figure(f0[0], None, f0_pred[0]), self.global_step)
            return
        f0 = denorm_f0(f0, sample['uv'], hparams)
        if hparams['pitch_type'] == 'frame':
            pitch_pred = denorm_f0(model_out['pitch_pred'][:, :, 0], sample['uv'], hparams)
            self.logger.experiment.add_figure(
                f'f0_{batch_idx}', f0_to_figure(f0[0], None, pitch_pred[0]), self.global_step)

    def plot_wav(self, batch_idx, gt_wav, wav_out, is_mel=False, gt_f0=None, f0=None, name=None):
        gt_wav = gt_wav[0].cpu().numpy()
        wav_out = wav_out[0].cpu().numpy()
        gt_f0 = gt_f0[0].cpu().numpy()
        f0 = f0[0].cpu().numpy()
        if is_mel:
            gt_wav = self.vocoder.spec2wav(gt_wav, f0=gt_f0)
            wav_out = self.vocoder.spec2wav(wav_out, f0=f0)
        self.logger.experiment.add_audio(f'gt_{batch_idx}', gt_wav, sample_rate=hparams['audio_sample_rate'],
                                         global_step=self.global_step)
        self.logger.experiment.add_audio(f'wav_{batch_idx}', wav_out, sample_rate=hparams['audio_sample_rate'],
                                         global_step=self.global_step)

    ############
    # infer
    ############
    def test_step(self, sample, batch_idx):
        spk_embed = sample.get('spk_embed') if not hparams['use_spk_id'] else sample.get('spk_ids')
        hubert = sample['hubert']
        ref_mels = None
        mel2ph = sample['mel2ph']
        f0 = sample['f0']
        uv = sample['uv']
        outputs = self.model(hubert, spk_embed=spk_embed, mel2ph=mel2ph, f0=f0, uv=uv, ref_mels=ref_mels,
                             infer=True)
        sample['outputs'] = self.model.out2mel(outputs['mel_out'])
        sample['mel2ph_pred'] = outputs['mel2ph']
        sample['f0'] = denorm_f0(sample['f0'], sample['uv'], hparams)
        sample['f0_pred'] = outputs.get('f0_denorm')
        return self.after_infer(sample)

    def after_infer(self, predictions):
        if self.saving_result_pool is None and not hparams['profile_infer']:
            self.saving_result_pool = Pool(min(int(os.getenv('N_PROC', os.cpu_count())), 16))
            self.saving_results_futures = []
        predictions = utils.unpack_dict_to_list(predictions)
        t = tqdm(predictions)
        for num_predictions, prediction in enumerate(t):
            for k, v in prediction.items():
                if type(v) is torch.Tensor:
                    prediction[k] = v.cpu().numpy()

            item_name = prediction.get('item_name')

            # remove paddings
            mel_gt = prediction["mels"]
            mel_gt_mask = np.abs(mel_gt).sum(-1) > 0
            mel_gt = mel_gt[mel_gt_mask]
            mel_pred = prediction["outputs"]
            mel_pred_mask = np.abs(mel_pred).sum(-1) > 0
            mel_pred = mel_pred[mel_pred_mask]
            mel_gt = np.clip(mel_gt, hparams['mel_vmin'], hparams['mel_vmax'])
            mel_pred = np.clip(mel_pred, hparams['mel_vmin'], hparams['mel_vmax'])

            f0_gt = prediction.get("f0")
            f0_pred = f0_gt
            if f0_pred is not None:
                f0_gt = f0_gt[mel_gt_mask]
                if len(f0_pred) > len(mel_pred_mask):
                    f0_pred = f0_pred[:len(mel_pred_mask)]
                f0_pred = f0_pred[mel_pred_mask]
            gen_dir = os.path.join(hparams['work_dir'],
                                   f'generated_{self.trainer.global_step}_{hparams["gen_dir_name"]}')
            wav_pred = self.vocoder.spec2wav(mel_pred, f0=f0_pred)
            if not hparams['profile_infer']:
                os.makedirs(gen_dir, exist_ok=True)
                os.makedirs(f'{gen_dir}/wavs', exist_ok=True)
                os.makedirs(f'{gen_dir}/plot', exist_ok=True)
                os.makedirs(os.path.join(hparams['work_dir'], 'P_mels_npy'), exist_ok=True)
                os.makedirs(os.path.join(hparams['work_dir'], 'G_mels_npy'), exist_ok=True)
                self.saving_results_futures.append(
                    self.saving_result_pool.apply_async(self.save_result, args=[
                        wav_pred, mel_pred, 'P', item_name, gen_dir]))

                if mel_gt is not None and hparams['save_gt']:
                    wav_gt = self.vocoder.spec2wav(mel_gt, f0=f0_gt)
                    self.saving_results_futures.append(
                        self.saving_result_pool.apply_async(self.save_result, args=[
                            wav_gt, mel_gt, 'G', item_name, gen_dir]))
                    if hparams['save_f0']:
                        import matplotlib.pyplot as plt
                        f0_pred_ = f0_pred
                        f0_gt_, _ = get_pitch_parselmouth(wav_gt, mel_gt, hparams)
                        fig = plt.figure()
                        plt.plot(f0_pred_, label=r'$f0_P$')
                        plt.plot(f0_gt_, label=r'$f0_G$')
                        plt.legend()
                        plt.tight_layout()
                        plt.savefig(f'{gen_dir}/plot/[F0][{item_name}]{text}.png', format='png')
                        plt.close(fig)

                t.set_description(
                    f"Pred_shape: {mel_pred.shape}, gt_shape: {mel_gt.shape}")
            else:
                if 'gen_wav_time' not in self.stats:
                    self.stats['gen_wav_time'] = 0
                self.stats['gen_wav_time'] += len(wav_pred) / hparams['audio_sample_rate']
                print('gen_wav_time: ', self.stats['gen_wav_time'])

        return {}

    @staticmethod
    def save_result(wav_out, mel, prefix, item_name, gen_dir):
        item_name = item_name.replace('/', '-')
        base_fn = f'[{item_name}][{prefix}]'
        base_fn += ('-' + hparams['exp_name'])
        np.save(os.path.join(hparams['work_dir'], f'{prefix}_mels_npy', item_name), mel)
        audio.save_wav(wav_out, f'{gen_dir}/wavs/{base_fn}.wav', 24000,  # hparams['audio_sample_rate'],
                       norm=hparams['out_wav_norm'])
        fig = plt.figure(figsize=(14, 10))
        spec_vmin = hparams['mel_vmin']
        spec_vmax = hparams['mel_vmax']
        heatmap = plt.pcolor(mel.T, vmin=spec_vmin, vmax=spec_vmax)
        fig.colorbar(heatmap)
        f0, _ = get_pitch_parselmouth(wav_out, mel, hparams)
        f0 = (f0 - 100) / (800 - 100) * 80 * (f0 > 0)
        plt.plot(f0, c='white', linewidth=1, alpha=0.6)
        plt.tight_layout()
        plt.savefig(f'{gen_dir}/plot/{base_fn}.png', format='png', dpi=1000)
        plt.close(fig)

    ##############
    # utils
    ##############
    @staticmethod
    def expand_f0_ph(f0, mel2ph):
        f0 = denorm_f0(f0, None, hparams)
        f0 = F.pad(f0, [1, 0])
        f0 = torch.gather(f0, 1, mel2ph)  # [B, T_mel]
        return f0

    @staticmethod
    def weights_nonzero_speech(target):
        # target : B x T x mel
        # Assign weight 1.0 to all labels except for padding (id=0).
        dim = target.size(-1)
        return target.abs().sum(-1, keepdim=True).ne(0).float().repeat(1, 1, dim)
