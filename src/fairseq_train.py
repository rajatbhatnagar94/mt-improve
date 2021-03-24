from fairseq_cli.train import cli_main
import sys
import os
import fairseq
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
import argparse
from fairseq import options
from fairseq.optim.adam import FairseqAdam
from fairseq_cli import train
from fairseq.distributed import fsdp_enable_wrap, fsdp_wrap, utils as distributed_utils

base_dir = "/data/rajat/datasets/mt/ro-en/data/en-ro/75000"

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

parameters = [
	# '/projects/rabh7066/software/anaconda/envs/fairseq/bin/fairseq-train',
	'--task', 'translation',
	f'{base_dir}/data-bin',
	'--source-lang', 'ro', '--target-lang', 'en', '--arch', 'transformer',
	'--encoder-layers', '6', '--decoder-layers', '6',
	'--encoder-embed-dim', '512', '--decoder-embed-dim', '512', '--encoder-ffn-embed-dim', '1024',
	'--decoder-ffn-embed-dim', '1024', '--encoder-attention-heads', '4', '--decoder-attention-heads',
	'4', '--encoder-normalize-before', '--decoder-normalize-before', '--dropout', '0.3', '--attention-dropout',
	'0.2', '--relu-dropout', '0.2', '--weight-decay', '0.0001', '--label-smoothing', '0.2', '--criterion',
	'label_smoothed_cross_entropy', '--optimizer', 'adam', '--adam-betas', '(0.9, 0.98)', '--clip-norm', '0',
	'--lr-scheduler', 'inverse_sqrt', '--warmup-updates', '4000', '--warmup-init-lr', '1e-7',
	'--lr', '5e-4',  '--max-tokens', '1000', '--update-freq', '4',
	'--tensorboard-logdir', f'{base_dir}/logs/tensorboard',
	'--max-epoch', '200', '--save-interval', '20', '--keep-best-checkpoints', '1',
	'--save-dir', f'{base_dir}/checkpoints',
	'--log-format', 'simple', '--log-interval', '200',
	'--seed', '1', '--skip-invalid-size-inputs-valid-test'
]

def train_main():
	train_parser = options.get_training_parser()
	train_args = options.parse_args_and_arch(train_parser, parameters)
	cfg = convert_namespace_to_omegaconf(train_args)
	distributed_utils.call_main(cfg, train.main)


def main():
	epochs = 10
	train_parser = options.get_training_parser()
	train_args = options.parse_args_and_arch(train_parser, parameters)
	cfg = convert_namespace_to_omegaconf(train_args)
	task = fairseq.tasks.setup_task(cfg.task)

	model = task.build_model(cfg.model)
	criterion = task.build_criterion(cfg.criterion)
	optimizer = FairseqAdam(cfg.optimizer, model.parameters())
	task.load_dataset('train')
	task.load_dataset('valid')

	dataset = task.dataset('train')
	for epoch in range(epochs):
		batch_itr = task.get_batch_iterator(dataset, max_tokens=cfg.dataset.max_tokens).next_epoch_itr(shuffle=True)
		for batch_idx, batch in enumerate(batch_itr):
			output = task.train_step(batch, model, criterion, optimizer, batch_idx)
			loss = output[0].item() / output[1]
			task.optimizer_step(optimizer, model, batch_idx)
			print(f"loss: {loss} after epoch: {epoch}, batch_idx: {batch_idx}")

def display(batch, task):
	for sentence in batch['target']:
		for token in sentence:
			index = token.item()
			if index == 1:
				break
			print(task.target_dictionary[token.item()])

# main()
train_main()
# sys.exit(cli_main())
# CUDA_VISIBLE_DEVICES=0 fairseq-train /data/rajat/datasets/mt/500/data-bin
# --source-lang hi --target-lang en --arch transformer
# --encoder-layers 6 --decoder-layers 6 --encoder-embed-dim 512
# --decoder-embed-dim 512 --encoder-ffn-embed-dim 1024
# --decoder-ffn-embed-dim 1024 --encoder-attention-heads 4
# --decoder-attention-heads 4 --encoder-normalize-before
# --decoder-normalize-before --dropout 0.3
# --attention-dropout 0.2 --relu-dropout 0.2 --weight-decay 0.0001
# --label-smoothing 0.2 --criterion label_smoothed_cross_entropy
# --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0
# --lr-scheduler inverse_sqrt --warmup-updates 4000
# --warmup-init-lr 1e-7 --lr 5e-4 --max-tokens 1000
# --update-freq 4
# --save-dir /data/rajat/datasets/mt/500/checkpoints --seed 1
# --skip-invalid-size-inputs-valid-test
# --max-epoch 200 --save-interval 20 --keep-best-checkpoints 1

# CUDA_VISIBLE_DEVICES=0 python3 ../fairseq/fairseq_cli/train.py /data/rajat/datasets/mt/500/data-bin --task translation --source-lang hi --target-lang en --arch transformer --encoder-layers 6 --decoder-layers 6 --encoder-embed-dim 512 --decoder-embed-dim 512 --encoder-ffn-embed-dim 1024 --decoder-ffn-embed-dim 1024 --encoder-attention-heads 4 --decoder-attention-heads 4 --encoder-normalize-before --decoder-normalize-before --dropout 0.3 --attention-dropout 0.2 --relu-dropout 0.2 --weight-decay 0.0001 --label-smoothing 0.2 --criterion label_smoothed_cross_entropy --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0 --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-7 --lr 5e-4 --max-tokens 1000 --update-freq 4 --max-epoch 200 --save-interval 20 --keep-best-checkpoints 1 --save-dir /data/rajat/datasets/mt/500/checkpoints --seed 1 --skip-invalid-size-inputs-valid-test
