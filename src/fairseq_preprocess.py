from fairseq_cli.preprocess import cli_main
import sys
from sentencepiece import SentencePieceTrainer, SentencePieceProcessor
from split import split_start

def sentencepiece_train(data_dir, src, tgt):
	SentencePieceTrainer.train(
		input=[f"{data_dir}/train.{src}", f"{data_dir}/train.{tgt}"],
		character_coverage=1.0,
		model_type='bpe',
		accept_language=[src, tgt],
		model_prefix=f"{data_dir}/sentencepiece.bpe"
	)

def sentencepiece_encode(data_dir, src, tgt):
	model = SentencePieceProcessor(model_file=f"{data_dir}/sentencepiece.bpe.model")
	for split in ['train', 'valid', 'test']:
		for ext in [src, tgt]:
			try:
				with open(f"{data_dir}/{split}.{ext}", 'r') as in_file, open(f"{data_dir}/{split}.spm.{ext}", 'w') as out_file:
					for line in in_file:
						out_file.write(' '.join(model.encode(line, out_type=str)) + '\n')
			except Exception as e:
					print(f"Encode error: {e}")

def main():
	base_dir = "/data/rajat/datasets/mt/ro-en/data/en-ro"
	src, tgt = "ro", "en"
	counts = [75000]
	split_start(counts)
	for count in counts:
		data_dir = f"{base_dir}/{count}"
		sentencepiece_train(data_dir, src, tgt)
		sentencepiece_encode(data_dir, src, tgt)
		sys.argv = [
		'/data/rajat/repos/fairseq/bin/fairseq-preprocess',
		'--source-lang', src, '--target-lang', tgt,
		'--trainpref', f"{data_dir}/train.spm",
		'--validpref', f"{data_dir}/valid.spm",
		'--testpref', f"{data_dir}/test.spm",
		'--destdir', f"{data_dir}/data-bin",
		'--thresholdtgt', '0', '--thresholdsrc', '0',
		'--bpe', 'sentencepiece',
		'--joined-dictionary',
		'--workers', '70'
		]
		cli_main()

main()
