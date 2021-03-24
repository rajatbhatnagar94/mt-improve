import os
from sklearn.model_selection import train_test_split

SEED=100
base_dir="/data/rajat/datasets/mt/ro-en"

def read(filename):
	f = open(filename, 'r')
	lines = f.read().split('\n')
	return lines

def split(x, src, y, tgt, train_size):
	test_size = 1000
	obj = {}
	obj[f"train.{src}"], obj[f"test.{src}"], obj[f"train.{tgt}"], obj[f"test.{tgt}"] = train_test_split(x, y, random_state=SEED, test_size=test_size * 2, train_size=train_size)
	obj[f"valid.{src}"], obj[f"test.{src}"], obj[f"valid.{tgt}"], obj[f"test.{tgt}"] = train_test_split(obj[f"test.{src}"], obj[f"test.{tgt}"], random_state=SEED, test_size=test_size)

	directory = f"{base_dir}/data/{src}-{tgt}/{train_size}"
	for key in obj:
		write(directory, key, obj[key])

def write(directory, filename, data):
	if not os.path.exists(directory):
		os.makedirs(directory)
	
	f = open(f"{directory}/{filename}", "w")
	f.write("\n".join(data))

def split_start(counts=[10000, 50000, 100000, 300000]):
	input_dir = "/data/rajat/repos/sigmorphon-2020-inflection/data"
	src = read(f"{input_dir}/europarl-v7.ro-en.en")
	tgt = read(f"{input_dir}/europarl-v7.ro-en.ro")
	src_lang = "en"
	tgt_lang = "ro"
	for count in counts:
		split(src, src_lang, tgt, tgt_lang, count)

# main()
