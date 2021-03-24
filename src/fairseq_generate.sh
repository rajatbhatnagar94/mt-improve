count=100000
dir=/data/rajat/datasets/mt/ro-en/data/en-ro/$count
dataset=valid
src=ro
tgt=en
results_dir=$dir/results
sentencepiece_model=$dir/sentencepiece.bpe.model

mkdir -p $results_dir

fairseq-generate $dir/data-bin \
  --path $dir/checkpoints/checkpoint_best.pt \
  --task translation \
  --gen-subset $dataset \
  -t $tgt -s $src \
  --bpe 'sentencepiece' --sentencepiece-model $sentencepiece_model \
  --scoring sacrebleu \
  --batch-size 8 --results-path $results_dir

output_file=$results_dir/$tgt.$dataset
cat $results_dir/generate-$dataset.txt | grep -P "^H" |sort -V |cut -f 3- | sed 's/\[$tgt\]//g' > $output_file.hyp
cat $results_dir/generate-$dataset.txt | grep -P "^T" |sort -V |cut -f 2- | sed 's/\[$tgt\]//g' > $output_file.ref

ref="$dir/$dataset.$tgt"
score=$(sacrebleu -tok 'none' -s 'none' $ref < $output_file.hyp)

echo "$score" > $output_file.score

scores="scores"
mkdir -p $scores
result_text="$count, $score, $dataset, $dir"
echo "$result_text" >> $scores/$dataset-$src-$tgt
