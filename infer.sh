# !/bin/bash
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export NCCL_DEBUG=INFO
export WENET_DIR="src/Group-MoE"
export PYTHONIOENCODING=UTF-8
export PYTHONPATH=$WENET_DIR:$PYTHONPATH
stage=5
stop_stage=5
num_nodes=1
node_rank=0
cmvn=false
train_set=train
# dir=exp/group_moe_pretrain # 预训练共享块阶段
dir=exp/group

. tools/parse_options.sh || exit 1;

mkdir -p $dir
mkdir -p $dir/log

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  # Test model, please specify the model you want to test by --checkpoint
  average_num=5
  average_checkpoint=true
  max_epoch=49
  if [ ${average_checkpoint} == true ]; then
    decode_checkpoint=$dir/avg_${average_num}_$max_epoch.pt
    echo "do model average and final checkpoint is $decode_checkpoint"
    python $WENET_DIR/wenet/bin/average_model.py \
      --dst_model $decode_checkpoint \
      --src_path $dir  \
      --num ${average_num} \
      --max_epoch $max_epoch \
      # --val_best
  fi
  # exit 0
  decode_dir=$dir/decode/${max_epoch}_avg${average_num}
  mkdir -p $decode_dir
  # Please specify decoding_chunk_size for unified streaming and
  # non-streaming model. The default value is -1, which is full chunk
  # for non-streaming inference
  decoding_chunk_size=
  ctc_weight=0.5
  reverse_weight=0.3
  recog_set="asru-cs asru-man clean"
  recog_set="all"
  # modes="attention_rescoring"
  modes="ctc_greedy_search"
  data_type=raw
  dict=data/dict/mix_dict.txt
  data=data
  bpemodel=data/dict/train_960_unigram5000.model
  gpu_id=$(echo $CUDA_VISIBLE_DEVICES | cut -d',' -f$[$i+1])
  for epoch in $max_epoch ;do
    for mode in ${modes}; do
      for rec_set in ${recog_set}; do
        test_dir=$decode_dir/$mode/${rec_set}
        mkdir -p $test_dir
        python $WENET_DIR/wenet/bin/recognize.py --gpu 3 \
          --mode $mode \
          --config $dir/train.yaml \
          --data_type $data_type \
          --test_data ${data}/test/$rec_set/data.list \
          --checkpoint $decode_checkpoint \
          --beam_size 10 \
          --batch_size 16 \
          --penalty 0.0 \
          --dict $dict \
          --ctc_weight $ctc_weight \
          --reverse_weight $reverse_weight \
          --result_file $test_dir/text_bpe \
          ${decoding_chunk_size:+--decoding_chunk_size $decoding_chunk_size}
        
        cut -f2- -d " " $test_dir/text_bpe > $test_dir/text_bpe_value_tmp
        cut -f1 -d " " $test_dir/text_bpe > $test_dir/text_bpe_key_tmp
        tools/spm_decode --model=${bpemodel} --input_format=piece \
          < $test_dir/text_bpe_value_tmp | sed -e "s/▁/ /g" > $test_dir/text_value_tmp
        paste -d " " $test_dir/text_bpe_key_tmp $test_dir/text_value_tmp > $test_dir/text
    
        python tools/compute-wer.py --char=1 --v=1 \
          ${data}/test/${rec_set}/text $test_dir/text > $test_dir/wer
      done
    done
  done
fi
