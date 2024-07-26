# !/bin/bash
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export NCCL_DEBUG=INFO
export WENET_DIR="src/Group-MoE"
export PYTHONIOENCODING=UTF-8
export PYTHONPATH=$WENET_DIR:$PYTHONPATH
stage=4
stop_stage=4
train_set=train
dir=exp/group
num_nodes=1
node_rank=0
. tools/parse_options.sh || exit 1;
mkdir -p $dir
mkdir -p $dir/log
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  INIT_FILE=$dir/ddp_init
  rm -f $INIT_FILE
  init_method=file://$(readlink -f $INIT_FILE)
  echo "$0: init method is $init_method"
  num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
  dist_backend="nccl"
  world_size=`expr $num_gpus \* $num_nodes`
  echo "total gpus is: $world_size"
  for ((i = 0; i < $num_gpus; ++i)); do
  { 
    data=data
    train_data=$data/train/data.list
    train_config=conf/DLG-8e-dynamic.yaml
    data_type=raw
    dict=data/dict/mix_dict.txt
    bpemodel=data/dict/train_960_unigram5000.model
    checkpoint=
    cmvn=data/train/global_cmvn
    gpu_id=$(echo $CUDA_VISIBLE_DEVICES | cut -d',' -f$[$i+1])
    # Rank of each gpu/process used for knowing whether it is
    # the master of a worker.
    rank=`expr $node_rank \* $num_gpus + $i`
    python $WENET_DIR/wenet/bin/train.py --gpu $gpu_id \
      --config $train_config \
      --data_type $data_type \
      --symbol_table $dict \
      --train_data $train_data \
      --log_dir $dir/log \
      --bpe_model ${bpemodel} \
      --cmvn $cmvn \
      --cv_data data/test/asru-man/data.list \
      --cv_data data/test/clean/data.list \
      --cv_data data/dev/asru-cs/data.list \
      ${checkpoint:+--checkpoint $checkpoint} \
      --model_dir $dir \
      --ddp.init_method $init_method \
      --ddp.world_size $world_size \
      --ddp.rank $rank \
      --ddp.dist_backend $dist_backend \
      --num_workers 4 
  } &
  done
  wait
fi
