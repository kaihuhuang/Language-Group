# !/bin/bash
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export NCCL_DEBUG=INFO
export WENET_DIR="src/Group-MoE"
export PYTHONIOENCODING=UTF-8
export PYTHONPATH=$WENET_DIR:$PYTHONPATH
stage=6
stop_stage=6
dir=exp/group

. tools/parse_options.sh || exit 1;
mkdir -p $dir
mkdir -p $dir/log

export_checkpoint=$dir/avg_5_49.pt

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  # Export the best model you want
  python $WENET_DIR/wenet/bin/export_jit.py \
    --config $dir/train.yaml \
    --checkpoint $export_checkpoint \
    --output_file $dir/final.zip \
    --output_quant_file $dir/final_quant.zip
fi

exit 0