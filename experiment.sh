export D=4
export FOLDER=data/${D}_by_${D}_mult_no_cot_w_partial_sums
export EPOCHS=45
export LR=5e-5
export BSZ=32
export SEED=2818
export DATE=$(date +%Y%m%d_%H%M%S)
export SAVE=results/${D}_by_${D}_mult/gpt2_${DATE}
export SAVE_CKPTS=results/${D}_by_${D}_mult/gpt2_${DATE}/checkpoints
mkdir -p $SAVE


export GPU=7


python3 -m src.train \
    --train_path ${FOLDER}/train.txt \
    --val_path ${FOLDER}/valid.txt \
    --test_path ${FOLDER}/test_bigbench.txt \
    --gpu_ord ${GPU} \
    --max_size -1 \
    --epochs ${EPOCHS} \
    --lr ${LR} \
    --batch_size ${BSZ} \
    --seed ${SEED} \
    --save_model ${SAVE_CKPTS} \
    --save_config ${SAVE} \
    2>&1 | tee "${SAVE}/log.train"