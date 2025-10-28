CKPT_FNAME="$1"

if [ -z "$CKPT_FNAME" ]; then
    CKPT_FNAME=None
fi
export CKPT_FNAME


export DATE=$(date +%Y%m%d_%H%M%S)

if [ "$CKPT_FNAME" != "None" ]; then
    RESULTS_DNAME=$(dirname "$(dirname "$CKPT_FNAME")")
else
    RESULTS_DNAME=results/${D}_by_${D}_mult/mse_output_loss/gpt2_${DATE}
fi
export RESULTS_DNAME


export D=4
export DATA_DNAME=data/${D}_by_${D}
export SEED=2818
export GPU=7

export EPOCHS=55
export LR=5e-5
export BSZ=32
export P_LAMBDA=0
export MSE_LAMBDA=1e3



python3 -u -m src.train \
    --train_fname ${DATA_DNAME}/train.txt \
    --val_fname ${DATA_DNAME}/val.txt \
    --test_fname ${DATA_DNAME}/test.txt \
    --results_dname ${RESULTS_DNAME} \
    --ckpt_fname ${CKPT_FNAME} \
    --gpu_ord ${GPU} \
    --max_size -1 \
    --epochs ${EPOCHS} \
    --lr ${LR} \
    --batch_size ${BSZ} \
    --seed ${SEED} \
    --partial_sums_lambda ${P_LAMBDA} \
    --mse_loss_lambda ${MSE_LAMBDA} \
    2>&1 | tee "${RESULTS_DNAME}/log.train"