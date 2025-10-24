TRAIN_SIZE=100000
VALIDATION_SIZE=1000
TEST_SIZE=1000

partial_sums=true

for i in {5..15}
do  
    if [ "$partial_sums" = true ]; then
        python3 -m src.data_generator --num_digits "$i" --num_samples "$TEST_SIZE" --fname test --generate_partial_sums
        python3 -m src.data_generator --num_digits "$i" --num_samples "$VALIDATION_SIZE" --fname val --generate_partial_sums --previous_datasets test
        python3 -m src.data_generator --num_digits "$i" --num_samples "$TRAIN_SIZE" --fname train --generate_partial_sums --previous_datasets test val
    else        
        python3 -m src.data_generator --num_digits "$i" --num_samples "$TEST_SIZE" --fname test
        python3 -m src.data_generator --num_digits "$i" --num_samples "$VALIDATION_SIZE" --fname val --previous_datasets test
        python3 -m src.data_generator --num_digits "$i" --num_samples "$TRAIN_SIZE" --fname train --previous_datasets test val
    fi
done