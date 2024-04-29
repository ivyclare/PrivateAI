# Auditing Code

To run the code:

1. Create Observations (using gradient_canary): 
   `python dp_white_box_auditing.py --dataset_name "cifar10" --num_observations 5000 --epochs_per_observation 80 --lr 0.01 --target_epsilon 8 --gradient_canary`
2. Compute Lower Bounds 
   `python compute_lower_bounds.py --gradient_canary`

Don't use the input canary. But if you want to, here is how:

` python dp_white_box_auditing.py --dataset_name "cifar10" --num_observations 5000 --epochs_per_observation 80 --lr 0.01 --target_epsilon 8 --input_canary`
