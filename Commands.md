Reanudar
``` bash

python multi_order_cubes/scripts/train_sb3_sac.py \
  --num_envs 1024 --total_timesteps 20000000 \
  --checkpoint logs/sb3/multi_order_cubes_sac/run_20260831_120709/final_sac.zip \
  --learning_starts 4096 --lr_start 1.5e-4 \
  --headless --keep_all_info

```