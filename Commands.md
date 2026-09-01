Reanudar
``` bash

python multi_order_cubes/scripts/train_sb3_sac.py \
  --num_envs 1024 --total_timesteps 20000000 \
  --checkpoint logs/sb3/multi_order_cubes_sac/run_20260831_120709/final_sac.zip \
  --learning_starts 4096 --lr_start 1.5e-4 \
  --headless --keep_all_info

```

``` bash

python multi_order_cubes/scripts/play_sb3.py   --checkpoint logs/sb3/multi_order_cubes_sac/run_20260831_14_lift/final_sac.zip   --vecnormalize logs/sb3/multi_order_cubes_sac/run_20260831_14_lift/vecnormalize.pkl   --diag_csv ./multi_order_cubes/assets/data/diagnostics.csv

```



=== Reward by behavioral state (does hovering already pay off?) ===
state         n_rows    pct reach_xy  reach_z grip_readiness grasp_contact object_lifted total_reward
not_trying     55714  69.6%    0.468    0.270    0.022    0.000    0.007    0.767
hovering        3736   4.7%    0.512    0.455    0.034    0.094    0.155    1.250
attempting        18   0.0%    0.494    0.454    0.083    0.000    0.000    1.031
grasping       20532  25.7%    0.501    0.455    0.093    0.428    0.942    2.419

=== Outcome rate by target cube size ===
size          n_rows not_trying   hovering attempting   grasping mean_total
s              28156      69.8%       2.3%       0.1%      27.8%      1.214
m              27503      69.4%       3.2%       0.0%      27.4%      1.213
l              24341      69.7%       9.0%       0.0%      21.2%      1.212

=== Outcome rate by target slot (0=Y+0.3 ... 3=Y-0.3) ===
slot          n_rows not_trying   hovering attempting   grasping mean_total
0              15645      50.1%       5.4%       0.1%      44.4%      1.464
1              17422     100.0%       0.0%       0.0%       0.0%      0.766
2              29062      67.0%       7.0%       0.0%      25.9%      1.301
3              17871      61.5%       4.7%       0.0%      33.8%      1.287

=== Outcome rate by neighbor clearance ("more room to maneuver" hypothesis) ===
bucket                            n_rows    pct reach_xy  reach_z grip_readiness grasp_contact object_lifted total_reward
middle: has empty neighbor         27471  34.3%    0.505    0.366    0.035    0.116    0.273    1.294
middle: boxed in (both occupied)   19013  23.8%    0.509    0.272    0.034    0.000    0.006    0.821
edge slot: neighbor empty           2321   2.9%    0.219    0.097    0.006    0.000    0.007    0.327
edge slot: neighbor occupied       31195  39.0%    0.456    0.340    0.054    0.191    0.406    1.447
Wrote 80000 rows to ./multi_order_cubes/assets/data/diagnostics.csv
