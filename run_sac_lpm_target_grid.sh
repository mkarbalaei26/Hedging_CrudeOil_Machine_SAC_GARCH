[compute] torch_threads=8 | blas_threads=4
Loaded scenario kinds: ['oracle_universe', 'oracle_all', 'baseline']
Loaded scenario_id examples: ['d7033126641fbae628ab', '8329d03647e71e861793', '8524c70aa57048a0545c', 'b0fcf9cb687bd1f742df', 'ce6abdaa7897eda9c2a7']
==========================================================================================
SAC Portfolio-LPM rolling-window training
asset/exposure:       OPEC / OPEC_BASKET
precompute:           /Users/mohammadkarbalaei/University/Final Project/Final_Code/rl_cache/precompute_OPEC.npz
features:             38
train scenario kinds: ['oracle_universe']
eval scenario kinds:  ['oracle_universe', 'oracle_all', 'baseline']
train/val/test days:  730 / 183 / 183
step days:            183
reward weights:       LPM=0.45 | VOL=0.35 | COST=0.2
reward memory:        LPM=0.70*level+0.30*delta | VOL=running level | COST=cumulative decision cost
output:               /Users/mohammadkarbalaei/University/Final Project/Final_Code/rl_outputs/sac_portfolio_lpm/opec_sac_lpm3aware_ROLL_T1000_train2.0y_val6.0m_test6.0m_hmin0.0_hmax1.25_dh0.05_wLPM0.45_wVOL0.35_wCOST0.2
==========================================================================================
[parallel] enabled | workers=12 | torch_threads/process=8 | blas_threads/process=4
[parallel] windows to run: 28
[compute] torch_threads=8 | blas_threads=4
[compute] torch_threads=8 | blas_threads=4
[compute] torch_threads=8 | blas_threads=4
[compute] torch_threads=8 | blas_threads=4
[compute] torch_threads=8 | blas_threads=4
[compute] torch_threads=8 | blas_threads=4
[compute] torch_threads=8 | blas_threads=4
[compute] torch_threads=8 | blas_threads=4
[compute] torch_threads=8 | blas_threads=4
[compute] torch_threads=8 | blas_threads=4
[compute] torch_threads=8 | blas_threads=4
[compute] torch_threads=8 | blas_threads=4
[parallel window 00] start | train=7,300 val=1,830 test=2,509
[parallel window 01] start | train=7,300 val=1,830 test=2,509
[parallel window 02] start | train=7,300 val=1,830 test=2,520
[parallel window 03] start | train=7,300 val=1,830 test=2,509
[parallel window 04] start | train=7,300 val=1,830 test=2,530
[parallel window 05] start | train=7,300 val=1,830 test=2,529
[parallel window 06] start | train=7,300 val=1,830 test=2,511
[parallel window 07] start | train=7,300 val=1,830 test=2,538
[parallel window 08] start | train=7,300 val=1,830 test=2,508
[parallel window 09] start | train=7,300 val=1,830 test=2,527
[parallel window 10] start | train=7,300 val=1,830 test=2,538
[parallel window 11] start | train=7,300 val=1,830 test=2,503
Using cpu device
Logging to /Users/mohammadkarbalaei/University/Final Project/Final_Code/rl_outputs/sac_portfolio_lpm/opec_sac_lpm3aware_ROLL_T1000_train2.0y_val6.0m_test6.0m_hmin0.0_hmax1.25_dh0.05_wLPM0.45_wVOL0.35_wCOST0.2/window_07/tb/SAC_3
Using cpu device
Logging to /Users/mohammadkarbalaei/University/Final Project/Final_Code/rl_outputs/sac_portfolio_lpm/opec_sac_lpm3aware_ROLL_T1000_train2.0y_val6.0m_test6.0m_hmin0.0_hmax1.25_dh0.05_wLPM0.45_wVOL0.35_wCOST0.2/window_00/tb/SAC_3
Using cpu device
Logging to /Users/mohammadkarbalaei/University/Final Project/Final_Code/rl_outputs/sac_portfolio_lpm/opec_sac_lpm3aware_ROLL_T1000_train2.0y_val6.0m_test6.0m_hmin0.0_hmax1.25_dh0.05_wLPM0.45_wVOL0.35_wCOST0.2/window_02/tb/SAC_3
Using cpu device
Logging to /Users/mohammadkarbalaei/University/Final Project/Final_Code/rl_outputs/sac_portfolio_lpm/opec_sac_lpm3aware_ROLL_T1000_train2.0y_val6.0m_test6.0m_hmin0.0_hmax1.25_dh0.05_wLPM0.45_wVOL0.35_wCOST0.2/window_06/tb/SAC_3
Using cpu device
Logging to /Users/mohammadkarbalaei/University/Final Project/Final_Code/rl_outputs/sac_portfolio_lpm/opec_sac_lpm3aware_ROLL_T1000_train2.0y_val6.0m_test6.0m_hmin0.0_hmax1.25_dh0.05_wLPM0.45_wVOL0.35_wCOST0.2/window_05/tb/SAC_3
Using cpu device
Logging to /Users/mohammadkarbalaei/University/Final Project/Final_Code/rl_outputs/sac_portfolio_lpm/opec_sac_lpm3aware_ROLL_T1000_train2.0y_val6.0m_test6.0m_hmin0.0_hmax1.25_dh0.05_wLPM0.45_wVOL0.35_wCOST0.2/window_01/tb/SAC_3
Using cpu device
Logging to /Users/mohammadkarbalaei/University/Final Project/Final_Code/rl_outputs/sac_portfolio_lpm/opec_sac_lpm3aware_ROLL_T1000_train2.0y_val6.0m_test6.0m_hmin0.0_hmax1.25_dh0.05_wLPM0.45_wVOL0.35_wCOST0.2/window_04/tb/SAC_3
Using cpu device
Logging to /Users/mohammadkarbalaei/University/Final Project/Final_Code/rl_outputs/sac_portfolio_lpm/opec_sac_lpm3aware_ROLL_T1000_train2.0y_val6.0m_test6.0m_hmin0.0_hmax1.25_dh0.05_wLPM0.45_wVOL0.35_wCOST0.2/window_03/tb/SAC_3
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 18.2     |
|    ep_rew_mean     | -8.36    |
| time/              |          |
|    episodes        | 4        |
|    fps             | 6513     |
|    time_elapsed    | 0        |
|    total_timesteps | 73       |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 18.5     |
|    ep_rew_mean     | -3.08    |
| time/              |          |
|    episodes        | 4        |
|    fps             | 7133     |
|    time_elapsed    | 0        |
|    total_timesteps | 74       |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 17       |
|    ep_rew_mean     | -2.63    |
| time/              |          |
|    episodes        | 4        |
|    fps             | 6503     |
|    time_elapsed    | 0        |
|    total_timesteps | 68       |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 22       |
|    ep_rew_mean     | -7.46    |
| time/              |          |
|    episodes        | 4        |
|    fps             | 6925     |
|    time_elapsed    | 0        |
|    total_timesteps | 88       |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 23.5     |
|    ep_rew_mean     | -3.58    |
| time/              |          |
|    episodes        | 4        |
|    fps             | 6825     |
|    time_elapsed    | 0        |
|    total_timesteps | 94       |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 21.8     |
|    ep_rew_mean     | -4.47    |
| time/              |          |
|    episodes        | 4        |
|    fps             | 6715     |
|    time_elapsed    | 0        |
|    total_timesteps | 87       |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 19.8     |
|    ep_rew_mean     | -1.58    |
| time/              |          |
|    episodes        | 4        |
|    fps             | 4515     |
|    time_elapsed    | 0        |
|    total_timesteps | 79       |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 21.8     |
|    ep_rew_mean     | -4.75    |
| time/              |          |
|    episodes        | 4        |
|    fps             | 5969     |
|    time_elapsed    | 0        |
|    total_timesteps | 87       |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 18       |
|    ep_rew_mean     | -4.66    |
| time/              |          |
|    episodes        | 8        |
|    fps             | 7343     |
|    time_elapsed    | 0        |
|    total_timesteps | 144      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 19.2     |
|    ep_rew_mean     | -12.6    |
| time/              |          |
|    episodes        | 8        |
|    fps             | 7452     |
|    time_elapsed    | 0        |
|    total_timesteps | 154      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 20.5     |
|    ep_rew_mean     | -5.89    |
| time/              |          |
|    episodes        | 8        |
|    fps             | 7172     |
|    time_elapsed    | 0        |
|    total_timesteps | 164      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 18.1     |
|    ep_rew_mean     | -8.66    |
| time/              |          |
|    episodes        | 8        |
|    fps             | 5624     |
|    time_elapsed    | 0        |
|    total_timesteps | 145      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 22.8     |
|    ep_rew_mean     | -4.42    |
| time/              |          |
|    episodes        | 8        |
|    fps             | 6934     |
|    time_elapsed    | 0        |
|    total_timesteps | 182      |
---------------------------------
Using cpu device
Logging to /Users/mohammadkarbalaei/University/Final Project/Final_Code/rl_outputs/sac_portfolio_lpm/opec_sac_lpm3aware_ROLL_T1000_train2.0y_val6.0m_test6.0m_hmin0.0_hmax1.25_dh0.05_wLPM0.45_wVOL0.35_wCOST0.2/window_09/tb/SAC_3
Using cpu device
Logging to /Users/mohammadkarbalaei/University/Final Project/Final_Code/rl_outputs/sac_portfolio_lpm/opec_sac_lpm3aware_ROLL_T1000_train2.0y_val6.0m_test6.0m_hmin0.0_hmax1.25_dh0.05_wLPM0.45_wVOL0.35_wCOST0.2/window_10/tb/SAC_3
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 20.2     |
|    ep_rew_mean     | -2.3     |
| time/              |          |
|    episodes        | 8        |
|    fps             | 5600     |
|    time_elapsed    | 0        |
|    total_timesteps | 162      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 21.6     |
|    ep_rew_mean     | -3.85    |
| time/              |          |
|    episodes        | 8        |
|    fps             | 6778     |
|    time_elapsed    | 0        |
|    total_timesteps | 173      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 18.1     |
|    ep_rew_mean     | -3.95    |
| time/              |          |
|    episodes        | 12       |
|    fps             | 7439     |
|    time_elapsed    | 0        |
|    total_timesteps | 217      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 22.8     |
|    ep_rew_mean     | -4.6     |
| time/              |          |
|    episodes        | 8        |
|    fps             | 6496     |
|    time_elapsed    | 0        |
|    total_timesteps | 182      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 20.1     |
|    ep_rew_mean     | -6.48    |
| time/              |          |
|    episodes        | 12       |
|    fps             | 7440     |
|    time_elapsed    | 0        |
|    total_timesteps | 241      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 18.8     |
|    ep_rew_mean     | -9.09    |
| time/              |          |
|    episodes        | 12       |
|    fps             | 6929     |
|    time_elapsed    | 0        |
|    total_timesteps | 225      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 18.1     |
|    ep_rew_mean     | -6.69    |
| time/              |          |
|    episodes        | 12       |
|    fps             | 6019     |
|    time_elapsed    | 0        |
|    total_timesteps | 217      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 20.8     |
|    ep_rew_mean     | -3.77    |
| time/              |          |
|    episodes        | 12       |
|    fps             | 6782     |
|    time_elapsed    | 0        |
|    total_timesteps | 250      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 17.9     |
|    ep_rew_mean     | -3.56    |
| time/              |          |
|    episodes        | 16       |
|    fps             | 7595     |
|    time_elapsed    | 0        |
|    total_timesteps | 286      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 19.8     |
|    ep_rew_mean     | -2.57    |
| time/              |          |
|    episodes        | 12       |
|    fps             | 6236     |
|    time_elapsed    | 0        |
|    total_timesteps | 238      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 21.2     |
|    ep_rew_mean     | -2.42    |
| time/              |          |
|    episodes        | 4        |
|    fps             | 7003     |
|    time_elapsed    | 0        |
|    total_timesteps | 85       |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 22.2     |
|    ep_rew_mean     | -4.87    |
| time/              |          |
|    episodes        | 12       |
|    fps             | 7304     |
|    time_elapsed    | 0        |
|    total_timesteps | 266      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 23.5     |
|    ep_rew_mean     | -3.21    |
| time/              |          |
|    episodes        | 4        |
|    fps             | 6206     |
|    time_elapsed    | 0        |
|    total_timesteps | 94       |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 22       |
|    ep_rew_mean     | -4.44    |
| time/              |          |
|    episodes        | 12       |
|    fps             | 6358     |
|    time_elapsed    | 0        |
|    total_timesteps | 264      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 18.9     |
|    ep_rew_mean     | -7.63    |
| time/              |          |
|    episodes        | 16       |
|    fps             | 6779     |
|    time_elapsed    | 0        |
|    total_timesteps | 302      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 20.6     |
|    ep_rew_mean     | -5.62    |
| time/              |          |
|    episodes        | 16       |
|    fps             | 7204     |
|    time_elapsed    | 0        |
|    total_timesteps | 330      |
---------------------------------
Using cpu device
Logging to /Users/mohammadkarbalaei/University/Final Project/Final_Code/rl_outputs/sac_portfolio_lpm/opec_sac_lpm3aware_ROLL_T1000_train2.0y_val6.0m_test6.0m_hmin0.0_hmax1.25_dh0.05_wLPM0.45_wVOL0.35_wCOST0.2/window_08/tb/SAC_3
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 21.4     |
|    ep_rew_mean     | -3.67    |
| time/              |          |
|    episodes        | 16       |
|    fps             | 7234     |
|    time_elapsed    | 0        |
|    total_timesteps | 342      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 18.5     |
|    ep_rew_mean     | -6.79    |
| time/              |          |
|    episodes        | 16       |
|    fps             | 6176     |
|    time_elapsed    | 0        |
|    total_timesteps | 296      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 18.8     |
|    ep_rew_mean     | -2.53    |
| time/              |          |
|    episodes        | 8        |
|    fps             | 7301     |
|    time_elapsed    | 0        |
|    total_timesteps | 150      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 21.3     |
|    ep_rew_mean     | -4.14    |
| time/              |          |
|    episodes        | 16       |
|    fps             | 7540     |
|    time_elapsed    | 0        |
|    total_timesteps | 341      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 20.4     |
|    ep_rew_mean     | -2.79    |
| time/              |          |
|    episodes        | 16       |
|    fps             | 6657     |
|    time_elapsed    | 0        |
|    total_timesteps | 327      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 20.9     |
|    ep_rew_mean     | -2.48    |
| time/              |          |
|    episodes        | 8        |
|    fps             | 6865     |
|    time_elapsed    | 0        |
|    total_timesteps | 167      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 18.4     |
|    ep_rew_mean     | -5.17    |
| time/              |          |
|    episodes        | 20       |
|    fps             | 6879     |
|    time_elapsed    | 0        |
|    total_timesteps | 367      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 22.2     |
|    ep_rew_mean     | -4.55    |
| time/              |          |
|    episodes        | 16       |
|    fps             | 6797     |
|    time_elapsed    | 0        |
|    total_timesteps | 355      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 19.9     |
|    ep_rew_mean     | -7.64    |
| time/              |          |
|    episodes        | 20       |
|    fps             | 7097     |
|    time_elapsed    | 0        |
|    total_timesteps | 399      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 21.2     |
|    ep_rew_mean     | -3.82    |
| time/              |          |
|    episodes        | 20       |
|    fps             | 7500     |
|    time_elapsed    | 0        |
|    total_timesteps | 425      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 21       |
|    ep_rew_mean     | -4.06    |
| time/              |          |
|    episodes        | 4        |
|    fps             | 6835     |
|    time_elapsed    | 0        |
|    total_timesteps | 84       |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 20.5     |
|    ep_rew_mean     | -4.97    |
| time/              |          |
|    episodes        | 20       |
|    fps             | 7088     |
|    time_elapsed    | 0        |
|    total_timesteps | 410      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 19.2     |
|    ep_rew_mean     | -2.42    |
| time/              |          |
|    episodes        | 12       |
|    fps             | 7512     |
|    time_elapsed    | 0        |
|    total_timesteps | 230      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 19.1     |
|    ep_rew_mean     | -5.82    |
| time/              |          |
|    episodes        | 20       |
|    fps             | 6398     |
|    time_elapsed    | 0        |
|    total_timesteps | 383      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 20.1     |
|    ep_rew_mean     | -2.85    |
| time/              |          |
|    episodes        | 20       |
|    fps             | 6818     |
|    time_elapsed    | 0        |
|    total_timesteps | 403      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 20.2     |
|    ep_rew_mean     | -2.39    |
| time/              |          |
|    episodes        | 12       |
|    fps             | 6457     |
|    time_elapsed    | 0        |
|    total_timesteps | 242      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 20.4     |
|    ep_rew_mean     | -3.4     |
| time/              |          |
|    episodes        | 24       |
|    fps             | 7643     |
|    time_elapsed    | 0        |
|    total_timesteps | 489      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 21.3     |
|    ep_rew_mean     | -4.15    |
| time/              |          |
|    episodes        | 20       |
|    fps             | 6974     |
|    time_elapsed    | 0        |
|    total_timesteps | 426      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 21.9     |
|    ep_rew_mean     | -4.41    |
| time/              |          |
|    episodes        | 20       |
|    fps             | 7365     |
|    time_elapsed    | 0        |
|    total_timesteps | 439      |
---------------------------------
Using cpu device
Logging to /Users/mohammadkarbalaei/University/Final Project/Final_Code/rl_outputs/sac_portfolio_lpm/opec_sac_lpm3aware_ROLL_T1000_train2.0y_val6.0m_test6.0m_hmin0.0_hmax1.25_dh0.05_wLPM0.45_wVOL0.35_wCOST0.2/window_11/tb/SAC_3
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 19.1     |
|    ep_rew_mean     | -9.46    |
| time/              |          |
|    episodes        | 24       |
|    fps             | 7118     |
|    time_elapsed    | 0        |
|    total_timesteps | 458      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 20.4     |
|    ep_rew_mean     | -4.97    |
| time/              |          |
|    episodes        | 24       |
|    fps             | 7233     |
|    time_elapsed    | 0        |
|    total_timesteps | 490      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 20.5     |
|    ep_rew_mean     | -7.05    |
| time/              |          |
|    episodes        | 24       |
|    fps             | 7299     |
|    time_elapsed    | 0        |
|    total_timesteps | 493      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 20       |
|    ep_rew_mean     | -2.75    |
| time/              |          |
|    episodes        | 24       |
|    fps             | 7066     |
|    time_elapsed    | 0        |
|    total_timesteps | 481      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 19.9     |
|    ep_rew_mean     | -2.43    |
| time/              |          |
|    episodes        | 16       |
|    fps             | 7614     |
|    time_elapsed    | 0        |
|    total_timesteps | 318      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 18.8     |
|    ep_rew_mean     | -5.07    |
| time/              |          |
|    episodes        | 24       |
|    fps             | 6373     |
|    time_elapsed    | 0        |
|    total_timesteps | 452      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 20.6     |
|    ep_rew_mean     | -11.2    |
| time/              |          |
|    episodes        | 8        |
|    fps             | 6327     |
|    time_elapsed    | 0        |
|    total_timesteps | 165      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 20.4     |
|    ep_rew_mean     | -3.69    |
| time/              |          |
|    episodes        | 28       |
|    fps             | 7759     |
|    time_elapsed    | 0        |
|    total_timesteps | 571      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 20.9     |
|    ep_rew_mean     | -2.27    |
| time/              |          |
|    episodes        | 16       |
|    fps             | 6866     |
|    time_elapsed    | 0        |
|    total_timesteps | 335      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 21.9     |
|    ep_rew_mean     | -4.18    |
| time/              |          |
|    episodes        | 24       |
|    fps             | 7229     |
|    time_elapsed    | 0        |
|    total_timesteps | 525      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 22.2     |
|    ep_rew_mean     | -4.55    |
| time/              |          |
|    episodes        | 24       |
|    fps             | 7485     |
|    time_elapsed    | 0        |
|    total_timesteps | 533      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 20.2     |
|    ep_rew_mean     | -6.44    |
| time/              |          |
|    episodes        | 28       |
|    fps             | 7467     |
|    time_elapsed    | 0        |
|    total_timesteps | 567      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 19       |
|    ep_rew_mean     | -9.33    |
| time/              |          |
|    episodes        | 28       |
|    fps             | 7040     |
|    time_elapsed    | 0        |
|    total_timesteps | 532      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 18.3     |
|    ep_rew_mean     | -5.24    |
| time/              |          |
|    episodes        | 28       |
|    fps             | 6575     |
|    time_elapsed    | 0        |
|    total_timesteps | 512      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 18.5     |
|    ep_rew_mean     | -1.86    |
| time/              |          |
|    episodes        | 4        |
|    fps             | 5067     |
|    time_elapsed    | 0        |
|    total_timesteps | 74       |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 20.4     |
|    ep_rew_mean     | -4.77    |
| time/              |          |
|    episodes        | 28       |
|    fps             | 7225     |
|    time_elapsed    | 0        |
|    total_timesteps | 570      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 20.6     |
|    ep_rew_mean     | -2.89    |
| time/              |          |
|    episodes        | 28       |
|    fps             | 7316     |
|    time_elapsed    | 0        |
|    total_timesteps | 578      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 19.5     |
|    ep_rew_mean     | -2.14    |
| time/              |          |
|    episodes        | 20       |
|    fps             | 7409     |
|    time_elapsed    | 0        |
|    total_timesteps | 390      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 20.2     |
|    ep_rew_mean     | -10.7    |
| time/              |          |
|    episodes        | 12       |
|    fps             | 6368     |
|    time_elapsed    | 0        |
|    total_timesteps | 243      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 20.2     |
|    ep_rew_mean     | -3.48    |
| time/              |          |
|    episodes        | 32       |
|    fps             | 7756     |
|    time_elapsed    | 0        |
|    total_timesteps | 648      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 18.8     |
|    ep_rew_mean     | -8.63    |
| time/              |          |
|    episodes        | 32       |
|    fps             | 7188     |
|    time_elapsed    | 0        |
|    total_timesteps | 603      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 20.9     |
|    ep_rew_mean     | -2.57    |
| time/              |          |
|    episodes        | 20       |
|    fps             | 7077     |
|    time_elapsed    | 0        |
|    total_timesteps | 419      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 21.9     |
|    ep_rew_mean     | -4.27    |
| time/              |          |
|    episodes        | 28       |
|    fps             | 7582     |
|    time_elapsed    | 0        |
|    total_timesteps | 614      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 21.8     |
|    ep_rew_mean     | -4.08    |
| time/              |          |
|    episodes        | 28       |
|    fps             | 7272     |
|    time_elapsed    | 0        |
|    total_timesteps | 611      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 18.2     |
|    ep_rew_mean     | -5.09    |
| time/              |          |
|    episodes        | 32       |
|    fps             | 6616     |
|    time_elapsed    | 0        |
|    total_timesteps | 584      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 20.8     |
|    ep_rew_mean     | -2.47    |
| time/              |          |
|    episodes        | 8        |
|    fps             | 6531     |
|    time_elapsed    | 0        |
|    total_timesteps | 166      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 20.4     |
|    ep_rew_mean     | -4.45    |
| time/              |          |
|    episodes        | 32       |
|    fps             | 7287     |
|    time_elapsed    | 0        |
|    total_timesteps | 654      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 20.4     |
|    ep_rew_mean     | -6.99    |
| time/              |          |
|    episodes        | 32       |
|    fps             | 7310     |
|    time_elapsed    | 0        |
|    total_timesteps | 654      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 19.8     |
|    ep_rew_mean     | -2.16    |
| time/              |          |
|    episodes        | 24       |
|    fps             | 7377     |
|    time_elapsed    | 0        |
|    total_timesteps | 474      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 20.8     |
|    ep_rew_mean     | -2.9     |
| time/              |          |
|    episodes        | 32       |
|    fps             | 7274     |
|    time_elapsed    | 0        |
|    total_timesteps | 667      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 20.3     |
|    ep_rew_mean     | -3.6     |
| time/              |          |
|    episodes        | 36       |
|    fps             | 7798     |
|    time_elapsed    | 0        |
|    total_timesteps | 732      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 18.6     |
|    ep_rew_mean     | -8.06    |
| time/              |          |
|    episodes        | 36       |
|    fps             | 7174     |
|    time_elapsed    | 0        |
|    total_timesteps | 669      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 21.7     |
|    ep_rew_mean     | -4.1     |
| time/              |          |
|    episodes        | 32       |
|    fps             | 7718     |
|    time_elapsed    | 0        |
|    total_timesteps | 695      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 21.1     |
|    ep_rew_mean     | -2.57    |
| time/              |          |
|    episodes        | 24       |
|    fps             | 7151     |
|    time_elapsed    | 0        |
|    total_timesteps | 506      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 21.8     |
|    ep_rew_mean     | -4.36    |
| time/              |          |
|    episodes        | 32       |
|    fps             | 7408     |
|    time_elapsed    | 0        |
|    total_timesteps | 697      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 19.7     |
|    ep_rew_mean     | -1.9     |
| time/              |          |
|    episodes        | 12       |
|    fps             | 6891     |
|    time_elapsed    | 0        |
|    total_timesteps | 236      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 20.2     |
|    ep_rew_mean     | -4.56    |
| time/              |          |
|    episodes        | 36       |
|    fps             | 7369     |
|    time_elapsed    | 0        |
|    total_timesteps | 729      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 21       |
|    ep_rew_mean     | -8.62    |
| time/              |          |
|    episodes        | 16       |
|    fps             | 6247     |
|    time_elapsed    | 0        |
|    total_timesteps | 336      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 20.2     |
|    ep_rew_mean     | -7.9     |
| time/              |          |
|    episodes        | 36       |
|    fps             | 7290     |
|    time_elapsed    | 0        |
|    total_timesteps | 728      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 20.6     |
|    ep_rew_mean     | -2.89    |
| time/              |          |
|    episodes        | 36       |
|    fps             | 7403     |
|    time_elapsed    | 0        |
|    total_timesteps | 743      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 20.3     |
|    ep_rew_mean     | -3.61    |
| time/              |          |
|    episodes        | 40       |
|    fps             | 7882     |
|    time_elapsed    | 0        |
|    total_timesteps | 812      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 18.5     |
|    ep_rew_mean     | -6.24    |
| time/              |          |
|    episodes        | 36       |
|    fps             | 6396     |
|    time_elapsed    | 0        |
|    total_timesteps | 665      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 20       |
|    ep_rew_mean     | -2.09    |
| time/              |          |
|    episodes        | 28       |
|    fps             | 7406     |
|    time_elapsed    | 0        |
|    total_timesteps | 561      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 21.3     |
|    ep_rew_mean     | -4.25    |
| time/              |          |
|    episodes        | 36       |
|    fps             | 7512     |
|    time_elapsed    | 0        |
|    total_timesteps | 766      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 19.8     |
|    ep_rew_mean     | -4.35    |
| time/              |          |
|    episodes        | 40       |
|    fps             | 7416     |
|    time_elapsed    | 0        |
|    total_timesteps | 793      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 21.8     |
|    ep_rew_mean     | -4.02    |
| time/              |          |
|    episodes        | 36       |
|    fps             | 7703     |
|    time_elapsed    | 0        |
|    total_timesteps | 785      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 19.9     |
|    ep_rew_mean     | -1.92    |
| time/              |          |
|    episodes        | 16       |
|    fps             | 7319     |
|    time_elapsed    | 0        |
|    total_timesteps | 318      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 20.6     |
|    ep_rew_mean     | -7.44    |
| time/              |          |
|    episodes        | 20       |
|    fps             | 6563     |
|    time_elapsed    | 0        |
|    total_timesteps | 411      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 21.5     |
|    ep_rew_mean     | -2.52    |
| time/              |          |
|    episodes        | 28       |
|    fps             | 7266     |
|    time_elapsed    | 0        |
|    total_timesteps | 601      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 18.8     |
|    ep_rew_mean     | -8.78    |
| time/              |          |
|    episodes        | 40       |
|    fps             | 6862     |
|    time_elapsed    | 0        |
|    total_timesteps | 751      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 19.9     |
|    ep_rew_mean     | -7.47    |
| time/              |          |
|    episodes        | 40       |
|    fps             | 7101     |
|    time_elapsed    | 0        |
|    total_timesteps | 795      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 20.3     |
|    ep_rew_mean     | -2.27    |
| time/              |          |
|    episodes        | 32       |
|    fps             | 7585     |
|    time_elapsed    | 0        |
|    total_timesteps | 651      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 18.6     |
|    ep_rew_mean     | -6.08    |
| time/              |          |
|    episodes        | 40       |
|    fps             | 6441     |
|    time_elapsed    | 0        |
|    total_timesteps | 743      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 20.3     |
|    ep_rew_mean     | -6.58    |
| time/              |          |
|    episodes        | 24       |
|    fps             | 6825     |
|    time_elapsed    | 0        |
|    total_timesteps | 487      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 21.8     |
|    ep_rew_mean     | -3.85    |
| time/              |          |
|    episodes        | 40       |
|    fps             | 7760     |
|    time_elapsed    | 0        |
|    total_timesteps | 870      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 20       |
|    ep_rew_mean     | -1.8     |
| time/              |          |
|    episodes        | 20       |
|    fps             | 7437     |
|    time_elapsed    | 0        |
|    total_timesteps | 400      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 21.4     |
|    ep_rew_mean     | -4.27    |
| time/              |          |
|    episodes        | 40       |
|    fps             | 7507     |
|    time_elapsed    | 0        |
|    total_timesteps | 858      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 20.6     |
|    ep_rew_mean     | -2.87    |
| time/              |          |
|    episodes        | 40       |
|    fps             | 7040     |
|    time_elapsed    | 0        |
|    total_timesteps | 823      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 21.5     |
|    ep_rew_mean     | -2.47    |
| time/              |          |
|    episodes        | 32       |
|    fps             | 7420     |
|    time_elapsed    | 0        |
|    total_timesteps | 687      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 18.7     |
|    ep_rew_mean     | -8.14    |
| time/              |          |
|    episodes        | 44       |
|    fps             | 6933     |
|    time_elapsed    | 0        |
|    total_timesteps | 822      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 20.2     |
|    ep_rew_mean     | -3.53    |
| time/              |          |
|    episodes        | 44       |
|    fps             | 7250     |
|    time_elapsed    | 0        |
|    total_timesteps | 889      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 20.2     |
|    ep_rew_mean     | -7.02    |
| time/              |          |
|    episodes        | 44       |
|    fps             | 7168     |
|    time_elapsed    | 0        |
|    total_timesteps | 887      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 18.8     |
|    ep_rew_mean     | -5.81    |
| time/              |          |
|    episodes        | 44       |
|    fps             | 6585     |
|    time_elapsed    | 0        |
|    total_timesteps | 825      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 19.9     |
|    ep_rew_mean     | -4.47    |
| time/              |          |
|    episodes        | 44       |
|    fps             | 6979     |
|    time_elapsed    | 0        |
|    total_timesteps | 874      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 20.5     |
|    ep_rew_mean     | -2.31    |
| time/              |          |
|    episodes        | 36       |
|    fps             | 7577     |
|    time_elapsed    | 0        |
|    total_timesteps | 739      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 19.5     |
|    ep_rew_mean     | -1.77    |
| time/              |          |
|    episodes        | 24       |
|    fps             | 7518     |
|    time_elapsed    | 0        |
|    total_timesteps | 469      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 21       |
|    ep_rew_mean     | -4.02    |
| time/              |          |
|    episodes        | 44       |
|    fps             | 7543     |
|    time_elapsed    | 0        |
|    total_timesteps | 926      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 20.3     |
|    ep_rew_mean     | -5.98    |
| time/              |          |
|    episodes        | 28       |
|    fps             | 7006     |
|    time_elapsed    | 0        |
|    total_timesteps | 569      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 21.2     |
|    ep_rew_mean     | -2.35    |
| time/              |          |
|    episodes        | 36       |
|    fps             | 7540     |
|    time_elapsed    | 0        |
|    total_timesteps | 763      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 20.5     |
|    ep_rew_mean     | -2.93    |
| time/              |          |
|    episodes        | 44       |
|    fps             | 7112     |
|    time_elapsed    | 0        |
|    total_timesteps | 902      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 18.7     |
|    ep_rew_mean     | -7.65    |
| time/              |          |
|    episodes        | 48       |
|    fps             | 7065     |
|    time_elapsed    | 0        |
|    total_timesteps | 896      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 21.7     |
|    ep_rew_mean     | -3.76    |
| time/              |          |
|    episodes        | 44       |
|    fps             | 7563     |
|    time_elapsed    | 0        |
|    total_timesteps | 953      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 20.7     |
|    ep_rew_mean     | -3.93    |
| time/              |          |
|    episodes        | 48       |
|    fps             | 7606     |
|    time_elapsed    | 0        |
|    total_timesteps | 995      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 20.1     |
|    ep_rew_mean     | -6.68    |
| time/              |          |
|    episodes        | 48       |
|    fps             | 7214     |
|    time_elapsed    | 0        |
|    total_timesteps | 967      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 20.4     |
|    ep_rew_mean     | -3.48    |
| time/              |          |
|    episodes        | 48       |
|    fps             | 7295     |
|    time_elapsed    | 0        |
|    total_timesteps | 981      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 18.9     |
|    ep_rew_mean     | -7.88    |
| time/              |          |
|    episodes        | 48       |
|    fps             | 6697     |
|    time_elapsed    | 0        |
|    total_timesteps | 908      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 20.3     |
|    ep_rew_mean     | -6.74    |
| time/              |          |
|    episodes        | 32       |
|    fps             | 7065     |
|    time_elapsed    | 0        |
|    total_timesteps | 651      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 19.5     |
|    ep_rew_mean     | -1.73    |
| time/              |          |
|    episodes        | 28       |
|    fps             | 7447     |
|    time_elapsed    | 0        |
|    total_timesteps | 547      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 20.8     |
|    ep_rew_mean     | -2.26    |
| time/              |          |
|    episodes        | 40       |
|    fps             | 7613     |
|    time_elapsed    | 0        |
|    total_timesteps | 830      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 20.2     |
|    ep_rew_mean     | -4.61    |
| time/              |          |
|    episodes        | 48       |
|    fps             | 7031     |
|    time_elapsed    | 0        |
|    total_timesteps | 969      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 19.1     |
|    ep_rew_mean     | -7.63    |
| time/              |          |
|    episodes        | 52       |
|    fps             | 7226     |
|    time_elapsed    | 0        |
|    total_timesteps | 994      |
---------------------------------
 100% ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1,000/1,000  [ 0:00:00 < 0:00:00 , 7,568 it/s ]
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 20.8     |
|    ep_rew_mean     | -3.36    |
| time/              |          |
|    episodes        | 48       |
|    fps             | 7222     |
|    time_elapsed    | 0        |
|    total_timesteps | 998      |
---------------------------------
 100% ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1,000/1,000  [ 0:00:00 < 0:00:00 , 7,240 it/s ] 100% ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1,000/1,000  [ 0:00:00 < 0:00:00 , 7,249 it/s ] 100% ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1,000/1,000  [ 0:00:00 < 0:00:00 , 7,645 it/s ]


 100% ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1,000/1,000  [ 0:00:00 < 0:00:00 , 7,285 it/s ]
 100% ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1,000/1,000  [ 0:00:00 < 0:00:00 , 7,602 it/s ]
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 21.1     |
|    ep_rew_mean     | -2.25    |
| time/              |          |
|    episodes        | 40       |
|    fps             | 7320     |
|    time_elapsed    | 0        |
|    total_timesteps | 844      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 18.8     |
|    ep_rew_mean     | -7.95    |
| time/              |          |
|    episodes        | 52       |
|    fps             | 6798     |
|    time_elapsed    | 0        |
|    total_timesteps | 976      |
---------------------------------
 100% ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1,000/1,000  [ 0:00:00 < 0:00:00 , 6,995 it/s ]
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 20.3     |
|    ep_rew_mean     | -7.39    |
| time/              |          |
|    episodes        | 36       |
|    fps             | 7167     |
|    time_elapsed    | 0        |
|    total_timesteps | 730      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 19.6     |
|    ep_rew_mean     | -1.7     |
| time/              |          |
|    episodes        | 32       |
|    fps             | 7498     |
|    time_elapsed    | 0        |
|    total_timesteps | 628      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 20.7     |
|    ep_rew_mean     | -2.16    |
| time/              |          |
|    episodes        | 44       |
|    fps             | 7616     |
|    time_elapsed    | 0        |
|    total_timesteps | 912      |
---------------------------------
 100% ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1,000/1,000  [ 0:00:00 < 0:00:00 , 6,791 it/s ]
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 21.2     |
|    ep_rew_mean     | -2.4     |
| time/              |          |
|    episodes        | 44       |
|    fps             | 7398     |
|    time_elapsed    | 0        |
|    total_timesteps | 932      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 20.4     |
|    ep_rew_mean     | -6.88    |
| time/              |          |
|    episodes        | 40       |
|    fps             | 7121     |
|    time_elapsed    | 0        |
|    total_timesteps | 815      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 20.1     |
|    ep_rew_mean     | -1.79    |
| time/              |          |
|    episodes        | 36       |
|    fps             | 7500     |
|    time_elapsed    | 0        |
|    total_timesteps | 724      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 20.7     |
|    ep_rew_mean     | -2.07    |
| time/              |          |
|    episodes        | 48       |
|    fps             | 7485     |
|    time_elapsed    | 0        |
|    total_timesteps | 995      |
---------------------------------
 100% ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1,000/1,000  [ 0:00:00 < 0:00:00 , 7,502 it/s ]
 100% ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1,000/1,000  [ 0:00:00 < 0:00:00 , 7,453 it/s ]
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 20.5     |
|    ep_rew_mean     | -6.95    |
| time/              |          |
|    episodes        | 44       |
|    fps             | 7205     |
|    time_elapsed    | 0        |
|    total_timesteps | 901      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 20       |
|    ep_rew_mean     | -1.81    |
| time/              |          |
|    episodes        | 40       |
|    fps             | 7448     |
|    time_elapsed    | 0        |
|    total_timesteps | 799      |
---------------------------------
[eval val w04] 0/1,830 (  0.0%) | elapsed=0s | eta=0s | starting evaluation
[eval val w05] 0/1,830 (  0.0%) | elapsed=0s | eta=0s | starting evaluation
[eval val w02] 0/1,830 (  0.0%) | elapsed=0s | eta=0s | starting evaluation
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 20.3     |
|    ep_rew_mean     | -6.52    |
| time/              |          |
|    episodes        | 48       |
|    fps             | 7252     |
|    time_elapsed    | 0        |
|    total_timesteps | 976      |
---------------------------------
[eval val w01] 0/1,830 (  0.0%) | elapsed=0s | eta=0s | starting evaluation
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 20.2     |
|    ep_rew_mean     | -1.85    |
| time/              |          |
|    episodes        | 44       |
|    fps             | 7472     |
|    time_elapsed    | 0        |
|    total_timesteps | 887      |
---------------------------------
 100% ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1,000/1,000  [ 0:00:00 < 0:00:00 , 7,292 it/s ][eval val w03] 0/1,830 (  0.0%) | elapsed=0s | eta=0s | starting evaluation

[eval val w00] 0/1,830 (  0.0%) | elapsed=0s | eta=0s | starting evaluation
[eval val w06] 0/1,830 (  0.0%) | elapsed=0s | eta=0s | starting evaluation
[eval val w07] 0/1,830 (  0.0%) | elapsed=0s | eta=0s | starting evaluation
[eval val w09] 0/1,830 (  0.0%) | elapsed=0s | eta=0s | starting evaluation
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 20.2     |
|    ep_rew_mean     | -1.82    |
| time/              |          |
|    episodes        | 48       |
|    fps             | 6994     |
|    time_elapsed    | 0        |
|    total_timesteps | 969      |
---------------------------------
[eval val w10] 0/1,830 (  0.0%) | elapsed=0s | eta=0s | starting evaluation
 100% ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1,000/1,000  [ 0:00:00 < 0:00:00 , 6,759 it/s ]
[eval val w08] 0/1,830 (  0.0%) | elapsed=0s | eta=0s | starting evaluation
[eval val w11] 0/1,830 (  0.0%) | elapsed=0s | eta=0s | starting evaluation
[eval val w01] 183/1,830 ( 10.0%) | elapsed=1s | eta=5s | latest_scenario=56d845022153dc4f19fd
[eval val w05] 183/1,830 ( 10.0%) | elapsed=1s | eta=5s | latest_scenario=4d8d0b5ac310d011fdcd
[eval val w00] 183/1,830 ( 10.0%) | elapsed=1s | eta=5s | latest_scenario=72fac5860574b6fcc6e8
[eval val w06] 183/1,830 ( 10.0%) | elapsed=1s | eta=5s | latest_scenario=8882d5baba997c904970
[eval val w04] 183/1,830 ( 10.0%) | elapsed=1s | eta=6s | latest_scenario=f0044b2271ccd21defe0
[eval val w02] 183/1,830 ( 10.0%) | elapsed=1s | eta=6s | latest_scenario=d4cb54496c9839152ccd
[eval val w03] 183/1,830 ( 10.0%) | elapsed=1s | eta=6s | latest_scenario=21d38e03d010a53b7648
[eval val w07] 183/1,830 ( 10.0%) | elapsed=1s | eta=6s | latest_scenario=d0079a4b4aaea22f0c26
[eval val w10] 183/1,830 ( 10.0%) | elapsed=1s | eta=6s | latest_scenario=04f5e03b5993bdd6c48e
[eval val w08] 183/1,830 ( 10.0%) | elapsed=1s | eta=6s | latest_scenario=c0d1aa0aa58129cc6b98
[eval val w09] 183/1,830 ( 10.0%) | elapsed=1s | eta=6s | latest_scenario=4c5e54e1b94ddf644df2
[eval val w11] 183/1,830 ( 10.0%) | elapsed=1s | eta=6s | latest_scenario=389ea2fcccd49452562f
[eval val w04] 366/1,830 ( 20.0%) | elapsed=1s | eta=5s | latest_scenario=6689bd0003560c60ecef
[eval val w03] 366/1,830 ( 20.0%) | elapsed=1s | eta=5s | latest_scenario=4f87ff71a51e5c4c3ef6
[eval val w01] 366/1,830 ( 20.0%) | elapsed=1s | eta=5s | latest_scenario=dbbfcc10a3283a0fcfb1
[eval val w06] 366/1,830 ( 20.0%) | elapsed=1s | eta=5s | latest_scenario=cecf41c2bbff05293cc2
[eval val w10] 366/1,830 ( 20.0%) | elapsed=1s | eta=5s | latest_scenario=0b7f4ed292a01ecf865e
[eval val w05] 366/1,830 ( 20.0%) | elapsed=1s | eta=5s | latest_scenario=df4b926423103c209077
[eval val w00] 366/1,830 ( 20.0%) | elapsed=1s | eta=5s | latest_scenario=738f73d6c101a5f7cae5
[eval val w07] 366/1,830 ( 20.0%) | elapsed=1s | eta=5s | latest_scenario=9751a201800cabcc5c32
[eval val w08] 366/1,830 ( 20.0%) | elapsed=1s | eta=5s | latest_scenario=6fe409104d2a26f04047
[eval val w02] 366/1,830 ( 20.0%) | elapsed=1s | eta=5s | latest_scenario=ae58ae4439f10189b327
[eval val w11] 366/1,830 ( 20.0%) | elapsed=1s | eta=5s | latest_scenario=b2b0f26f0ef949f7b0cd
[eval val w09] 366/1,830 ( 20.0%) | elapsed=1s | eta=5s | latest_scenario=37b0b527a43b47cd152a
[eval val w06] 549/1,830 ( 30.0%) | elapsed=2s | eta=4s | latest_scenario=e6fe5bfcd2e0958a90a8
[eval val w04] 549/1,830 ( 30.0%) | elapsed=2s | eta=4s | latest_scenario=5a5fe3a32c18b0593ea5
[eval val w03] 549/1,830 ( 30.0%) | elapsed=2s | eta=4s | latest_scenario=ef4ba224068a12ab69a3
[eval val w01] 549/1,830 ( 30.0%) | elapsed=2s | eta=4s | latest_scenario=aa69b241e9d670888c9d
[eval val w07] 549/1,830 ( 30.0%) | elapsed=2s | eta=4s | latest_scenario=08d836da632a69c46ae4
[eval val w00] 549/1,830 ( 30.0%) | elapsed=2s | eta=4s | latest_scenario=cbf88f625024fc9f40d2
[eval val w10] 549/1,830 ( 30.0%) | elapsed=2s | eta=4s | latest_scenario=d876df438ece3b38253d
[eval val w05] 549/1,830 ( 30.0%) | elapsed=2s | eta=5s | latest_scenario=8458aa0e8e5c580e1b7f
[eval val w02] 549/1,830 ( 30.0%) | elapsed=2s | eta=5s | latest_scenario=77952ff55f010c262e95
[eval val w08] 549/1,830 ( 30.0%) | elapsed=2s | eta=5s | latest_scenario=b84bd14bbf4f8f2515a5
[eval val w09] 549/1,830 ( 30.0%) | elapsed=2s | eta=5s | latest_scenario=d81039d9fb1c23e04fe7
[eval val w11] 549/1,830 ( 30.0%) | elapsed=2s | eta=4s | latest_scenario=d4922da6dee8f2a6ccf6
[eval val w06] 732/1,830 ( 40.0%) | elapsed=2s | eta=4s | latest_scenario=58a2e106afc7afaf7f06
[eval val w03] 732/1,830 ( 40.0%) | elapsed=2s | eta=4s | latest_scenario=03fabc7e8b019cc6003a
[eval val w04] 732/1,830 ( 40.0%) | elapsed=3s | eta=4s | latest_scenario=7e6c39f1b4cbb20776ab
[eval val w01] 732/1,830 ( 40.0%) | elapsed=3s | eta=4s | latest_scenario=7d5cb949652ef5d9adce
[eval val w00] 732/1,830 ( 40.0%) | elapsed=3s | eta=4s | latest_scenario=26488bfc8865e016fcdd
[eval val w07] 732/1,830 ( 40.0%) | elapsed=3s | eta=4s | latest_scenario=50a34fbe181d23ad4458
[eval val w05] 732/1,830 ( 40.0%) | elapsed=3s | eta=4s | latest_scenario=f027aa842724e7123189
[eval val w11] 732/1,830 ( 40.0%) | elapsed=2s | eta=4s | latest_scenario=1ae858c4412e212d8007
[eval val w10] 732/1,830 ( 40.0%) | elapsed=3s | eta=4s | latest_scenario=79ffcdf3a8a6607bbdfc
[eval val w08] 732/1,830 ( 40.0%) | elapsed=3s | eta=4s | latest_scenario=f64e41ca6dbbb9f4cf44
[eval val w02] 732/1,830 ( 40.0%) | elapsed=3s | eta=4s | latest_scenario=d8ed582813f24a476ec0
[eval val w09] 732/1,830 ( 40.0%) | elapsed=3s | eta=4s | latest_scenario=25e3a643bc34bad9eb8b
[eval val w06] 915/1,830 ( 50.0%) | elapsed=3s | eta=3s | latest_scenario=a53152f43a586fee5351
[eval val w03] 915/1,830 ( 50.0%) | elapsed=3s | eta=3s | latest_scenario=f68368e02a289972232f
[eval val w00] 915/1,830 ( 50.0%) | elapsed=3s | eta=3s | latest_scenario=9adec467650a2cb09831
[eval val w01] 915/1,830 ( 50.0%) | elapsed=3s | eta=3s | latest_scenario=1a4308e8c0ee54aa6f1d
[eval val w04] 915/1,830 ( 50.0%) | elapsed=3s | eta=3s | latest_scenario=187c72e8777abbc2586a
[eval val w11] 915/1,830 ( 50.0%) | elapsed=3s | eta=3s | latest_scenario=91fd9e265e19cc1a9f61
[eval val w08] 915/1,830 ( 50.0%) | elapsed=3s | eta=3s | latest_scenario=20b67b31660cfc07e2ca
[eval val w05] 915/1,830 ( 50.0%) | elapsed=3s | eta=3s | latest_scenario=0877bbc78629db705420
[eval val w07] 915/1,830 ( 50.0%) | elapsed=3s | eta=3s | latest_scenario=d6748b308d289409a19b
[eval val w02] 915/1,830 ( 50.0%) | elapsed=3s | eta=3s | latest_scenario=5686eb8c03ca05fb57cf
[eval val w10] 915/1,830 ( 50.0%) | elapsed=3s | eta=3s | latest_scenario=5fa8e38795771f2e4f52
[eval val w09] 915/1,830 ( 50.0%) | elapsed=3s | eta=3s | latest_scenario=254ee4b452c140518eaf
[eval val w06] 1,098/1,830 ( 60.0%) | elapsed=4s | eta=2s | latest_scenario=85c2ddc2d68807a33940
[eval val w01] 1,098/1,830 ( 60.0%) | elapsed=4s | eta=2s | latest_scenario=7a34362512d291032f28
[eval val w03] 1,098/1,830 ( 60.0%) | elapsed=4s | eta=2s | latest_scenario=561a3d8c86a4be7dc032
[eval val w00] 1,098/1,830 ( 60.0%) | elapsed=4s | eta=3s | latest_scenario=f52ec98ad5a84e2d0721
[eval val w04] 1,098/1,830 ( 60.0%) | elapsed=4s | eta=3s | latest_scenario=57c5ea4b6c5ad5eda34d
[eval val w08] 1,098/1,830 ( 60.0%) | elapsed=4s | eta=3s | latest_scenario=6a0794b984af2f71f64c
[eval val w11] 1,098/1,830 ( 60.0%) | elapsed=4s | eta=3s | latest_scenario=b5ee6547a76ad8044450
[eval val w05] 1,098/1,830 ( 60.0%) | elapsed=4s | eta=3s | latest_scenario=daa6de6747ed9a4cd8cc
[eval val w07] 1,098/1,830 ( 60.0%) | elapsed=4s | eta=3s | latest_scenario=a37cfd3f9d07622fa77b
[eval val w10] 1,098/1,830 ( 60.0%) | elapsed=4s | eta=3s | latest_scenario=6c006b29ea0851fd6723
[eval val w02] 1,098/1,830 ( 60.0%) | elapsed=4s | eta=3s | latest_scenario=ce880518d7396feb20e7
[eval val w09] 1,098/1,830 ( 60.0%) | elapsed=4s | eta=3s | latest_scenario=b9df5281c6de5c503528
[eval val w06] 1,281/1,830 ( 70.0%) | elapsed=4s | eta=2s | latest_scenario=97437b0036bc9d1cbec6
[eval val w01] 1,281/1,830 ( 70.0%) | elapsed=4s | eta=2s | latest_scenario=1cc448b9031bdbef8ad1
[eval val w03] 1,281/1,830 ( 70.0%) | elapsed=4s | eta=2s | latest_scenario=4b0e7f1e63e49a5d156c
[eval val w00] 1,281/1,830 ( 70.0%) | elapsed=4s | eta=2s | latest_scenario=869ce5fbb1bebaa3b6f5
[eval val w04] 1,281/1,830 ( 70.0%) | elapsed=4s | eta=2s | latest_scenario=1e9631d0fa9652504e44
[eval val w08] 1,281/1,830 ( 70.0%) | elapsed=4s | eta=2s | latest_scenario=98521e6ec4513ec07fb6
[eval val w05] 1,281/1,830 ( 70.0%) | elapsed=4s | eta=2s | latest_scenario=a4d493a56ecd8f8c039e
[eval val w11] 1,281/1,830 ( 70.0%) | elapsed=4s | eta=2s | latest_scenario=36d38449f8e0218280a8
[eval val w02] 1,281/1,830 ( 70.0%) | elapsed=5s | eta=2s | latest_scenario=6b602f244e719061d817
[eval val w07] 1,281/1,830 ( 70.0%) | elapsed=5s | eta=2s | latest_scenario=102953c58830d97852fe
[eval val w09] 1,281/1,830 ( 70.0%) | elapsed=5s | eta=2s | latest_scenario=e24bb7d1c08699d541bd
[eval val w10] 1,281/1,830 ( 70.0%) | elapsed=5s | eta=2s | latest_scenario=f24749eb095fa591463e
[eval val w06] 1,464/1,830 ( 80.0%) | elapsed=5s | eta=1s | latest_scenario=dcf1f57e6a68a0d1c69d
[eval val w01] 1,464/1,830 ( 80.0%) | elapsed=5s | eta=1s | latest_scenario=8eb549b0dd502c8ea0eb
[eval val w03] 1,464/1,830 ( 80.0%) | elapsed=5s | eta=1s | latest_scenario=afc3018124bb6a522e2d
[eval val w00] 1,464/1,830 ( 80.0%) | elapsed=5s | eta=1s | latest_scenario=12f25459f7c6cfa4beda
[eval val w04] 1,464/1,830 ( 80.0%) | elapsed=5s | eta=1s | latest_scenario=f46fa9ce8b8780b80d91
[eval val w05] 1,464/1,830 ( 80.0%) | elapsed=5s | eta=1s | latest_scenario=d017c886a1aef93f7758
[eval val w08] 1,464/1,830 ( 80.0%) | elapsed=5s | eta=1s | latest_scenario=cff2358065effece5e40
[eval val w11] 1,464/1,830 ( 80.0%) | elapsed=5s | eta=1s | latest_scenario=5ce13bd82a25ef4d2b4a
[eval val w09] 1,464/1,830 ( 80.0%) | elapsed=5s | eta=1s | latest_scenario=da3aea72b966071e4ffd
[eval val w02] 1,464/1,830 ( 80.0%) | elapsed=5s | eta=1s | latest_scenario=36210cf9256a7ae9f5a6
[eval val w07] 1,464/1,830 ( 80.0%) | elapsed=5s | eta=1s | latest_scenario=ea38479b1e366ec5ae71
[eval val w10] 1,464/1,830 ( 80.0%) | elapsed=5s | eta=1s | latest_scenario=aa6c72fe46d2b6aa8c10
[eval val w01] 1,647/1,830 ( 90.0%) | elapsed=6s | eta=1s | latest_scenario=909102e25a591fb3a1d6
[eval val w06] 1,647/1,830 ( 90.0%) | elapsed=6s | eta=1s | latest_scenario=27d1ddc3126d3b3625de
[eval val w03] 1,647/1,830 ( 90.0%) | elapsed=6s | eta=1s | latest_scenario=c9f476275254ac4238c7
[eval val w00] 1,647/1,830 ( 90.0%) | elapsed=6s | eta=1s | latest_scenario=27360fd30ace579967d2
[eval val w04] 1,647/1,830 ( 90.0%) | elapsed=6s | eta=1s | latest_scenario=9be8cab09f205d373a03
[eval val w05] 1,647/1,830 ( 90.0%) | elapsed=6s | eta=1s | latest_scenario=8c99b76d413b345861a4
[eval val w08] 1,647/1,830 ( 90.0%) | elapsed=6s | eta=1s | latest_scenario=aced447bbcbdf9d57e68
[eval val w11] 1,647/1,830 ( 90.0%) | elapsed=6s | eta=1s | latest_scenario=961b6e0ed630d19e24d2
[eval val w09] 1,647/1,830 ( 90.0%) | elapsed=6s | eta=1s | latest_scenario=5f16090110dde1dbdfd2
[eval val w07] 1,647/1,830 ( 90.0%) | elapsed=6s | eta=1s | latest_scenario=a9d2ec6e6b2029798223
[eval val w10] 1,647/1,830 ( 90.0%) | elapsed=6s | eta=1s | latest_scenario=65651438137e8ca5a355
[eval val w02] 1,647/1,830 ( 90.0%) | elapsed=6s | eta=1s | latest_scenario=8e357ad6191c75f27abd
[eval val w01] 1,830/1,830 (100.0%) | elapsed=6s | eta=0s | latest_scenario=ea41e96aaaaf6ddec02a
[eval val w06] 1,830/1,830 (100.0%) | elapsed=6s | eta=0s | latest_scenario=c22c60c26ffdead87978
[eval val w03] 1,830/1,830 (100.0%) | elapsed=6s | eta=0s | latest_scenario=8813d852163eef0ce6ae
[eval val w04] 1,830/1,830 (100.0%) | elapsed=6s | eta=0s | latest_scenario=6f027998fd64febcbf99
[eval val w00] 1,830/1,830 (100.0%) | elapsed=6s | eta=0s | latest_scenario=09b10d83c5199edec0c0
[eval val w05] 1,830/1,830 (100.0%) | elapsed=6s | eta=0s | latest_scenario=c2a54da4227babb0d9bd
[eval val w08] 1,830/1,830 (100.0%) | elapsed=6s | eta=0s | latest_scenario=635ffcc7b3f74b69b8b4
[eval val w11] 1,830/1,830 (100.0%) | elapsed=6s | eta=0s | latest_scenario=5abf85b4c60bb64f9522
[eval val w10] 1,830/1,830 (100.0%) | elapsed=6s | eta=0s | latest_scenario=eb3de9c9d9ea0cb20bcf
[eval val w09] 1,830/1,830 (100.0%) | elapsed=6s | eta=0s | latest_scenario=33fe648425c13c6dc300
[eval val w07] 1,830/1,830 (100.0%) | elapsed=6s | eta=0s | latest_scenario=ceb6487f0e21a27d759d
[eval val w02] 1,830/1,830 (100.0%) | elapsed=6s | eta=0s | latest_scenario=450d8a5394bf9e207c64
[eval test w06] 0/2,511 (  0.0%) | elapsed=0s | eta=0s | starting evaluation
[eval test w01] 0/2,509 (  0.0%) | elapsed=0s | eta=0s | starting evaluation
[eval test w03] 0/2,509 (  0.0%) | elapsed=0s | eta=0s | starting evaluation
[eval test w00] 0/2,509 (  0.0%) | elapsed=0s | eta=0s | starting evaluation
[eval test w04] 0/2,530 (  0.0%) | elapsed=0s | eta=0s | starting evaluation
[eval test w05] 0/2,529 (  0.0%) | elapsed=0s | eta=0s | starting evaluation
[eval test w08] 0/2,508 (  0.0%) | elapsed=0s | eta=0s | starting evaluation
[eval test w09] 0/2,527 (  0.0%) | elapsed=0s | eta=0s | starting evaluation
[eval test w07] 0/2,538 (  0.0%) | elapsed=0s | eta=0s | starting evaluation
[eval test w11] 0/2,503 (  0.0%) | elapsed=0s | eta=0s | starting evaluation
[eval test w10] 0/2,538 (  0.0%) | elapsed=0s | eta=0s | starting evaluation
[eval test w02] 0/2,520 (  0.0%) | elapsed=0s | eta=0s | starting evaluation
[eval test w06] 125/2,511 (  5.0%) | elapsed=1s | eta=10s | latest_scenario=5f96e7f00793fd202bd4
[eval test w01] 125/2,509 (  5.0%) | elapsed=1s | eta=11s | latest_scenario=d0e95451fb56c4ab6549
[eval test w05] 126/2,529 (  5.0%) | elapsed=0s | eta=7s | latest_scenario=72e592ccc873aa5fabad
[eval test w04] 126/2,530 (  5.0%) | elapsed=0s | eta=8s | latest_scenario=5b8a672064d85514ce90
[eval test w03] 125/2,509 (  5.0%) | elapsed=1s | eta=11s | latest_scenario=b350d49f08d432bc4b9e
[eval test w00] 125/2,509 (  5.0%) | elapsed=0s | eta=9s | latest_scenario=54e0b8e372e85a9ed2a3
[eval test w09] 126/2,527 (  5.0%) | elapsed=0s | eta=8s | latest_scenario=d958bb6f06c6981097f9
[eval test w08] 125/2,508 (  5.0%) | elapsed=0s | eta=9s | latest_scenario=85bc203bf8ea59593e99
[eval test w07] 126/2,538 (  5.0%) | elapsed=0s | eta=8s | latest_scenario=7b329362cdaf1f9ef17f
[eval test w10] 126/2,538 (  5.0%) | elapsed=0s | eta=7s | latest_scenario=42989b801923c9f0fdea
[eval test w02] 126/2,520 (  5.0%) | elapsed=0s | eta=8s | latest_scenario=e11b95f4cf13f5e09f04
[eval test w11] 125/2,503 (  5.0%) | elapsed=0s | eta=9s | latest_scenario=5a3770ab0687cba33e80
[eval test w06] 250/2,511 ( 10.0%) | elapsed=1s | eta=9s | latest_scenario=54319f2adbe7ad80376f
[eval test w01] 250/2,509 ( 10.0%) | elapsed=1s | eta=9s | latest_scenario=d4cb54496c9839152ccd
[eval test w05] 252/2,529 ( 10.0%) | elapsed=1s | eta=7s | latest_scenario=40bd8b0114d39b1500de
[eval test w03] 250/2,509 ( 10.0%) | elapsed=1s | eta=9s | latest_scenario=f0044b2271ccd21defe0
[eval test w04] 252/2,530 ( 10.0%) | elapsed=1s | eta=8s | latest_scenario=13298b58a3c8e54c6c20
[eval test w00] 250/2,509 ( 10.0%) | elapsed=1s | eta=8s | latest_scenario=87dafbf4520a750ee097
[eval test w09] 252/2,527 ( 10.0%) | elapsed=1s | eta=8s | latest_scenario=64bb993da6ebb07dae48
[eval test w08] 250/2,508 ( 10.0%) | elapsed=1s | eta=8s | latest_scenario=4c5e54e1b94ddf644df2
[eval test w07] 252/2,538 (  9.9%) | elapsed=1s | eta=8s | latest_scenario=11e8c2d6343499be4277
[eval test w10] 252/2,538 (  9.9%) | elapsed=1s | eta=7s | latest_scenario=7061a65c3bd41b88a30f
[eval test w02] 252/2,520 ( 10.0%) | elapsed=1s | eta=7s | latest_scenario=2f41150ec545febb2d03
[eval test w11] 250/2,503 ( 10.0%) | elapsed=1s | eta=8s | latest_scenario=6cfd30d14e127398386e
[eval test w01] 375/2,509 ( 14.9%) | elapsed=1s | eta=8s | latest_scenario=2de98e0f68e11cd2707e
[eval test w06] 375/2,511 ( 14.9%) | elapsed=1s | eta=8s | latest_scenario=8950c7a82be0144aa2fb
[eval test w05] 378/2,529 ( 14.9%) | elapsed=1s | eta=7s | latest_scenario=3e83622c6c8eca2955b2
[eval test w03] 375/2,509 ( 14.9%) | elapsed=1s | eta=8s | latest_scenario=482cc5a95987faccbbd6
[eval test w04] 378/2,530 ( 14.9%) | elapsed=1s | eta=7s | latest_scenario=3c5beeeec32c3e97a392
[eval test w09] 378/2,527 ( 15.0%) | elapsed=1s | eta=7s | latest_scenario=05dd74b3812a6774c84b
[eval test w00] 375/2,509 ( 14.9%) | elapsed=1s | eta=8s | latest_scenario=4eb429166c83f63e9f33
[eval test w08] 375/2,508 ( 15.0%) | elapsed=1s | eta=7s | latest_scenario=7e3edbeb06c42462de0d
[eval test w10] 378/2,538 ( 14.9%) | elapsed=1s | eta=7s | latest_scenario=87f40896c8aa0d1ceb08
[eval test w07] 378/2,538 ( 14.9%) | elapsed=1s | eta=7s | latest_scenario=c64e40b63cdac27e0200
[eval test w02] 378/2,520 ( 15.0%) | elapsed=1s | eta=7s | latest_scenario=7079f5ed9a5e9a1f0dbf
[eval test w11] 375/2,503 ( 15.0%) | elapsed=1s | eta=7s | latest_scenario=1de1a1c8f02d03663065
[eval test w01] 500/2,509 ( 19.9%) | elapsed=2s | eta=7s | latest_scenario=be1266b32e0e561e169a
[eval test w06] 500/2,511 ( 19.9%) | elapsed=2s | eta=7s | latest_scenario=dcf76b25f19359415f27
[eval test w05] 504/2,529 ( 19.9%) | elapsed=2s | eta=6s | latest_scenario=fee660c4bc43933c9792
[eval test w04] 504/2,530 ( 19.9%) | elapsed=2s | eta=7s | latest_scenario=d552aba12aa9a82b9fce
[eval test w03] 500/2,509 ( 19.9%) | elapsed=2s | eta=7s | latest_scenario=846d818c18c2c8d6cad0
[eval test w00] 500/2,509 ( 19.9%) | elapsed=2s | eta=7s | latest_scenario=04b7c891d62132b355cf
[eval test w09] 504/2,527 ( 19.9%) | elapsed=2s | eta=7s | latest_scenario=e907f03c46fd4cb8ef2a
[eval test w10] 504/2,538 ( 19.9%) | elapsed=2s | eta=6s | latest_scenario=1b77ce94945ed8e8305d
[eval test w08] 500/2,508 ( 19.9%) | elapsed=2s | eta=7s | latest_scenario=4762b00d45ecac1414cf
[eval test w07] 504/2,538 ( 19.9%) | elapsed=2s | eta=7s | latest_scenario=23e4a21cbc2ab31a79ee
[eval test w11] 500/2,503 ( 20.0%) | elapsed=2s | eta=7s | latest_scenario=5c5121ce38370e8927c2
[eval test w02] 504/2,520 ( 20.0%) | elapsed=2s | eta=7s | latest_scenario=a2f0648f5afbd37cddb3
[eval test w06] 625/2,511 ( 24.9%) | elapsed=2s | eta=7s | latest_scenario=b6adc2a02901fead8c46
[eval test w05] 630/2,529 ( 24.9%) | elapsed=2s | eta=6s | latest_scenario=42acf7606db344a87060
[eval test w01] 625/2,509 ( 24.9%) | elapsed=2s | eta=7s | latest_scenario=d98842fcd2b16e01cec6
[eval test w04] 630/2,530 ( 24.9%) | elapsed=2s | eta=6s | latest_scenario=23ad48cd752ae3276d6f
[eval test w03] 625/2,509 ( 24.9%) | elapsed=2s | eta=7s | latest_scenario=e5d313eae6ea1ca09e6f
[eval test w00] 625/2,509 ( 24.9%) | elapsed=2s | eta=6s | latest_scenario=da627c1b44f103d4f276
[eval test w08] 625/2,508 ( 24.9%) | elapsed=2s | eta=6s | latest_scenario=9e72d6227a70584aec4b
[eval test w09] 630/2,527 ( 24.9%) | elapsed=2s | eta=6s | latest_scenario=e245327f721a5da71f90
[eval test w10] 630/2,538 ( 24.8%) | elapsed=2s | eta=6s | latest_scenario=0c0241b02f9c5cb508df
[eval test w07] 630/2,538 ( 24.8%) | elapsed=2s | eta=6s | latest_scenario=a50d89751e0cdfe60cdf
[eval test w11] 625/2,503 ( 25.0%) | elapsed=2s | eta=6s | latest_scenario=8f9521e3755712c028d6
[eval test w02] 630/2,520 ( 25.0%) | elapsed=2s | eta=6s | latest_scenario=f70f33cc0948a9909603
[eval test w06] 750/2,511 ( 29.9%) | elapsed=3s | eta=6s | latest_scenario=b30f8c6d51a3f0fbfdf6
[eval test w05] 756/2,529 ( 29.9%) | elapsed=2s | eta=6s | latest_scenario=d360d0aa85dce6591f2a
[eval test w01] 750/2,509 ( 29.9%) | elapsed=3s | eta=6s | latest_scenario=a90a7f21b9e06908de82
[eval test w00] 750/2,509 ( 29.9%) | elapsed=3s | eta=6s | latest_scenario=9187d99ee1abe71796d3
[eval test w03] 750/2,509 ( 29.9%) | elapsed=3s | eta=6s | latest_scenario=2f6d7dcc79e26543a277
[eval test w04] 756/2,530 ( 29.9%) | elapsed=3s | eta=6s | latest_scenario=8458aa0e8e5c580e1b7f
[eval test w08] 750/2,508 ( 29.9%) | elapsed=2s | eta=6s | latest_scenario=47c76862040c436dd362
[eval test w09] 756/2,527 ( 29.9%) | elapsed=2s | eta=6s | latest_scenario=d876df438ece3b38253d
[eval test w10] 756/2,538 ( 29.8%) | elapsed=2s | eta=6s | latest_scenario=d4d99e70fc4cdb0d4d50
[eval test w07] 756/2,538 ( 29.8%) | elapsed=3s | eta=6s | latest_scenario=1bf982ee346a36e72eea
[eval test w11] 750/2,503 ( 30.0%) | elapsed=2s | eta=6s | latest_scenario=fe80f05c8e0f6e7cad04
[eval test w02] 756/2,520 ( 30.0%) | elapsed=3s | eta=6s | latest_scenario=65c9aefd87c0397741ad
[eval test w06] 875/2,511 ( 34.8%) | elapsed=3s | eta=6s | latest_scenario=dfa40fa9f91c14c90e3e
[eval test w05] 882/2,529 ( 34.9%) | elapsed=3s | eta=5s | latest_scenario=19f92cd8144d46af04d9
[eval test w01] 875/2,509 ( 34.9%) | elapsed=3s | eta=6s | latest_scenario=c35320a508b2d45cec5e
[eval test w00] 875/2,509 ( 34.9%) | elapsed=3s | eta=5s | latest_scenario=db175344c32a5f94f7eb
[eval test w03] 875/2,509 ( 34.9%) | elapsed=3s | eta=6s | latest_scenario=6ea4353f1aacd11d9125
[eval test w10] 882/2,538 ( 34.8%) | elapsed=3s | eta=5s | latest_scenario=263145dce0c1f4bda0eb
[eval test w08] 875/2,508 ( 34.9%) | elapsed=3s | eta=5s | latest_scenario=b37ba39893531b2a7e34
[eval test w04] 882/2,530 ( 34.9%) | elapsed=3s | eta=6s | latest_scenario=444603ff7c801d50a360
[eval test w09] 882/2,527 ( 34.9%) | elapsed=3s | eta=5s | latest_scenario=22db4ba41a6ca41e4ffe
[eval test w11] 875/2,503 ( 35.0%) | elapsed=3s | eta=5s | latest_scenario=75138d4001aff2d6ecd1
[eval test w07] 882/2,538 ( 34.8%) | elapsed=3s | eta=5s | latest_scenario=50da09dead997de95a48
[eval test w02] 882/2,520 ( 35.0%) | elapsed=3s | eta=5s | latest_scenario=a51ed36262e461116e35
[eval test w05] 1,008/2,529 ( 39.9%) | elapsed=3s | eta=5s | latest_scenario=907d411ffa27ad156f1a
[eval test w06] 1,000/2,511 ( 39.8%) | elapsed=3s | eta=5s | latest_scenario=201031452886d4d4d0c9
[eval test w01] 1,000/2,509 ( 39.9%) | elapsed=4s | eta=5s | latest_scenario=24c80ed374bc74f91d3c
[eval test w00] 1,000/2,509 ( 39.9%) | elapsed=3s | eta=5s | latest_scenario=ff05a68a336eda5ac034
[eval test w10] 1,008/2,538 ( 39.7%) | elapsed=3s | eta=5s | latest_scenario=6ff01a6c01dd78d6d516
[eval test w03] 1,000/2,509 ( 39.9%) | elapsed=3s | eta=5s | latest_scenario=759f2af0af12f9868c79
[eval test w08] 1,000/2,508 ( 39.9%) | elapsed=3s | eta=5s | latest_scenario=092164c652894c90c896
[eval test w04] 1,008/2,530 ( 39.8%) | elapsed=3s | eta=5s | latest_scenario=ebd7cb58af3697224591
[eval test w09] 1,008/2,527 ( 39.9%) | elapsed=3s | eta=5s | latest_scenario=23c2ee36aa21ebdf2989
[eval test w11] 1,000/2,503 ( 40.0%) | elapsed=3s | eta=5s | latest_scenario=8f8ec2ed4e61f14520d8
[eval test w07] 1,008/2,538 ( 39.7%) | elapsed=3s | eta=5s | latest_scenario=50859e965bf707057b17
[eval test w02] 1,008/2,520 ( 40.0%) | elapsed=3s | eta=5s | latest_scenario=5e056ede1a47802ca9b3
[eval test w05] 1,134/2,529 ( 44.8%) | elapsed=4s | eta=4s | latest_scenario=92e2182973bae9f37cb2
[eval test w06] 1,125/2,511 ( 44.8%) | elapsed=4s | eta=5s | latest_scenario=bf5aac05553b3dcf1832
[eval test w01] 1,125/2,509 ( 44.8%) | elapsed=4s | eta=5s | latest_scenario=053b77572acf1f3a999f
[eval test w00] 1,125/2,509 ( 44.8%) | elapsed=4s | eta=5s | latest_scenario=fef189d52a8f616e641e
[eval test w08] 1,125/2,508 ( 44.9%) | elapsed=4s | eta=5s | latest_scenario=268c612c62e6621a9550
[eval test w03] 1,125/2,509 ( 44.8%) | elapsed=4s | eta=5s | latest_scenario=702fb4d3cd3f39555708
[eval test w10] 1,134/2,538 ( 44.7%) | elapsed=4s | eta=5s | latest_scenario=627145183249b37aef6f
[eval test w04] 1,134/2,530 ( 44.8%) | elapsed=4s | eta=5s | latest_scenario=98acdfe104f4e4f465da
[eval test w09] 1,134/2,527 ( 44.9%) | elapsed=4s | eta=5s | latest_scenario=13c432fe8b22ec6a65aa
[eval test w11] 1,125/2,503 ( 44.9%) | elapsed=4s | eta=5s | latest_scenario=0de756144523b0a32706
[eval test w07] 1,134/2,538 ( 44.7%) | elapsed=4s | eta=5s | latest_scenario=598070d228a53f12f1ed
[eval test w02] 1,134/2,520 ( 45.0%) | elapsed=4s | eta=5s | latest_scenario=e3ae77e68653b3baa1d7
[eval test w05] 1,260/2,529 ( 49.8%) | elapsed=4s | eta=4s | latest_scenario=434450c5d873acc18c5c
[eval test w06] 1,250/2,511 ( 49.8%) | elapsed=4s | eta=4s | latest_scenario=2fb14203a5df2b93fd6b
[eval test w03] 1,250/2,509 ( 49.8%) | elapsed=4s | eta=4s | latest_scenario=dedea7106d887dbba43b
[eval test w00] 1,250/2,509 ( 49.8%) | elapsed=4s | eta=4s | latest_scenario=7d19e8243362f03db93f
[eval test w01] 1,250/2,509 ( 49.8%) | elapsed=4s | eta=4s | latest_scenario=fdfbf9a9dda54e9f08d0
[eval test w10] 1,260/2,538 ( 49.6%) | elapsed=4s | eta=4s | latest_scenario=0b01d21bd4f2f9d79a82
[eval test w08] 1,250/2,508 ( 49.8%) | elapsed=4s | eta=4s | latest_scenario=bd9693433dfce305f1a8
[eval test w04] 1,260/2,530 ( 49.8%) | elapsed=4s | eta=4s | latest_scenario=203256e120f192e0f1de
[eval test w09] 1,260/2,527 ( 49.9%) | elapsed=4s | eta=4s | latest_scenario=d23481e7bb3855ba0af3
[eval test w11] 1,250/2,503 ( 49.9%) | elapsed=4s | eta=4s | latest_scenario=97117485e0574f2fc2d1
[eval test w07] 1,260/2,538 ( 49.6%) | elapsed=4s | eta=4s | latest_scenario=f58b353738642509fac9
[eval test w02] 1,260/2,520 ( 50.0%) | elapsed=4s | eta=4s | latest_scenario=d6216bffa503306282fd
[eval test w05] 1,386/2,529 ( 54.8%) | elapsed=4s | eta=4s | latest_scenario=f9292af405cffc5c49f2
[eval test w06] 1,375/2,511 ( 54.8%) | elapsed=5s | eta=4s | latest_scenario=011eed960edabfe6b4ed
[eval test w03] 1,375/2,509 ( 54.8%) | elapsed=5s | eta=4s | latest_scenario=31337f78e81a17dc0f24
[eval test w00] 1,375/2,509 ( 54.8%) | elapsed=5s | eta=4s | latest_scenario=1db60820de9e192debd5
[eval test w01] 1,375/2,509 ( 54.8%) | elapsed=5s | eta=4s | latest_scenario=254abe7a9377bb8c852b
[eval test w10] 1,386/2,538 ( 54.6%) | elapsed=4s | eta=4s | latest_scenario=5f27eb213718784463cb
[eval test w08] 1,375/2,508 ( 54.8%) | elapsed=5s | eta=4s | latest_scenario=5f14e4e11d721cc43f2b
[eval test w09] 1,386/2,527 ( 54.8%) | elapsed=5s | eta=4s | latest_scenario=5eb663c8701ad3cc618a
[eval test w04] 1,386/2,530 ( 54.8%) | elapsed=5s | eta=4s | latest_scenario=87b9e9de98be2e0dd629
[eval test w11] 1,375/2,503 ( 54.9%) | elapsed=5s | eta=4s | latest_scenario=d05775af8ce3aa8901a5
[eval test w07] 1,386/2,538 ( 54.6%) | elapsed=5s | eta=4s | latest_scenario=30fabab13ffe6e81e84b
[eval test w02] 1,386/2,520 ( 55.0%) | elapsed=5s | eta=4s | latest_scenario=8759d7f5f43160f05df9
[eval test w05] 1,512/2,529 ( 59.8%) | elapsed=5s | eta=3s | latest_scenario=9e647ea2cd0a7ef65f5b
[eval test w06] 1,500/2,511 ( 59.7%) | elapsed=5s | eta=3s | latest_scenario=2df1be64304723c284e3
[eval test w03] 1,500/2,509 ( 59.8%) | elapsed=5s | eta=3s | latest_scenario=81a163e166578ff568d0
[eval test w00] 1,500/2,509 ( 59.8%) | elapsed=5s | eta=3s | latest_scenario=f967f81523f9f8770310
[eval test w01] 1,500/2,509 ( 59.8%) | elapsed=5s | eta=3s | latest_scenario=82d399efd940840ab26c
[eval test w08] 1,500/2,508 ( 59.8%) | elapsed=5s | eta=3s | latest_scenario=a77b3a58680e5cdf4201
[eval test w10] 1,512/2,538 ( 59.6%) | elapsed=5s | eta=3s | latest_scenario=94ab01e7ef73741bdb4b
[eval test w09] 1,512/2,527 ( 59.8%) | elapsed=5s | eta=3s | latest_scenario=512335f52b070c6587f8
[eval test w11] 1,500/2,503 ( 59.9%) | elapsed=5s | eta=3s | latest_scenario=f9463ea108399af0e38f
[eval test w04] 1,512/2,530 ( 59.8%) | elapsed=5s | eta=3s | latest_scenario=41145528e349d2755183
[eval test w07] 1,512/2,538 ( 59.6%) | elapsed=5s | eta=3s | latest_scenario=7e5662173ce3fcf0e65d
[eval test w05] 1,638/2,529 ( 64.8%) | elapsed=5s | eta=3s | latest_scenario=923a212897cd5d732c71
[eval test w02] 1,512/2,520 ( 60.0%) | elapsed=5s | eta=3s | latest_scenario=55c0b1a6952e2093c533
[eval test w06] 1,625/2,511 ( 64.7%) | elapsed=6s | eta=3s | latest_scenario=50d76dedcf1dc527eceb
[eval test w03] 1,625/2,509 ( 64.8%) | elapsed=5s | eta=3s | latest_scenario=dfe46680fe08089ca5c1
[eval test w00] 1,625/2,509 ( 64.8%) | elapsed=5s | eta=3s | latest_scenario=88d2c4069692abc3886d
[eval test w01] 1,625/2,509 ( 64.8%) | elapsed=6s | eta=3s | latest_scenario=d87cfb33bb2fae129679
[eval test w08] 1,625/2,508 ( 64.8%) | elapsed=5s | eta=3s | latest_scenario=cb092c7cdb75a77a8bd4
[eval test w10] 1,638/2,538 ( 64.5%) | elapsed=5s | eta=3s | latest_scenario=8c04ac6a46f1d7a17043
[eval test w09] 1,638/2,527 ( 64.8%) | elapsed=5s | eta=3s | latest_scenario=4b8d6a2468fbe53456e2
[eval test w04] 1,638/2,530 ( 64.7%) | elapsed=6s | eta=3s | latest_scenario=bf37e146485c8681f7e6
[eval test w11] 1,625/2,503 ( 64.9%) | elapsed=5s | eta=3s | latest_scenario=f3836b14a4276a6c3db4
[eval test w07] 1,638/2,538 ( 64.5%) | elapsed=5s | eta=3s | latest_scenario=f7a6aeb32cb2ad39f6d3
[eval test w05] 1,764/2,529 ( 69.8%) | elapsed=6s | eta=2s | latest_scenario=0863994340efdc238c75
[eval test w02] 1,638/2,520 ( 65.0%) | elapsed=6s | eta=3s | latest_scenario=1a2b66bf39326957880c
[eval test w03] 1,750/2,509 ( 69.7%) | elapsed=6s | eta=3s | latest_scenario=b33dada9b6dec8847892
[eval test w06] 1,750/2,511 ( 69.7%) | elapsed=6s | eta=3s | latest_scenario=102953c58830d97852fe
[eval test w01] 1,750/2,509 ( 69.7%) | elapsed=6s | eta=3s | latest_scenario=e4b7b6086a6be270592b
[eval test w00] 1,750/2,509 ( 69.7%) | elapsed=6s | eta=3s | latest_scenario=0d23f50711f151124540
[eval test w08] 1,750/2,508 ( 69.8%) | elapsed=6s | eta=2s | latest_scenario=e24bb7d1c08699d541bd
[eval test w10] 1,764/2,538 ( 69.5%) | elapsed=6s | eta=3s | latest_scenario=2bc29fa66529cfd56638
[eval test w04] 1,764/2,530 ( 69.7%) | elapsed=6s | eta=3s | latest_scenario=c78446141bc1066bd4d2
[eval test w09] 1,764/2,527 ( 69.8%) | elapsed=6s | eta=3s | latest_scenario=c6df35cff2f4cf7b6543
[eval test w11] 1,750/2,503 ( 69.9%) | elapsed=6s | eta=2s | latest_scenario=90b17e94a23418636f44
[eval test w07] 1,764/2,538 ( 69.5%) | elapsed=6s | eta=3s | latest_scenario=300f3c8a879541def49c
[eval test w05] 1,890/2,529 ( 74.7%) | elapsed=6s | eta=2s | latest_scenario=6a97d04e28c093a5d17e
[eval test w02] 1,764/2,520 ( 70.0%) | elapsed=6s | eta=3s | latest_scenario=4b0e7f1e63e49a5d156c
[eval test w03] 1,875/2,509 ( 74.7%) | elapsed=6s | eta=2s | latest_scenario=cf31d0d1c7d04050b407
[eval test w06] 1,875/2,511 ( 74.7%) | elapsed=6s | eta=2s | latest_scenario=48341f988c068c4d772b
[eval test w00] 1,875/2,509 ( 74.7%) | elapsed=6s | eta=2s | latest_scenario=b532034547373624a9ab
[eval test w01] 1,875/2,509 ( 74.7%) | elapsed=6s | eta=2s | latest_scenario=aff88615838500688705
[eval test w08] 1,875/2,508 ( 74.8%) | elapsed=6s | eta=2s | latest_scenario=6bf49b568650169dea1e
[eval test w10] 1,890/2,538 ( 74.5%) | elapsed=6s | eta=2s | latest_scenario=fd67b2281c6f29a41d06
[eval test w09] 1,890/2,527 ( 74.8%) | elapsed=6s | eta=2s | latest_scenario=2646331d9cb9f07d6a57
[eval test w04] 1,890/2,530 ( 74.7%) | elapsed=6s | eta=2s | latest_scenario=159d527b1e6a5f8edb85
[eval test w11] 1,875/2,503 ( 74.9%) | elapsed=6s | eta=2s | latest_scenario=d711cc76088eaecf7ba2
[eval test w07] 1,890/2,538 ( 74.5%) | elapsed=6s | eta=2s | latest_scenario=1a2ee5d3bb77d2e53f20
[eval test w05] 2,016/2,529 ( 79.7%) | elapsed=6s | eta=2s | latest_scenario=06b4ee35a59a5ca172c4
[eval test w02] 1,890/2,520 ( 75.0%) | elapsed=6s | eta=2s | latest_scenario=4bf8434a7389d175841f
[eval test w03] 2,000/2,509 ( 79.7%) | elapsed=7s | eta=2s | latest_scenario=10af8c1331c69ff4829f
[eval test w06] 2,000/2,511 ( 79.6%) | elapsed=7s | eta=2s | latest_scenario=934d8f0fc07127bb2419
[eval test w00] 2,000/2,509 ( 79.7%) | elapsed=7s | eta=2s | latest_scenario=1f537e2a1be5cb7dfed3
[eval test w08] 2,000/2,508 ( 79.7%) | elapsed=7s | eta=2s | latest_scenario=0a912ff9c2173fca6cdd
[eval test w01] 2,000/2,509 ( 79.7%) | elapsed=7s | eta=2s | latest_scenario=e9435da2ad5ce8b354d1
[eval test w10] 2,016/2,538 ( 79.4%) | elapsed=6s | eta=2s | latest_scenario=26fd03af5a6392fa4181
[eval test w09] 2,016/2,527 ( 79.8%) | elapsed=7s | eta=2s | latest_scenario=cd72940316eb70f5f62b
[eval test w04] 2,016/2,530 ( 79.7%) | elapsed=7s | eta=2s | latest_scenario=02a685301067abbb9079
[eval test w11] 2,000/2,503 ( 79.9%) | elapsed=7s | eta=2s | latest_scenario=eec692bd5f4b209553a3
[eval test w07] 2,016/2,538 ( 79.4%) | elapsed=7s | eta=2s | latest_scenario=1eabab54370a1a34cb1a
[eval test w05] 2,142/2,529 ( 84.7%) | elapsed=7s | eta=1s | latest_scenario=764bb885d799ca72aa15
[eval test w02] 2,016/2,520 ( 80.0%) | elapsed=7s | eta=2s | latest_scenario=3681d4b4daca4169f05d
[eval test w06] 2,125/2,511 ( 84.6%) | elapsed=7s | eta=1s | latest_scenario=e8a53f0af34500cf8fa9
[eval test w00] 2,125/2,509 ( 84.7%) | elapsed=7s | eta=1s | latest_scenario=5c206cf626204e302ad2
[eval test w03] 2,125/2,509 ( 84.7%) | elapsed=7s | eta=1s | latest_scenario=937081334dfecf8f939c
[eval test w08] 2,125/2,508 ( 84.7%) | elapsed=7s | eta=1s | latest_scenario=094c9b8e6c8804bc2b73
[eval test w10] 2,142/2,538 ( 84.4%) | elapsed=7s | eta=1s | latest_scenario=21c73f77e2981176b051
[eval test w01] 2,125/2,509 ( 84.7%) | elapsed=7s | eta=1s | latest_scenario=0a5193c1edc9a7b7af44
[eval test w09] 2,142/2,527 ( 84.8%) | elapsed=7s | eta=1s | latest_scenario=2d98b76b2475f943d8ed
[eval test w04] 2,142/2,530 ( 84.7%) | elapsed=7s | eta=1s | latest_scenario=f5560ae1500ddb9d552f
[eval test w11] 2,125/2,503 ( 84.9%) | elapsed=7s | eta=1s | latest_scenario=ca1bb34177773f980b70
[eval test w07] 2,142/2,538 ( 84.4%) | elapsed=7s | eta=1s | latest_scenario=060a026bb5ca4af71258
[eval test w05] 2,268/2,529 ( 89.7%) | elapsed=7s | eta=1s | latest_scenario=fe4fe34ddd84f8b94afd
[eval test w02] 2,142/2,520 ( 85.0%) | elapsed=7s | eta=1s | latest_scenario=3e8f8fd30a3a291af9f1
[eval test w06] 2,250/2,511 ( 89.6%) | elapsed=8s | eta=1s | latest_scenario=291588f85d8807ca941a
[eval test w00] 2,250/2,509 ( 89.7%) | elapsed=7s | eta=1s | latest_scenario=01d94effc47581302b72
[eval test w03] 2,250/2,509 ( 89.7%) | elapsed=8s | eta=1s | latest_scenario=c5a05c2ee18077a84d99
[eval test w08] 2,250/2,508 ( 89.7%) | elapsed=7s | eta=1s | latest_scenario=5b0db06772af975bfa46
[eval test w10] 2,268/2,538 ( 89.4%) | elapsed=7s | eta=1s | latest_scenario=b5a1ba2a1dcf7f396113
[eval test w01] 2,250/2,509 ( 89.7%) | elapsed=8s | eta=1s | latest_scenario=20ee58634d24a538b5bf
[eval test w09] 2,268/2,527 ( 89.8%) | elapsed=7s | eta=1s | latest_scenario=ad1936a969343a41ba3c
[eval test w04] 2,268/2,530 ( 89.6%) | elapsed=8s | eta=1s | latest_scenario=dfe7dad3720f8fdf4e99
[eval test w07] 2,268/2,538 ( 89.4%) | elapsed=7s | eta=1s | latest_scenario=434fea370ff8ca08fa2f
[eval test w11] 2,250/2,503 ( 89.9%) | elapsed=7s | eta=1s | latest_scenario=e6bdb45c498685aad7bb
[eval test w05] 2,394/2,529 ( 94.7%) | elapsed=8s | eta=0s | latest_scenario=87d654f320d6faafc6a9
[eval test w02] 2,268/2,520 ( 90.0%) | elapsed=8s | eta=1s | latest_scenario=3fdf796ecf199ac4d909
[eval test w03] 2,375/2,509 ( 94.7%) | elapsed=8s | eta=0s | latest_scenario=9473d2e2c92bdcda7637
[eval test w06] 2,375/2,511 ( 94.6%) | elapsed=8s | eta=0s | latest_scenario=53098e11491ef47b5fbe
[eval test w00] 2,375/2,509 ( 94.7%) | elapsed=8s | eta=0s | latest_scenario=12a0ca9584dec84f41de
[eval test w08] 2,375/2,508 ( 94.7%) | elapsed=8s | eta=0s | latest_scenario=b4f869d58f021997be84
[eval test w10] 2,394/2,538 ( 94.3%) | elapsed=8s | eta=0s | latest_scenario=86f317821c659d9d7aea
[eval test w01] 2,375/2,509 ( 94.7%) | elapsed=8s | eta=0s | latest_scenario=c60dcab7d4585bc15855
[eval test w09] 2,394/2,527 ( 94.7%) | elapsed=8s | eta=0s | latest_scenario=984fc63b214984c93954
[eval test w04] 2,394/2,530 ( 94.6%) | elapsed=8s | eta=0s | latest_scenario=8afdb18d2ae33029b8cc
[eval test w11] 2,375/2,503 ( 94.9%) | elapsed=8s | eta=0s | latest_scenario=1c70737b9bda3d595bd8
[eval test w07] 2,394/2,538 ( 94.3%) | elapsed=8s | eta=0s | latest_scenario=56b5ce77cbbbe37acef5
[eval test w05] 2,520/2,529 ( 99.6%) | elapsed=8s | eta=0s | latest_scenario=fb9a3a3855c92cb81836
[eval test w05] 2,529/2,529 (100.0%) | elapsed=8s | eta=0s | latest_scenario=c22c60c26ffdead87978
[eval test w02] 2,394/2,520 ( 95.0%) | elapsed=8s | eta=0s | latest_scenario=dd0c8549b0ac1929c808
[eval test w03] 2,500/2,509 ( 99.6%) | elapsed=8s | eta=0s | latest_scenario=9ceaa4f3b88da3d34bc0
[eval test w06] 2,500/2,511 ( 99.6%) | elapsed=8s | eta=0s | latest_scenario=8be16c2309d37e0ca441
[eval test w00] 2,500/2,509 ( 99.6%) | elapsed=8s | eta=0s | latest_scenario=8d7acdfd5a39b700b420
[eval test w03] 2,509/2,509 (100.0%) | elapsed=8s | eta=0s | latest_scenario=6f027998fd64febcbf99
[eval test w10] 2,520/2,538 ( 99.3%) | elapsed=8s | eta=0s | latest_scenario=dd4d8f5fc3da980e5f18
[eval test w06] 2,511/2,511 (100.0%) | elapsed=8s | eta=0s | latest_scenario=ceb6487f0e21a27d759d
[eval test w08] 2,500/2,508 ( 99.7%) | elapsed=8s | eta=0s | latest_scenario=4dab8c537d486750f6c9
[eval test w00] 2,509/2,509 (100.0%) | elapsed=8s | eta=0s | latest_scenario=ea41e96aaaaf6ddec02a
[eval test w01] 2,500/2,509 ( 99.6%) | elapsed=8s | eta=0s | latest_scenario=5423d86d633db9765d75
[eval test w09] 2,520/2,527 ( 99.7%) | elapsed=8s | eta=0s | latest_scenario=be85faf46157116c0748
[eval test w08] 2,508/2,508 (100.0%) | elapsed=8s | eta=0s | latest_scenario=33fe648425c13c6dc300
[eval test w01] 2,509/2,509 (100.0%) | elapsed=9s | eta=0s | latest_scenario=450d8a5394bf9e207c64
[eval test w09] 2,527/2,527 (100.0%) | elapsed=8s | eta=0s | latest_scenario=eb3de9c9d9ea0cb20bcf
[eval test w10] 2,538/2,538 (100.0%) | elapsed=8s | eta=0s | latest_scenario=5abf85b4c60bb64f9522
[eval test w04] 2,520/2,530 ( 99.6%) | elapsed=8s | eta=0s | latest_scenario=e90ff17cf4089ec5a3bc
[eval test w11] 2,500/2,503 ( 99.9%) | elapsed=8s | eta=0s | latest_scenario=9d06fb9c47d52454db73
[eval test w04] 2,530/2,530 (100.0%) | elapsed=8s | eta=0s | latest_scenario=c2a54da4227babb0d9bd
[eval test w11] 2,503/2,503 (100.0%) | elapsed=8s | eta=0s | latest_scenario=0f36bc5932b251133bb9
[eval test w07] 2,520/2,538 ( 99.3%) | elapsed=8s | eta=0s | latest_scenario=7b9a5b1638d3bc6c39c5
[eval test w07] 2,538/2,538 (100.0%) | elapsed=8s | eta=0s | latest_scenario=635ffcc7b3f74b69b8b4
[eval test w02] 2,520/2,520 (100.0%) | elapsed=8s | eta=0s | latest_scenario=8813d852163eef0ce6ae
[parallel window 05] done | elapsed=20s
[parallel windows] 1/28 (  3.6%) | elapsed=22s | eta=9.8m | latest_window=5 status=ok
[parallel window 00] done | elapsed=20s
[WARN] could not set torch thread counts: Error: cannot set number of interop threads after parallel work has started or set_num_interop_threads called
[compute] torch_threads=8 | blas_threads=4
[parallel windows] 2/28 (  7.1%) | elapsed=22s | eta=4.7m | latest_window=0 status=ok
[WARN] could not set torch thread counts: Error: cannot set number of interop threads after parallel work has started or set_num_interop_threads called
[compute] torch_threads=8 | blas_threads=4
[parallel window 10] done | elapsed=20s
[parallel window 08] done | elapsed=20s
[parallel windows] 3/28 ( 10.7%) | elapsed=22s | eta=3.0m | latest_window=10 status=ok
[parallel windows] 4/28 ( 14.3%) | elapsed=22s | eta=2.2m | latest_window=8 status=ok
[parallel window 03] done | elapsed=20s
[parallel windows] 5/28 ( 17.9%) | elapsed=22s | eta=1.7m | latest_window=3 status=ok
[parallel window 09] done | elapsed=20s
[WARN] could not set torch thread counts: Error: cannot set number of interop threads after parallel work has started or set_num_interop_threads called
[compute] torch_threads=8 | blas_threads=4
[parallel windows] 6/28 ( 21.4%) | elapsed=22s | eta=1.3m | latest_window=9 status=ok
[WARN] could not set torch thread counts: Error: cannot set number of interop threads after parallel work has started or set_num_interop_threads called
[compute] torch_threads=8 | blas_threads=4
[WARN] could not set torch thread counts: Error: cannot set number of interop threads after parallel work has started or set_num_interop_threads called
[compute] torch_threads=8 | blas_threads=4
[parallel window 11] done | elapsed=20s
[parallel windows] 7/28 ( 25.0%) | elapsed=22s | eta=1.1m | latest_window=11 status=ok
[WARN] could not set torch thread counts: Error: cannot set number of interop threads after parallel work has started or set_num_interop_threads called
[compute] torch_threads=8 | blas_threads=4
[parallel window 06] done | elapsed=20s
[parallel window 01] done | elapsed=20s
[parallel windows] 8/28 ( 28.6%) | elapsed=22s | eta=55s | latest_window=6 status=ok
[WARN] could not set torch thread counts: Error: cannot set number of interop threads after parallel work has started or set_num_interop_threads called
[compute] torch_threads=8 | blas_threads=4
[parallel windows] 9/28 ( 32.1%) | elapsed=22s | eta=47s | latest_window=1 status=ok
[WARN] could not set torch thread counts: Error: cannot set number of interop threads after parallel work has started or set_num_interop_threads called
[compute] torch_threads=8 | blas_threads=4
[WARN] could not set torch thread counts: Error: cannot set number of interop threads after parallel work has started or set_num_interop_threads called
[compute] torch_threads=8 | blas_threads=4
[parallel window 04] done | elapsed=20s
[parallel windows] 10/28 ( 35.7%) | elapsed=22s | eta=40s | latest_window=4 status=ok
[WARN] could not set torch thread counts: Error: cannot set number of interop threads after parallel work has started or set_num_interop_threads called
[compute] torch_threads=8 | blas_threads=4
[parallel window 07] done | elapsed=20s
[parallel windows] 11/28 ( 39.3%) | elapsed=22s | eta=34s | latest_window=7 status=ok
[WARN] could not set torch thread counts: Error: cannot set number of interop threads after parallel work has started or set_num_interop_threads called
[compute] torch_threads=8 | blas_threads=4
[parallel window 02] done | elapsed=20s
[parallel window 12] start | train=7,300 val=1,830 test=2,516
[parallel windows] 12/28 ( 42.9%) | elapsed=22s | eta=30s | latest_window=2 status=ok
Using cpu device
Logging to /Users/mohammadkarbalaei/University/Final Project/Final_Code/rl_outputs/sac_portfolio_lpm/opec_sac_lpm3aware_ROLL_T1000_train2.0y_val6.0m_test6.0m_hmin0.0_hmax1.25_dh0.05_wLPM0.45_wVOL0.35_wCOST0.2/window_12/tb/SAC_3
[parallel window 13] start | train=7,300 val=1,830 test=2,512
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 22.2     |
|    ep_rew_mean     | -2.06    |
| time/              |          |
|    episodes        | 4        |
|    fps             | 4917     |
|    time_elapsed    | 0        |
|    total_timesteps | 89       |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 20.9     |
|    ep_rew_mean     | -1.35    |
| time/              |          |
|    episodes        | 8        |
|    fps             | 5031     |
|    time_elapsed    | 0        |
|    total_timesteps | 167      |
---------------------------------
[WARN] could not set torch thread counts: Error: cannot set number of interop threads after parallel work has started or set_num_interop_threads called
[compute] torch_threads=8 | blas_threads=4
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 19.9     |
|    ep_rew_mean     | -1.23    |
| time/              |          |
|    episodes        | 12       |
|    fps             | 5260     |
|    time_elapsed    | 0        |
|    total_timesteps | 239      |
---------------------------------
Using cpu device
Logging to /Users/mohammadkarbalaei/University/Final Project/Final_Code/rl_outputs/sac_portfolio_lpm/opec_sac_lpm3aware_ROLL_T1000_train2.0y_val6.0m_test6.0m_hmin0.0_hmax1.25_dh0.05_wLPM0.45_wVOL0.35_wCOST0.2/window_13/tb/SAC_3
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 19.2     |
|    ep_rew_mean     | -0.773   |
| time/              |          |
|    episodes        | 4        |
|    fps             | 2756     |
|    time_elapsed    | 0        |
|    total_timesteps | 77       |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 20.3     |
|    ep_rew_mean     | -1.27    |
| time/              |          |
|    episodes        | 16       |
|    fps             | 5189     |
|    time_elapsed    | 0        |
|    total_timesteps | 325      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 20.4     |
|    ep_rew_mean     | -1.31    |
| time/              |          |
|    episodes        | 20       |
|    fps             | 5307     |
|    time_elapsed    | 0        |
|    total_timesteps | 407      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 19.8     |
|    ep_rew_mean     | -0.655   |
| time/              |          |
|    episodes        | 8        |
|    fps             | 3633     |
|    time_elapsed    | 0        |
|    total_timesteps | 158      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 20       |
|    ep_rew_mean     | -1.32    |
| time/              |          |
|    episodes        | 24       |
|    fps             | 5351     |
|    time_elapsed    | 0        |
|    total_timesteps | 481      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 19.4     |
|    ep_rew_mean     | -0.685   |
| time/              |          |
|    episodes        | 12       |
|    fps             | 4042     |
|    time_elapsed    | 0        |
|    total_timesteps | 233      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 19.9     |
|    ep_rew_mean     | -1.33    |
| time/              |          |
|    episodes        | 28       |
|    fps             | 5504     |
|    time_elapsed    | 0        |
|    total_timesteps | 557      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 19.9     |
|    ep_rew_mean     | -0.764   |
| time/              |          |
|    episodes        | 16       |
|    fps             | 3954     |
|    time_elapsed    | 0        |
|    total_timesteps | 318      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 19.8     |
|    ep_rew_mean     | -1.32    |
| time/              |          |
|    episodes        | 32       |
|    fps             | 5500     |
|    time_elapsed    | 0        |
|    total_timesteps | 635      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 19.6     |
|    ep_rew_mean     | -0.843   |
| time/              |          |
|    episodes        | 20       |
|    fps             | 4204     |
|    time_elapsed    | 0        |
|    total_timesteps | 393      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 19.9     |
|    ep_rew_mean     | -1.34    |
| time/              |          |
|    episodes        | 36       |
|    fps             | 5365     |
|    time_elapsed    | 0        |
|    total_timesteps | 716      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 20       |
|    ep_rew_mean     | -0.866   |
| time/              |          |
|    episodes        | 24       |
|    fps             | 4557     |
|    time_elapsed    | 0        |
|    total_timesteps | 480      |
---------------------------------
[parallel window 14] start | train=7,300 val=1,830 test=2,529
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 20.1     |
|    ep_rew_mean     | -1.34    |
| time/              |          |
|    episodes        | 40       |
|    fps             | 5495     |
|    time_elapsed    | 0        |
|    total_timesteps | 806      |
---------------------------------
Using cpu device
Logging to /Users/mohammadkarbalaei/University/Final Project/Final_Code/rl_outputs/sac_portfolio_lpm/opec_sac_lpm3aware_ROLL_T1000_train2.0y_val6.0m_test6.0m_hmin0.0_hmax1.25_dh0.05_wLPM0.45_wVOL0.35_wCOST0.2/window_14/tb/SAC_3
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 20.5     |
|    ep_rew_mean     | -1.46    |
| time/              |          |
|    episodes        | 44       |
|    fps             | 5620     |
|    time_elapsed    | 0        |
|    total_timesteps | 900      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 19.9     |
|    ep_rew_mean     | -0.857   |
| time/              |          |
|    episodes        | 28       |
|    fps             | 4359     |
|    time_elapsed    | 0        |
|    total_timesteps | 557      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 20.3     |
|    ep_rew_mean     | -1.42    |
| time/              |          |
|    episodes        | 48       |
|    fps             | 5747     |
|    time_elapsed    | 0        |
|    total_timesteps | 974      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 19.9     |
|    ep_rew_mean     | -0.911   |
| time/              |          |
|    episodes        | 32       |
|    fps             | 4637     |
|    time_elapsed    | 0        |
|    total_timesteps | 637      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 19.8     |
|    ep_rew_mean     | -1.89    |
| time/              |          |
|    episodes        | 4        |
|    fps             | 4849     |
|    time_elapsed    | 0        |
|    total_timesteps | 79       |
---------------------------------
 100% ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1,000/1,000  [ 0:00:00 < 0:00:00 , 5,882 it/s ]
[parallel window 15] start | train=7,300 val=1,830 test=2,525
[eval val w12] 0/1,830 (  0.0%) | elapsed=0s | eta=0s | starting evaluation
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 20.1     |
|    ep_rew_mean     | -0.998   |
| time/              |          |
|    episodes        | 36       |
|    fps             | 4692     |
|    time_elapsed    | 0        |
|    total_timesteps | 722      |
---------------------------------
[parallel window 16] start | train=7,300 val=1,830 test=2,517
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 19.6     |
|    ep_rew_mean     | -1.39    |
| time/              |          |
|    episodes        | 8        |
|    fps             | 4627     |
|    time_elapsed    | 0        |
|    total_timesteps | 157      |
---------------------------------
Using cpu device
Logging to /Users/mohammadkarbalaei/University/Final Project/Final_Code/rl_outputs/sac_portfolio_lpm/opec_sac_lpm3aware_ROLL_T1000_train2.0y_val6.0m_test6.0m_hmin0.0_hmax1.25_dh0.05_wLPM0.45_wVOL0.35_wCOST0.2/window_15/tb/SAC_3
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 19.6     |
|    ep_rew_mean     | -1.26    |
| time/              |          |
|    episodes        | 12       |
|    fps             | 4881     |
|    time_elapsed    | 0        |
|    total_timesteps | 235      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 20.1     |
|    ep_rew_mean     | -1.18    |
| time/              |          |
|    episodes        | 40       |
|    fps             | 4654     |
|    time_elapsed    | 0        |
|    total_timesteps | 802      |
---------------------------------
Using cpu device
Logging to /Users/mohammadkarbalaei/University/Final Project/Final_Code/rl_outputs/sac_portfolio_lpm/opec_sac_lpm3aware_ROLL_T1000_train2.0y_val6.0m_test6.0m_hmin0.0_hmax1.25_dh0.05_wLPM0.45_wVOL0.35_wCOST0.2/window_16/tb/SAC_3
[parallel window 17] start | train=7,300 val=1,830 test=2,517
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 20.1     |
|    ep_rew_mean     | -1.57    |
| time/              |          |
|    episodes        | 16       |
|    fps             | 5321     |
|    time_elapsed    | 0        |
|    total_timesteps | 322      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 20.5     |
|    ep_rew_mean     | -2.02    |
| time/              |          |
|    episodes        | 4        |
|    fps             | 4658     |
|    time_elapsed    | 0        |
|    total_timesteps | 82       |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 20.1     |
|    ep_rew_mean     | -1.33    |
| time/              |          |
|    episodes        | 44       |
|    fps             | 4762     |
|    time_elapsed    | 0        |
|    total_timesteps | 885      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 22       |
|    ep_rew_mean     | -11.1    |
| time/              |          |
|    episodes        | 4        |
|    fps             | 4050     |
|    time_elapsed    | 0        |
|    total_timesteps | 88       |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 20.1     |
|    ep_rew_mean     | -1.76    |
| time/              |          |
|    episodes        | 20       |
|    fps             | 5578     |
|    time_elapsed    | 0        |
|    total_timesteps | 401      |
---------------------------------
Using cpu device
Logging to /Users/mohammadkarbalaei/University/Final Project/Final_Code/rl_outputs/sac_portfolio_lpm/opec_sac_lpm3aware_ROLL_T1000_train2.0y_val6.0m_test6.0m_hmin0.0_hmax1.25_dh0.05_wLPM0.45_wVOL0.35_wCOST0.2/window_17/tb/SAC_3
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 20.2     |
|    ep_rew_mean     | -2.55    |
| time/              |          |
|    episodes        | 8        |
|    fps             | 5114     |
|    time_elapsed    | 0        |
|    total_timesteps | 162      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 19.8     |
|    ep_rew_mean     | -1.77    |
| time/              |          |
|    episodes        | 24       |
|    fps             | 5693     |
|    time_elapsed    | 0        |
|    total_timesteps | 474      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 19       |
|    ep_rew_mean     | -9.97    |
| time/              |          |
|    episodes        | 8        |
|    fps             | 4231     |
|    time_elapsed    | 0        |
|    total_timesteps | 152      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 20.3     |
|    ep_rew_mean     | -1.32    |
| time/              |          |
|    episodes        | 48       |
|    fps             | 4671     |
|    time_elapsed    | 0        |
|    total_timesteps | 975      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 18.8     |
|    ep_rew_mean     | -2       |
| time/              |          |
|    episodes        | 12       |
|    fps             | 5151     |
|    time_elapsed    | 0        |
|    total_timesteps | 226      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 25.2     |
|    ep_rew_mean     | -6.85    |
| time/              |          |
|    episodes        | 4        |
|    fps             | 6447     |
|    time_elapsed    | 0        |
|    total_timesteps | 101      |
---------------------------------
 100% ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1,000/1,000  [ 0:00:00 < 0:00:00 , 4,993 it/s ]
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 20.1     |
|    ep_rew_mean     | -10.9    |
| time/              |          |
|    episodes        | 12       |
|    fps             | 5054     |
|    time_elapsed    | 0        |
|    total_timesteps | 241      |
---------------------------------
[eval val w13] 0/1,830 (  0.0%) | elapsed=0s | eta=0s | starting evaluation
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 20.4     |
|    ep_rew_mean     | -1.77    |
| time/              |          |
|    episodes        | 28       |
|    fps             | 5657     |
|    time_elapsed    | 0        |
|    total_timesteps | 571      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 19.1     |
|    ep_rew_mean     | -2.99    |
| time/              |          |
|    episodes        | 16       |
|    fps             | 5135     |
|    time_elapsed    | 0        |
|    total_timesteps | 306      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 20.2     |
|    ep_rew_mean     | -11.1    |
| time/              |          |
|    episodes        | 16       |
|    fps             | 5419     |
|    time_elapsed    | 0        |
|    total_timesteps | 323      |
---------------------------------
[parallel window 18] start | train=7,300 val=1,830 test=2,514
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 25.1     |
|    ep_rew_mean     | -7.44    |
| time/              |          |
|    episodes        | 8        |
|    fps             | 5485     |
|    time_elapsed    | 0        |
|    total_timesteps | 201      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 18.4     |
|    ep_rew_mean     | -3.13    |
| time/              |          |
|    episodes        | 20       |
|    fps             | 4978     |
|    time_elapsed    | 0        |
|    total_timesteps | 368      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 24.1     |
|    ep_rew_mean     | -7.07    |
| time/              |          |
|    episodes        | 12       |
|    fps             | 5853     |
|    time_elapsed    | 0        |
|    total_timesteps | 289      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 20.7     |
|    ep_rew_mean     | -1.75    |
| time/              |          |
|    episodes        | 32       |
|    fps             | 5352     |
|    time_elapsed    | 0        |
|    total_timesteps | 663      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 20.9     |
|    ep_rew_mean     | -9.92    |
| time/              |          |
|    episodes        | 20       |
|    fps             | 5362     |
|    time_elapsed    | 0        |
|    total_timesteps | 418      |
---------------------------------
Using cpu device
Logging to /Users/mohammadkarbalaei/University/Final Project/Final_Code/rl_outputs/sac_portfolio_lpm/opec_sac_lpm3aware_ROLL_T1000_train2.0y_val6.0m_test6.0m_hmin0.0_hmax1.25_dh0.05_wLPM0.45_wVOL0.35_wCOST0.2/window_18/tb/SAC_3
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 18.8     |
|    ep_rew_mean     | -2.74    |
| time/              |          |
|    episodes        | 24       |
|    fps             | 5207     |
|    time_elapsed    | 0        |
|    total_timesteps | 452      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 20.5     |
|    ep_rew_mean     | -1.65    |
| time/              |          |
|    episodes        | 36       |
|    fps             | 5508     |
|    time_elapsed    | 0        |
|    total_timesteps | 739      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 20.4     |
|    ep_rew_mean     | -9.14    |
| time/              |          |
|    episodes        | 24       |
|    fps             | 5638     |
|    time_elapsed    | 0        |
|    total_timesteps | 490      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 19       |
|    ep_rew_mean     | -6.82    |
| time/              |          |
|    episodes        | 4        |
|    fps             | 5721     |
|    time_elapsed    | 0        |
|    total_timesteps | 76       |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 24.1     |
|    ep_rew_mean     | -7.27    |
| time/              |          |
|    episodes        | 16       |
|    fps             | 5881     |
|    time_elapsed    | 0        |
|    total_timesteps | 385      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 20.1     |
|    ep_rew_mean     | -8.33    |
| time/              |          |
|    episodes        | 28       |
|    fps             | 5871     |
|    time_elapsed    | 0        |
|    total_timesteps | 564      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 19.2     |
|    ep_rew_mean     | -2.92    |
| time/              |          |
|    episodes        | 28       |
|    fps             | 5346     |
|    time_elapsed    | 0        |
|    total_timesteps | 538      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 19.6     |
|    ep_rew_mean     | -5.04    |
| time/              |          |
|    episodes        | 8        |
|    fps             | 6786     |
|    time_elapsed    | 0        |
|    total_timesteps | 157      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 23.1     |
|    ep_rew_mean     | -6.85    |
| time/              |          |
|    episodes        | 20       |
|    fps             | 6122     |
|    time_elapsed    | 0        |
|    total_timesteps | 463      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 20.8     |
|    ep_rew_mean     | -1.57    |
| time/              |          |
|    episodes        | 40       |
|    fps             | 5527     |
|    time_elapsed    | 0        |
|    total_timesteps | 833      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 20.6     |
|    ep_rew_mean     | -7.77    |
| time/              |          |
|    episodes        | 32       |
|    fps             | 6071     |
|    time_elapsed    | 0        |
|    total_timesteps | 659      |
---------------------------------
[parallel window 20] start | train=7,300 val=1,830 test=2,520
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 19.5     |
|    ep_rew_mean     | -3.48    |
| time/              |          |
|    episodes        | 32       |
|    fps             | 5430     |
|    time_elapsed    | 0        |
|    total_timesteps | 623      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 20.1     |
|    ep_rew_mean     | -4.04    |
| time/              |          |
|    episodes        | 12       |
|    fps             | 6816     |
|    time_elapsed    | 0        |
|    total_timesteps | 241      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 20.6     |
|    ep_rew_mean     | -1.72    |
| time/              |          |
|    episodes        | 44       |
|    fps             | 5601     |
|    time_elapsed    | 0        |
|    total_timesteps | 908      |
---------------------------------
[parallel window 19] start | train=7,300 val=1,830 test=2,505
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 22.4     |
|    ep_rew_mean     | -6.1     |
| time/              |          |
|    episodes        | 24       |
|    fps             | 5859     |
|    time_elapsed    | 0        |
|    total_timesteps | 537      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 19.3     |
|    ep_rew_mean     | -3.37    |
| time/              |          |
|    episodes        | 36       |
|    fps             | 5606     |
|    time_elapsed    | 0        |
|    total_timesteps | 694      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 20.6     |
|    ep_rew_mean     | -7.59    |
| time/              |          |
|    episodes        | 36       |
|    fps             | 6035     |
|    time_elapsed    | 0        |
|    total_timesteps | 742      |
---------------------------------
Using cpu device
Logging to /Users/mohammadkarbalaei/University/Final Project/Final_Code/rl_outputs/sac_portfolio_lpm/opec_sac_lpm3aware_ROLL_T1000_train2.0y_val6.0m_test6.0m_hmin0.0_hmax1.25_dh0.05_wLPM0.45_wVOL0.35_wCOST0.2/window_20/tb/SAC_3
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 20.5     |
|    ep_rew_mean     | -1.76    |
| time/              |          |
|    episodes        | 48       |
|    fps             | 5669     |
|    time_elapsed    | 0        |
|    total_timesteps | 984      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 20.1     |
|    ep_rew_mean     | -4.21    |
| time/              |          |
|    episodes        | 16       |
|    fps             | 6282     |
|    time_elapsed    | 0        |
|    total_timesteps | 321      |
---------------------------------
 100% ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1,000/1,000  [ 0:00:00 < 0:00:00 , 5,699 it/s ]
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 22.4     |
|    ep_rew_mean     | -5.83    |
| time/              |          |
|    episodes        | 28       |
|    fps             | 5794     |
|    time_elapsed    | 0        |
|    total_timesteps | 627      |
---------------------------------
Using cpu device
Logging to /Users/mohammadkarbalaei/University/Final Project/Final_Code/rl_outputs/sac_portfolio_lpm/opec_sac_lpm3aware_ROLL_T1000_train2.0y_val6.0m_test6.0m_hmin0.0_hmax1.25_dh0.05_wLPM0.45_wVOL0.35_wCOST0.2/window_19/tb/SAC_3
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 19.3     |
|    ep_rew_mean     | -3.87    |
| time/              |          |
|    episodes        | 40       |
|    fps             | 5558     |
|    time_elapsed    | 0        |
|    total_timesteps | 771      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 18.5     |
|    ep_rew_mean     | -2.21    |
| time/              |          |
|    episodes        | 4        |
|    fps             | 5638     |
|    time_elapsed    | 0        |
|    total_timesteps | 74       |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 20.8     |
|    ep_rew_mean     | -7.11    |
| time/              |          |
|    episodes        | 40       |
|    fps             | 6051     |
|    time_elapsed    | 0        |
|    total_timesteps | 832      |
---------------------------------
[eval val w14] 0/1,830 (  0.0%) | elapsed=0s | eta=0s | starting evaluation
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 20.6     |
|    ep_rew_mean     | -5.07    |
| time/              |          |
|    episodes        | 20       |
|    fps             | 6482     |
|    time_elapsed    | 0        |
|    total_timesteps | 412      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 21       |
|    ep_rew_mean     | -3.21    |
| time/              |          |
|    episodes        | 4        |
|    fps             | 6082     |
|    time_elapsed    | 0        |
|    total_timesteps | 84       |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 19.4     |
|    ep_rew_mean     | -3.65    |
| time/              |          |
|    episodes        | 44       |
|    fps             | 5693     |
|    time_elapsed    | 0        |
|    total_timesteps | 853      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 22.2     |
|    ep_rew_mean     | -6.11    |
| time/              |          |
|    episodes        | 32       |
|    fps             | 5783     |
|    time_elapsed    | 0        |
|    total_timesteps | 710      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 19.1     |
|    ep_rew_mean     | -2.09    |
| time/              |          |
|    episodes        | 8        |
|    fps             | 6142     |
|    time_elapsed    | 0        |
|    total_timesteps | 153      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 20.7     |
|    ep_rew_mean     | -6.94    |
| time/              |          |
|    episodes        | 44       |
|    fps             | 6068     |
|    time_elapsed    | 0        |
|    total_timesteps | 912      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 20.5     |
|    ep_rew_mean     | -4.61    |
| time/              |          |
|    episodes        | 24       |
|    fps             | 6405     |
|    time_elapsed    | 0        |
|    total_timesteps | 492      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 20.4     |
|    ep_rew_mean     | -2.52    |
| time/              |          |
|    episodes        | 8        |
|    fps             | 6289     |
|    time_elapsed    | 0        |
|    total_timesteps | 163      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 19.4     |
|    ep_rew_mean     | -3.49    |
| time/              |          |
|    episodes        | 48       |
|    fps             | 5735     |
|    time_elapsed    | 0        |
|    total_timesteps | 931      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 22.3     |
|    ep_rew_mean     | -6.59    |
| time/              |          |
|    episodes        | 36       |
|    fps             | 5960     |
|    time_elapsed    | 0        |
|    total_timesteps | 802      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 20.8     |
|    ep_rew_mean     | -6.66    |
| time/              |          |
|    episodes        | 48       |
|    fps             | 6119     |
|    time_elapsed    | 0        |
|    total_timesteps | 997      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 20.1     |
|    ep_rew_mean     | -5.21    |
| time/              |          |
|    episodes        | 28       |
|    fps             | 6461     |
|    time_elapsed    | 0        |
|    total_timesteps | 564      |
---------------------------------
 100% ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1,000/1,000  [ 0:00:00 < 0:00:00 , 6,406 it/s ]
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 21.2     |
|    ep_rew_mean     | -2.94    |
| time/              |          |
|    episodes        | 12       |
|    fps             | 5945     |
|    time_elapsed    | 0        |
|    total_timesteps | 254      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 19.6     |
|    ep_rew_mean     | -2.05    |
| time/              |          |
|    episodes        | 12       |
|    fps             | 6512     |
|    time_elapsed    | 0        |
|    total_timesteps | 235      |
---------------------------------
 100% ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1,000/1,000  [ 0:00:00 < 0:00:00 , 5,979 it/s ]
[eval val w16] 0/1,830 (  0.0%) | elapsed=0s | eta=0s | starting evaluation
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 22.3     |
|    ep_rew_mean     | -6.37    |
| time/              |          |
|    episodes        | 40       |
|    fps             | 6018     |
|    time_elapsed    | 0        |
|    total_timesteps | 891      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 20.2     |
|    ep_rew_mean     | -4.84    |
| time/              |          |
|    episodes        | 32       |
|    fps             | 6515     |
|    time_elapsed    | 0        |
|    total_timesteps | 647      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 18.8     |
|    ep_rew_mean     | -2.33    |
| time/              |          |
|    episodes        | 16       |
|    fps             | 6696     |
|    time_elapsed    | 0        |
|    total_timesteps | 301      |
---------------------------------
[eval val w15] 0/1,830 (  0.0%) | elapsed=0s | eta=0s | starting evaluation
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 20.9     |
|    ep_rew_mean     | -2.96    |
| time/              |          |
|    episodes        | 16       |
|    fps             | 6026     |
|    time_elapsed    | 0        |
|    total_timesteps | 334      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 22       |
|    ep_rew_mean     | -6.09    |
| time/              |          |
|    episodes        | 44       |
|    fps             | 6042     |
|    time_elapsed    | 0        |
|    total_timesteps | 967      |
---------------------------------
[parallel window 21] start | train=7,300 val=1,830 test=2,513
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 19.1     |
|    ep_rew_mean     | -2.1     |
| time/              |          |
|    episodes        | 20       |
|    fps             | 6695     |
|    time_elapsed    | 0        |
|    total_timesteps | 382      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 20.4     |
|    ep_rew_mean     | -4.97    |
| time/              |          |
|    episodes        | 36       |
|    fps             | 6370     |
|    time_elapsed    | 0        |
|    total_timesteps | 734      |
---------------------------------
 100% ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1,000/1,000  [ 0:00:00 < 0:00:00 , 5,973 it/s ]
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 20.6     |
|    ep_rew_mean     | -2.72    |
| time/              |          |
|    episodes        | 20       |
|    fps             | 5545     |
|    time_elapsed    | 0        |
|    total_timesteps | 411      |
---------------------------------
[eval val w17] 0/1,830 (  0.0%) | elapsed=0s | eta=0s | starting evaluation
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 19.2     |
|    ep_rew_mean     | -2.15    |
| time/              |          |
|    episodes        | 24       |
|    fps             | 6601     |
|    time_elapsed    | 0        |
|    total_timesteps | 462      |
---------------------------------
Using cpu device
Logging to /Users/mohammadkarbalaei/University/Final Project/Final_Code/rl_outputs/sac_portfolio_lpm/opec_sac_lpm3aware_ROLL_T1000_train2.0y_val6.0m_test6.0m_hmin0.0_hmax1.25_dh0.05_wLPM0.45_wVOL0.35_wCOST0.2/window_21/tb/SAC_3
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 20.8     |
|    ep_rew_mean     | -2.61    |
| time/              |          |
|    episodes        | 24       |
|    fps             | 5601     |
|    time_elapsed    | 0        |
|    total_timesteps | 499      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 19.3     |
|    ep_rew_mean     | -2.37    |
| time/              |          |
|    episodes        | 28       |
|    fps             | 6678     |
|    time_elapsed    | 0        |
|    total_timesteps | 541      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 20.5     |
|    ep_rew_mean     | -5.28    |
| time/              |          |
|    episodes        | 40       |
|    fps             | 5959     |
|    time_elapsed    | 0        |
|    total_timesteps | 821      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 21.2     |
|    ep_rew_mean     | -122     |
| time/              |          |
|    episodes        | 4        |
|    fps             | 5103     |
|    time_elapsed    | 0        |
|    total_timesteps | 85       |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 20.4     |
|    ep_rew_mean     | -2.48    |
| time/              |          |
|    episodes        | 28       |
|    fps             | 5752     |
|    time_elapsed    | 0        |
|    total_timesteps | 572      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 20.8     |
|    ep_rew_mean     | -5.37    |
| time/              |          |
|    episodes        | 44       |
|    fps             | 6078     |
|    time_elapsed    | 0        |
|    total_timesteps | 914      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 21.4     |
|    ep_rew_mean     | -115     |
| time/              |          |
|    episodes        | 8        |
|    fps             | 5598     |
|    time_elapsed    | 0        |
|    total_timesteps | 171      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 19.9     |
|    ep_rew_mean     | -3.35    |
| time/              |          |
|    episodes        | 32       |
|    fps             | 6222     |
|    time_elapsed    | 0        |
|    total_timesteps | 637      |
---------------------------------
 100% ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1,000/1,000  [ 0:00:00 < 0:00:00 , 6,211 it/s ]
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 19.7     |
|    ep_rew_mean     | -2.25    |
| time/              |          |
|    episodes        | 32       |
|    fps             | 5350     |
|    time_elapsed    | 0        |
|    total_timesteps | 631      |
---------------------------------
[eval val w18] 0/1,830 (  0.0%) | elapsed=0s | eta=0s | starting evaluation
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 20       |
|    ep_rew_mean     | -3.4     |
| time/              |          |
|    episodes        | 36       |
|    fps             | 6016     |
|    time_elapsed    | 0        |
|    total_timesteps | 719      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 20.8     |
|    ep_rew_mean     | -77.1    |
| time/              |          |
|    episodes        | 12       |
|    fps             | 4797     |
|    time_elapsed    | 0        |
|    total_timesteps | 249      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 19.9     |
|    ep_rew_mean     | -2.2     |
| time/              |          |
|    episodes        | 36       |
|    fps             | 5438     |
|    time_elapsed    | 0        |
|    total_timesteps | 718      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 20       |
|    ep_rew_mean     | -3.49    |
| time/              |          |
|    episodes        | 40       |
|    fps             | 6083     |
|    time_elapsed    | 0        |
|    total_timesteps | 800      |
---------------------------------
[parallel window 22] start | train=7,300 val=1,830 test=2,509
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 20.6     |
|    ep_rew_mean     | -59.4    |
| time/              |          |
|    episodes        | 16       |
|    fps             | 5145     |
|    time_elapsed    | 0        |
|    total_timesteps | 330      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 20.1     |
|    ep_rew_mean     | -2.3     |
| time/              |          |
|    episodes        | 40       |
|    fps             | 5558     |
|    time_elapsed    | 0        |
|    total_timesteps | 802      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 19.8     |
|    ep_rew_mean     | -3.34    |
| time/              |          |
|    episodes        | 44       |
|    fps             | 6023     |
|    time_elapsed    | 0        |
|    total_timesteps | 869      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 21.1     |
|    ep_rew_mean     | -49.1    |
| time/              |          |
|    episodes        | 20       |
|    fps             | 5493     |
|    time_elapsed    | 0        |
|    total_timesteps | 421      |
---------------------------------
Using cpu device
Logging to /Users/mohammadkarbalaei/University/Final Project/Final_Code/rl_outputs/sac_portfolio_lpm/opec_sac_lpm3aware_ROLL_T1000_train2.0y_val6.0m_test6.0m_hmin0.0_hmax1.25_dh0.05_wLPM0.45_wVOL0.35_wCOST0.2/window_22/tb/SAC_3
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 20.3     |
|    ep_rew_mean     | -2.57    |
| time/              |          |
|    episodes        | 44       |
|    fps             | 5582     |
|    time_elapsed    | 0        |
|    total_timesteps | 895      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 19.9     |
|    ep_rew_mean     | -3.54    |
| time/              |          |
|    episodes        | 48       |
|    fps             | 6046     |
|    time_elapsed    | 0        |
|    total_timesteps | 955      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 20.8     |
|    ep_rew_mean     | -91      |
| time/              |          |
|    episodes        | 24       |
|    fps             | 5556     |
|    time_elapsed    | 0        |
|    total_timesteps | 498      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 20.2     |
|    ep_rew_mean     | -2.64    |
| time/              |          |
|    episodes        | 48       |
|    fps             | 5722     |
|    time_elapsed    | 0        |
|    total_timesteps | 969      |
---------------------------------
 100% ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1,000/1,000  [ 0:00:00 < 0:00:00 , 5,775 it/s ]
 100% ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1,000/1,000  [ 0:00:00 < 0:00:00 , 6,028 it/s ]
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 18.2     |
|    ep_rew_mean     | -0.858   |
| time/              |          |
|    episodes        | 4        |
|    fps             | 4271     |
|    time_elapsed    | 0        |
|    total_timesteps | 73       |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 21       |
|    ep_rew_mean     | -78.4    |
| time/              |          |
|    episodes        | 28       |
|    fps             | 5809     |
|    time_elapsed    | 0        |
|    total_timesteps | 589      |
---------------------------------
[eval val w19] 0/1,830 (  0.0%) | elapsed=0s | eta=0s | starting evaluation
[eval val w20] 0/1,830 (  0.0%) | elapsed=0s | eta=0s | starting evaluation
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 21       |
|    ep_rew_mean     | -2.28    |
| time/              |          |
|    episodes        | 8        |
|    fps             | 5172     |
|    time_elapsed    | 0        |
|    total_timesteps | 168      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 20.9     |
|    ep_rew_mean     | -100     |
| time/              |          |
|    episodes        | 32       |
|    fps             | 5799     |
|    time_elapsed    | 0        |
|    total_timesteps | 669      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 20.6     |
|    ep_rew_mean     | -89.2    |
| time/              |          |
|    episodes        | 36       |
|    fps             | 5765     |
|    time_elapsed    | 0        |
|    total_timesteps | 741      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 19.8     |
|    ep_rew_mean     | -2.43    |
| time/              |          |
|    episodes        | 12       |
|    fps             | 4787     |
|    time_elapsed    | 0        |
|    total_timesteps | 238      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 20.4     |
|    ep_rew_mean     | -80.4    |
| time/              |          |
|    episodes        | 40       |
|    fps             | 5728     |
|    time_elapsed    | 0        |
|    total_timesteps | 818      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 20.8     |
|    ep_rew_mean     | -4.2     |
| time/              |          |
|    episodes        | 16       |
|    fps             | 4830     |
|    time_elapsed    | 0        |
|    total_timesteps | 332      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 20.4     |
|    ep_rew_mean     | -73.2    |
| time/              |          |
|    episodes        | 44       |
|    fps             | 5662     |
|    time_elapsed    | 0        |
|    total_timesteps | 897      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 20.9     |
|    ep_rew_mean     | -14.3    |
| time/              |          |
|    episodes        | 20       |
|    fps             | 5003     |
|    time_elapsed    | 0        |
|    total_timesteps | 417      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 20.1     |
|    ep_rew_mean     | -67.3    |
| time/              |          |
|    episodes        | 48       |
|    fps             | 5727     |
|    time_elapsed    | 0        |
|    total_timesteps | 967      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 20.9     |
|    ep_rew_mean     | -12.8    |
| time/              |          |
|    episodes        | 24       |
|    fps             | 5270     |
|    time_elapsed    | 0        |
|    total_timesteps | 501      |
---------------------------------
 100% ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1,000/1,000  [ 0:00:00 < 0:00:00 , 5,656 it/s ]
[eval val w21] 0/1,830 (  0.0%) | elapsed=0s | eta=0s | starting evaluation
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 20.8     |
|    ep_rew_mean     | -11.5    |
| time/              |          |
|    episodes        | 28       |
|    fps             | 5375     |
|    time_elapsed    | 0        |
|    total_timesteps | 582      |
---------------------------------
[parallel window 23] start | train=7,300 val=1,830 test=2,535
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 20.3     |
|    ep_rew_mean     | -12.4    |
| time/              |          |
|    episodes        | 32       |
|    fps             | 5350     |
|    time_elapsed    | 0        |
|    total_timesteps | 650      |
---------------------------------
Using cpu device
Logging to /Users/mohammadkarbalaei/University/Final Project/Final_Code/rl_outputs/sac_portfolio_lpm/opec_sac_lpm3aware_ROLL_T1000_train2.0y_val6.0m_test6.0m_hmin0.0_hmax1.25_dh0.05_wLPM0.45_wVOL0.35_wCOST0.2/window_23/tb/SAC_3
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 20.6     |
|    ep_rew_mean     | -27.1    |
| time/              |          |
|    episodes        | 36       |
|    fps             | 5337     |
|    time_elapsed    | 0        |
|    total_timesteps | 741      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 18.5     |
|    ep_rew_mean     | -1.32    |
| time/              |          |
|    episodes        | 4        |
|    fps             | 5653     |
|    time_elapsed    | 0        |
|    total_timesteps | 74       |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 20.6     |
|    ep_rew_mean     | -24.6    |
| time/              |          |
|    episodes        | 40       |
|    fps             | 5317     |
|    time_elapsed    | 0        |
|    total_timesteps | 823      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 20.5     |
|    ep_rew_mean     | -2.52    |
| time/              |          |
|    episodes        | 8        |
|    fps             | 5884     |
|    time_elapsed    | 0        |
|    total_timesteps | 164      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 20.5     |
|    ep_rew_mean     | -22.5    |
| time/              |          |
|    episodes        | 44       |
|    fps             | 5423     |
|    time_elapsed    | 0        |
|    total_timesteps | 902      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 19.8     |
|    ep_rew_mean     | -18      |
| time/              |          |
|    episodes        | 12       |
|    fps             | 5618     |
|    time_elapsed    | 0        |
|    total_timesteps | 238      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 20.7     |
|    ep_rew_mean     | -20.9    |
| time/              |          |
|    episodes        | 48       |
|    fps             | 5603     |
|    time_elapsed    | 0        |
|    total_timesteps | 994      |
---------------------------------
 100% ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1,000/1,000  [ 0:00:00 < 0:00:00 , 5,687 it/s ]
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 20.2     |
|    ep_rew_mean     | -15      |
| time/              |          |
|    episodes        | 16       |
|    fps             | 5632     |
|    time_elapsed    | 0        |
|    total_timesteps | 323      |
---------------------------------
[eval val w22] 0/1,830 (  0.0%) | elapsed=0s | eta=0s | starting evaluation
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 20.4     |
|    ep_rew_mean     | -12.5    |
| time/              |          |
|    episodes        | 20       |
|    fps             | 5720     |
|    time_elapsed    | 0        |
|    total_timesteps | 407      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 20.2     |
|    ep_rew_mean     | -11.1    |
| time/              |          |
|    episodes        | 24       |
|    fps             | 5751     |
|    time_elapsed    | 0        |
|    total_timesteps | 486      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 20.9     |
|    ep_rew_mean     | -11.4    |
| time/              |          |
|    episodes        | 28       |
|    fps             | 5959     |
|    time_elapsed    | 0        |
|    total_timesteps | 586      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 20.6     |
|    ep_rew_mean     | -10.4    |
| time/              |          |
|    episodes        | 32       |
|    fps             | 5903     |
|    time_elapsed    | 0        |
|    total_timesteps | 660      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 20.5     |
|    ep_rew_mean     | -22.2    |
| time/              |          |
|    episodes        | 36       |
|    fps             | 6051     |
|    time_elapsed    | 0        |
|    total_timesteps | 739      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 20.9     |
|    ep_rew_mean     | -21.6    |
| time/              |          |
|    episodes        | 40       |
|    fps             | 6005     |
|    time_elapsed    | 0        |
|    total_timesteps | 834      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 20.7     |
|    ep_rew_mean     | -19.8    |
| time/              |          |
|    episodes        | 44       |
|    fps             | 6127     |
|    time_elapsed    | 0        |
|    total_timesteps | 912      |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 20.6     |
|    ep_rew_mean     | -18.3    |
| time/              |          |
|    episodes        | 48       |
|    fps             | 6208     |
|    time_elapsed    | 0        |
|    total_timesteps | 988      |
---------------------------------
 100% ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1,000/1,000  [ 0:00:00 < 0:00:00 , 6,273 it/s ]
[eval val w12] 183/1,830 ( 10.0%) | elapsed=1s | eta=5s | latest_scenario=c7f7c679e5988e5fa546
[eval val w23] 0/1,830 (  0.0%) | elapsed=0s | eta=0s | starting evaluation
[eval val w13] 183/1,830 ( 10.0%) | elapsed=1s | eta=5s | latest_scenario=d996bfbae5c409af88cb
[eval val w15] 183/1,830 ( 10.0%) | elapsed=1s | eta=5s | latest_scenario=b78c7c10d3fa35d38beb
[eval val w14] 183/1,830 ( 10.0%) | elapsed=1s | eta=5s | latest_scenario=aeae78eb476ed150d5e7
[eval val w16] 183/1,830 ( 10.0%) | elapsed=1s | eta=5s | latest_scenario=8f9d534e4a6e8ae97096
[eval val w17] 183/1,830 ( 10.0%) | elapsed=1s | eta=5s | latest_scenario=10f1c16cf89d2886970f
[eval val w18] 183/1,830 ( 10.0%) | elapsed=1s | eta=5s | latest_scenario=78664c63bd4684b6f59b
[eval val w19] 183/1,830 ( 10.0%) | elapsed=1s | eta=5s | latest_scenario=81da5877ccd36def9a6f
[eval val w20] 183/1,830 ( 10.0%) | elapsed=1s | eta=5s | latest_scenario=26f101c44bc8c6b1b49e
[eval val w21] 183/1,830 ( 10.0%) | elapsed=1s | eta=6s | latest_scenario=cfd9da1716b7f85456ec
[eval val w22] 183/1,830 ( 10.0%) | elapsed=1s | eta=5s | latest_scenario=609ac0c2d46c9ce8b433
[eval val w12] 366/1,830 ( 20.0%) | elapsed=1s | eta=5s | latest_scenario=96980eb60373c830fec9
[eval val w23] 183/1,830 ( 10.0%) | elapsed=1s | eta=6s | latest_scenario=2977811f52dc42bc569d
[eval val w13] 366/1,830 ( 20.0%) | elapsed=1s | eta=5s | latest_scenario=4e2356da631874d52aa9
[eval val w16] 366/1,830 ( 20.0%) | elapsed=1s | eta=5s | latest_scenario=5daf66d87e2e62cfe1eb
[eval val w15] 366/1,830 ( 20.0%) | elapsed=1s | eta=5s | latest_scenario=a4d17738899ace17301a
[eval val w14] 366/1,830 ( 20.0%) | elapsed=1s | eta=5s | latest_scenario=503df8c4fae6951c4107
[eval val w17] 366/1,830 ( 20.0%) | elapsed=1s | eta=5s | latest_scenario=b6cfcd080b4648776831
[eval val w18] 366/1,830 ( 20.0%) | elapsed=1s | eta=5s | latest_scenario=d888da5b19ce1528655f
[eval val w19] 366/1,830 ( 20.0%) | elapsed=1s | eta=5s | latest_scenario=0755c518290cbdd8526a
[eval val w20] 366/1,830 ( 20.0%) | elapsed=1s | eta=5s | latest_scenario=d6e469bcb699f2ff2530
[eval val w21] 366/1,830 ( 20.0%) | elapsed=1s | eta=5s | latest_scenario=0201b9b955a8b8b7a077
[eval val w22] 366/1,830 ( 20.0%) | elapsed=1s | eta=5s | latest_scenario=fd4ebe5dde758b524554
[eval val w12] 549/1,830 ( 30.0%) | elapsed=2s | eta=4s | latest_scenario=b536f4a9aac1a30eb513
[eval val w23] 366/1,830 ( 20.0%) | elapsed=1s | eta=5s | latest_scenario=3b7575caab37adfe508a
[eval val w13] 549/1,830 ( 30.0%) | elapsed=2s | eta=4s | latest_scenario=7b7812229714e9f8331e
[eval val w16] 549/1,830 ( 30.0%) | elapsed=2s | eta=4s | latest_scenario=84f65d91e9cfd00680ea
[eval val w15] 549/1,830 ( 30.0%) | elapsed=2s | eta=4s | latest_scenario=ecb79669690af39bdae4
[eval val w14] 549/1,830 ( 30.0%) | elapsed=2s | eta=4s | latest_scenario=33a026f50b5e5ba55381
[eval val w17] 549/1,830 ( 30.0%) | elapsed=2s | eta=4s | latest_scenario=15ff8e670ea4bf198dba
[eval val w18] 549/1,830 ( 30.0%) | elapsed=2s | eta=4s | latest_scenario=7d655ccd757e1ffccd30
[eval val w20] 549/1,830 ( 30.0%) | elapsed=2s | eta=4s | latest_scenario=af14dbd25bb9b6f8f523
[eval val w19] 549/1,830 ( 30.0%) | elapsed=2s | eta=4s | latest_scenario=e5dd3332987b08a39e48
[eval val w21] 549/1,830 ( 30.0%) | elapsed=2s | eta=4s | latest_scenario=23bd6a3f51af46cc4409
[eval val w22] 549/1,830 ( 30.0%) | elapsed=2s | eta=4s | latest_scenario=f54f95350359589ec233
[eval val w12] 732/1,830 ( 40.0%) | elapsed=2s | eta=4s | latest_scenario=2f5def0cb8fad3032148
[eval val w13] 732/1,830 ( 40.0%) | elapsed=2s | eta=4s | latest_scenario=1a9de01fe2722a728dc9
[eval val w23] 549/1,830 ( 30.0%) | elapsed=2s | eta=4s | latest_scenario=01c0d4b2899d5a8cef3e
[eval val w16] 732/1,830 ( 40.0%) | elapsed=2s | eta=4s | latest_scenario=38a4d2b654f8e1cce46e
[eval val w15] 732/1,830 ( 40.0%) | elapsed=2s | eta=4s | latest_scenario=96ab8a7406c57e0a9990
[eval val w14] 732/1,830 ( 40.0%) | elapsed=3s | eta=4s | latest_scenario=9342986ffc6beef583d4
[eval val w18] 732/1,830 ( 40.0%) | elapsed=2s | eta=4s | latest_scenario=0ec33ffff8f529261286
[eval val w17] 732/1,830 ( 40.0%) | elapsed=2s | eta=4s | latest_scenario=af1528a41a9495b004e8
[eval val w20] 732/1,830 ( 40.0%) | elapsed=2s | eta=4s | latest_scenario=20c25a1028018391b23d
[eval val w19] 732/1,830 ( 40.0%) | elapsed=3s | eta=4s | latest_scenario=6608451c413dd22c6528
[eval val w21] 732/1,830 ( 40.0%) | elapsed=3s | eta=4s | latest_scenario=9a3be7495daab4f6bab9
[eval val w22] 732/1,830 ( 40.0%) | elapsed=2s | eta=4s | latest_scenario=1f2c723e04cc0d2e842d
[eval val w23] 732/1,830 ( 40.0%) | elapsed=2s | eta=4s | latest_scenario=0c8c6c1ff232eda780d7
[eval val w12] 915/1,830 ( 50.0%) | elapsed=3s | eta=3s | latest_scenario=e69b023393db3f7839a5
[eval val w13] 915/1,830 ( 50.0%) | elapsed=3s | eta=3s | latest_scenario=de0b6e3a89cc582994f7
[eval val w16] 915/1,830 ( 50.0%) | elapsed=3s | eta=3s | latest_scenario=a00de17b016c93c1575d
[eval val w15] 915/1,830 ( 50.0%) | elapsed=3s | eta=3s | latest_scenario=1da538d06307dac61cea
[eval val w18] 915/1,830 ( 50.0%) | elapsed=3s | eta=3s | latest_scenario=23ea82080a02b9f34d6e
[eval val w14] 915/1,830 ( 50.0%) | elapsed=3s | eta=3s | latest_scenario=223d37b3a4a0f0ae2bdc
[eval val w17] 915/1,830 ( 50.0%) | elapsed=3s | eta=3s | latest_scenario=9038bfcbc2840d81171e
[eval val w20] 915/1,830 ( 50.0%) | elapsed=3s | eta=3s | latest_scenario=01219cadbd65319a3e0d
[eval val w19] 915/1,830 ( 50.0%) | elapsed=3s | eta=3s | latest_scenario=25b8566e0f545ff5b4f7
[eval val w22] 915/1,830 ( 50.0%) | elapsed=3s | eta=3s | latest_scenario=9f312bac15bc22511a5e
[eval val w21] 915/1,830 ( 50.0%) | elapsed=3s | eta=3s | latest_scenario=c9ed4297e5131e946a2c
[eval val w12] 1,098/1,830 ( 60.0%) | elapsed=4s | eta=2s | latest_scenario=87ea21e61162d67f3c75
[eval val w23] 915/1,830 ( 50.0%) | elapsed=3s | eta=3s | latest_scenario=934669d4f695632d2828
[eval val w13] 1,098/1,830 ( 60.0%) | elapsed=4s | eta=2s | latest_scenario=45178453e761dcda1ab9
[eval val w16] 1,098/1,830 ( 60.0%) | elapsed=4s | eta=2s | latest_scenario=29c7c6373b2c7db34992
[eval val w15] 1,098/1,830 ( 60.0%) | elapsed=4s | eta=2s | latest_scenario=1efe4bbfbb669efa4860
[eval val w18] 1,098/1,830 ( 60.0%) | elapsed=4s | eta=2s | latest_scenario=a83c589f4fdbaf77b8f3
[eval val w14] 1,098/1,830 ( 60.0%) | elapsed=4s | eta=3s | latest_scenario=ac4ae953eef2cb8fe6e9
[eval val w17] 1,098/1,830 ( 60.0%) | elapsed=4s | eta=2s | latest_scenario=7d49a8350a28fb444803
[eval val w20] 1,098/1,830 ( 60.0%) | elapsed=4s | eta=3s | latest_scenario=90bce47f73f2035434fb
[eval val w19] 1,098/1,830 ( 60.0%) | elapsed=4s | eta=3s | latest_scenario=142e603596b50f17d67f
[eval val w22] 1,098/1,830 ( 60.0%) | elapsed=4s | eta=2s | latest_scenario=a3c3f3c75ff1615295b4
[eval val w21] 1,098/1,830 ( 60.0%) | elapsed=4s | eta=3s | latest_scenario=e15c4ed726c4cd10ac7c
[eval val w12] 1,281/1,830 ( 70.0%) | elapsed=4s | eta=2s | latest_scenario=56a45f44aaa429388e28
[eval val w23] 1,098/1,830 ( 60.0%) | elapsed=4s | eta=2s | latest_scenario=4c4fefb61ef5714d9613
[eval val w13] 1,281/1,830 ( 70.0%) | elapsed=4s | eta=2s | latest_scenario=cf324b2a7b24a5049880
[eval val w16] 1,281/1,830 ( 70.0%) | elapsed=4s | eta=2s | latest_scenario=04c17e41fb40f2aca208
[eval val w15] 1,281/1,830 ( 70.0%) | elapsed=4s | eta=2s | latest_scenario=e821f4f45321d408fbbb
[eval val w18] 1,281/1,830 ( 70.0%) | elapsed=4s | eta=2s | latest_scenario=7af9408fea6b97d2d6bb
[eval val w14] 1,281/1,830 ( 70.0%) | elapsed=4s | eta=2s | latest_scenario=d2a19929cbdc5f662d1f
[eval val w17] 1,281/1,830 ( 70.0%) | elapsed=4s | eta=2s | latest_scenario=bcfcd675ed7eabfc63a9
[eval val w20] 1,281/1,830 ( 70.0%) | elapsed=4s | eta=2s | latest_scenario=a32a5b3fa71d37685e88
[eval val w19] 1,281/1,830 ( 70.0%) | elapsed=4s | eta=2s | latest_scenario=c1e87e95906cac44fc91
[eval val w22] 1,281/1,830 ( 70.0%) | elapsed=4s | eta=2s | latest_scenario=b63b0221276665447666
[eval val w21] 1,281/1,830 ( 70.0%) | elapsed=4s | eta=2s | latest_scenario=592b137af0a2c487f160
[eval val w12] 1,464/1,830 ( 80.0%) | elapsed=5s | eta=1s | latest_scenario=efae016ef65c975b6435
[eval val w23] 1,281/1,830 ( 70.0%) | elapsed=4s | eta=2s | latest_scenario=163562599e75fe2d2834
[eval val w13] 1,464/1,830 ( 80.0%) | elapsed=5s | eta=1s | latest_scenario=bfabf0e46b14134c2d24
[eval val w16] 1,464/1,830 ( 80.0%) | elapsed=5s | eta=1s | latest_scenario=ac37b53bf50d7f5b518d
[eval val w15] 1,464/1,830 ( 80.0%) | elapsed=5s | eta=1s | latest_scenario=a7cee11a0a76b82992b5
[eval val w18] 1,464/1,830 ( 80.0%) | elapsed=5s | eta=1s | latest_scenario=070ef92b2949889da108
[eval val w14] 1,464/1,830 ( 80.0%) | elapsed=5s | eta=1s | latest_scenario=0712da50ade718c7b352
[eval val w17] 1,464/1,830 ( 80.0%) | elapsed=5s | eta=1s | latest_scenario=dfef7c2a8a92c930af2e
[eval val w19] 1,464/1,830 ( 80.0%) | elapsed=5s | eta=1s | latest_scenario=c9796ed30e2f8b540307
[eval val w20] 1,464/1,830 ( 80.0%) | elapsed=5s | eta=1s | latest_scenario=3cf4c60d9bc51796578a
[eval val w21] 1,464/1,830 ( 80.0%) | elapsed=5s | eta=1s | latest_scenario=dbe215649ab5c522d699
[eval val w22] 1,464/1,830 ( 80.0%) | elapsed=5s | eta=1s | latest_scenario=9683bcc60396d4a7517a
[eval val w12] 1,647/1,830 ( 90.0%) | elapsed=6s | eta=1s | latest_scenario=64cedeaa933cf5eea426
[eval val w23] 1,464/1,830 ( 80.0%) | elapsed=5s | eta=1s | latest_scenario=817b4af7ccf92dd0059e
[eval val w13] 1,647/1,830 ( 90.0%) | elapsed=6s | eta=1s | latest_scenario=f1ead3dd8b4f9c76a284
[eval val w16] 1,647/1,830 ( 90.0%) | elapsed=6s | eta=1s | latest_scenario=32fc8d11982af9787bf9
[eval val w15] 1,647/1,830 ( 90.0%) | elapsed=6s | eta=1s | latest_scenario=02419df5b5c30e46758f
[eval val w14] 1,647/1,830 ( 90.0%) | elapsed=6s | eta=1s | latest_scenario=c14be26b2d8fa6656f9e
[eval val w18] 1,647/1,830 ( 90.0%) | elapsed=6s | eta=1s | latest_scenario=a2bb79314cf2f9df642c
[eval val w17] 1,647/1,830 ( 90.0%) | elapsed=6s | eta=1s | latest_scenario=fa43c5073560cde191c2
[eval val w19] 1,647/1,830 ( 90.0%) | elapsed=6s | eta=1s | latest_scenario=796a4dee6363c5cefca9
[eval val w20] 1,647/1,830 ( 90.0%) | elapsed=6s | eta=1s | latest_scenario=7310b50af34ab5d4be2e
[eval val w12] 1,830/1,830 (100.0%) | elapsed=6s | eta=0s | latest_scenario=0f36bc5932b251133bb9
[eval val w21] 1,647/1,830 ( 90.0%) | elapsed=6s | eta=1s | latest_scenario=6208d3eaec2f436b1536
[eval val w22] 1,647/1,830 ( 90.0%) | elapsed=6s | eta=1s | latest_scenario=c1747004244b5c637abd
[eval val w13] 1,830/1,830 (100.0%) | elapsed=6s | eta=0s | latest_scenario=fc6e5c1c6d0a4a6f8cf7
[eval val w23] 1,647/1,830 ( 90.0%) | elapsed=6s | eta=1s | latest_scenario=62090688695936010a65
[eval val w16] 1,830/1,830 (100.0%) | elapsed=6s | eta=0s | latest_scenario=6f8196611a673b609f05
[eval test w12] 0/2,516 (  0.0%) | elapsed=0s | eta=0s | starting evaluation
[eval val w18] 1,830/1,830 (100.0%) | elapsed=6s | eta=0s | latest_scenario=71ffd48096511a7bf8a7
[eval val w15] 1,830/1,830 (100.0%) | elapsed=6s | eta=0s | latest_scenario=f3267b910f9b10ae1c37
[eval val w14] 1,830/1,830 (100.0%) | elapsed=6s | eta=0s | latest_scenario=ee4f4f68edcc87a17a2f
[eval val w17] 1,830/1,830 (100.0%) | elapsed=6s | eta=0s | latest_scenario=f79a4cf333c54ddee8f3
[eval val w19] 1,830/1,830 (100.0%) | elapsed=6s | eta=0s | latest_scenario=94acc978f32d1c653900
[eval test w13] 0/2,512 (  0.0%) | elapsed=0s | eta=0s | starting evaluation
[eval val w20] 1,830/1,830 (100.0%) | elapsed=6s | eta=0s | latest_scenario=76699a65035733a1d987
[eval val w21] 1,830/1,830 (100.0%) | elapsed=6s | eta=0s | latest_scenario=80b81e082b66648cee7d
[eval val w22] 1,830/1,830 (100.0%) | elapsed=6s | eta=0s | latest_scenario=cac8faec61766d3e8ad5
[eval test w16] 0/2,517 (  0.0%) | elapsed=0s | eta=0s | starting evaluation
[eval test w12] 125/2,516 (  5.0%) | elapsed=0s | eta=9s | latest_scenario=20e829924ce4f27d5248
[eval test w18] 0/2,514 (  0.0%) | elapsed=0s | eta=0s | starting evaluation
[eval test w15] 0/2,525 (  0.0%) | elapsed=0s | eta=0s | starting evaluation
[eval test w14] 0/2,529 (  0.0%) | elapsed=0s | eta=0s | starting evaluation
[eval val w23] 1,830/1,830 (100.0%) | elapsed=6s | eta=0s | latest_scenario=f85a8888a952e317b813
[eval test w17] 0/2,517 (  0.0%) | elapsed=0s | eta=0s | starting evaluation
[eval test w13] 125/2,512 (  5.0%) | elapsed=0s | eta=8s | latest_scenario=d0a31ccad827dbaa089e
[eval test w19] 0/2,505 (  0.0%) | elapsed=0s | eta=0s | starting evaluation
[eval test w20] 0/2,520 (  0.0%) | elapsed=0s | eta=0s | starting evaluation
[eval test w21] 0/2,513 (  0.0%) | elapsed=0s | eta=0s | starting evaluation
[eval test w22] 0/2,509 (  0.0%) | elapsed=0s | eta=0s | starting evaluation
[eval test w16] 125/2,517 (  5.0%) | elapsed=0s | eta=8s | latest_scenario=8d5da74e45eaff9c123e
[eval test w18] 125/2,514 (  5.0%) | elapsed=0s | eta=7s | latest_scenario=1f65296c24fa666af2ac
[eval test w12] 250/2,516 (  9.9%) | elapsed=1s | eta=8s | latest_scenario=617231e556b031d4b9d6
[eval test w15] 126/2,525 (  5.0%) | elapsed=0s | eta=8s | latest_scenario=8ef4aa2f5ad1cb038d48
[eval test w23] 0/2,535 (  0.0%) | elapsed=0s | eta=0s | starting evaluation
[eval test w14] 126/2,529 (  5.0%) | elapsed=0s | eta=8s | latest_scenario=b18fdde2f027496fcd52
[eval test w17] 125/2,517 (  5.0%) | elapsed=0s | eta=8s | latest_scenario=c38ca5b6c8782e5601d1
[eval test w13] 250/2,512 ( 10.0%) | elapsed=1s | eta=7s | latest_scenario=7bfff1a90a198f9bd719
[eval test w19] 125/2,505 (  5.0%) | elapsed=0s | eta=7s | latest_scenario=2bf090798c9e65c7d26f
[eval test w20] 126/2,520 (  5.0%) | elapsed=0s | eta=8s | latest_scenario=3867c523b075afb0b2bb
[eval test w21] 125/2,513 (  5.0%) | elapsed=0s | eta=8s | latest_scenario=b745416c862039df6b92
[eval test w22] 125/2,509 (  5.0%) | elapsed=0s | eta=8s | latest_scenario=89a35a000103d0812a5f
[eval test w18] 250/2,514 (  9.9%) | elapsed=1s | eta=7s | latest_scenario=81da5877ccd36def9a6f
[eval test w12] 375/2,516 ( 14.9%) | elapsed=1s | eta=7s | latest_scenario=d207e1cd028ff4ff0066
[eval test w16] 250/2,517 (  9.9%) | elapsed=1s | eta=7s | latest_scenario=bc61bdb44ff9a5fcb5b7
[eval test w23] 126/2,535 (  5.0%) | elapsed=0s | eta=7s | latest_scenario=52b27258ccc9ab2c06ae
[eval test w15] 252/2,525 ( 10.0%) | elapsed=1s | eta=7s | latest_scenario=f8430af955ee317d4ef1
[eval test w14] 252/2,529 ( 10.0%) | elapsed=1s | eta=7s | latest_scenario=25d428efdaabdfe2db47
[eval test w17] 250/2,517 (  9.9%) | elapsed=1s | eta=8s | latest_scenario=a48be3acc7aaf282589b
[eval test w19] 250/2,505 ( 10.0%) | elapsed=1s | eta=7s | latest_scenario=26f101c44bc8c6b1b49e
[eval test w13] 375/2,512 ( 14.9%) | elapsed=1s | eta=7s | latest_scenario=125858574d7a83a8c6d4
[eval test w20] 252/2,520 ( 10.0%) | elapsed=1s | eta=7s | latest_scenario=d360c141e2fb4b30198b
[eval test w21] 250/2,513 (  9.9%) | elapsed=1s | eta=8s | latest_scenario=9b4f9b239ce452aa8d1b
[eval test w22] 250/2,509 ( 10.0%) | elapsed=1s | eta=8s | latest_scenario=15643fa39673df09f03c
[eval test w18] 375/2,514 ( 14.9%) | elapsed=1s | eta=7s | latest_scenario=fa6429d609781d22e9ef
[eval test w16] 375/2,517 ( 14.9%) | elapsed=1s | eta=7s | latest_scenario=4b860273dcc5f8d1ca78
[eval test w12] 500/2,516 ( 19.9%) | elapsed=2s | eta=7s | latest_scenario=54b39f334a42b74fba66
[eval test w15] 378/2,525 ( 15.0%) | elapsed=1s | eta=7s | latest_scenario=368fd1e0f02fa19d1cc0
[eval test w23] 252/2,535 (  9.9%) | elapsed=1s | eta=7s | latest_scenario=b770312c4250e30273f5
[eval test w14] 378/2,529 ( 14.9%) | elapsed=1s | eta=7s | latest_scenario=5269be5d863727963f2f
[eval test w17] 375/2,517 ( 14.9%) | elapsed=1s | eta=7s | latest_scenario=49c0df4fca91139a55aa
[eval test w19] 375/2,505 ( 15.0%) | elapsed=1s | eta=7s | latest_scenario=4061433b6c019f1805cd
[eval test w13] 500/2,512 ( 19.9%) | elapsed=2s | eta=7s | latest_scenario=1957e66b7a1ba9bb8280
[eval test w20] 378/2,520 ( 15.0%) | elapsed=1s | eta=7s | latest_scenario=ee2b49ff32cb2a484bc8
[eval test w18] 500/2,514 ( 19.9%) | elapsed=2s | eta=6s | latest_scenario=4c66a1f6ec750393e741
[eval test w21] 375/2,513 ( 14.9%) | elapsed=1s | eta=7s | latest_scenario=7a22af5d94049e7155ce
[eval test w22] 375/2,509 ( 14.9%) | elapsed=1s | eta=7s | latest_scenario=d6c8c4bb596d5eaa6f3a
[eval test w16] 500/2,517 ( 19.9%) | elapsed=2s | eta=7s | latest_scenario=5270238c7cd6179dcec7
[eval test w23] 378/2,535 ( 14.9%) | elapsed=1s | eta=7s | latest_scenario=04a539dc6d8e3648f60d
[eval test w15] 504/2,525 ( 20.0%) | elapsed=2s | eta=7s | latest_scenario=e8587dfef6f200b6c5ee
[eval test w12] 625/2,516 ( 24.8%) | elapsed=2s | eta=6s | latest_scenario=5f44a02962931ef477e3
[eval test w14] 504/2,529 ( 19.9%) | elapsed=2s | eta=6s | latest_scenario=476ca637eb74b7454b5c
[eval test w17] 500/2,517 ( 19.9%) | elapsed=2s | eta=7s | latest_scenario=909a59812ae11d4cb109
[eval test w19] 500/2,505 ( 20.0%) | elapsed=2s | eta=6s | latest_scenario=862a5612ca7fac6cae2c
[eval test w13] 625/2,512 ( 24.9%) | elapsed=2s | eta=6s | latest_scenario=16c67ff727bd1f201db6
[eval test w20] 504/2,520 ( 20.0%) | elapsed=2s | eta=7s | latest_scenario=0201b9b955a8b8b7a077
[eval test w21] 500/2,513 ( 19.9%) | elapsed=2s | eta=7s | latest_scenario=84a39299177dc67f600b
[eval test w18] 625/2,514 ( 24.9%) | elapsed=2s | eta=6s | latest_scenario=014add46168cb12cc673
[eval test w22] 500/2,509 ( 19.9%) | elapsed=2s | eta=7s | latest_scenario=6982296056eb10270c1b
[eval test w16] 625/2,517 ( 24.8%) | elapsed=2s | eta=6s | latest_scenario=fe49d766cbe88a7fd321
[eval test w12] 750/2,516 ( 29.8%) | elapsed=3s | eta=6s | latest_scenario=9138608a4aafa45dda80
[eval test w23] 504/2,535 ( 19.9%) | elapsed=2s | eta=7s | latest_scenario=e7f25993b00b03881367
[eval test w15] 630/2,525 ( 25.0%) | elapsed=2s | eta=6s | latest_scenario=00d514d1a369d09e9f4f
[eval test w14] 630/2,529 ( 24.9%) | elapsed=2s | eta=6s | latest_scenario=988da7fee7bf309a3f8d
[eval test w19] 625/2,505 ( 25.0%) | elapsed=2s | eta=6s | latest_scenario=358bd4432b85834d457a
[eval test w13] 750/2,512 ( 29.9%) | elapsed=2s | eta=6s | latest_scenario=df1e225e197343410f91
[eval test w20] 630/2,520 ( 25.0%) | elapsed=2s | eta=6s | latest_scenario=ff9093a660417fbf939f
[eval test w17] 625/2,517 ( 24.8%) | elapsed=2s | eta=7s | latest_scenario=cef7e86ff2481b0339cd
[eval test w21] 625/2,513 ( 24.9%) | elapsed=2s | eta=6s | latest_scenario=5ad50cbcc0453ce56a87
[eval test w18] 750/2,514 ( 29.8%) | elapsed=2s | eta=6s | latest_scenario=bd054caf5c540db690fa
[eval test w22] 625/2,509 ( 24.9%) | elapsed=2s | eta=6s | latest_scenario=5658c25d5949e666d91d
[eval test w12] 875/2,516 ( 34.8%) | elapsed=3s | eta=5s | latest_scenario=ccf96b5668d6c9c03401
[eval test w16] 750/2,517 ( 29.8%) | elapsed=2s | eta=6s | latest_scenario=f937fb88215ae06cb761
[eval test w23] 630/2,535 ( 24.9%) | elapsed=2s | eta=6s | latest_scenario=e5a968f62ef75b274568
[eval test w15] 756/2,525 ( 29.9%) | elapsed=3s | eta=6s | latest_scenario=4d37be829b766f80808a
[eval test w14] 756/2,529 ( 29.9%) | elapsed=3s | eta=6s | latest_scenario=7ca63fd63f87d44e6fc1
[eval test w19] 750/2,505 ( 29.9%) | elapsed=2s | eta=6s | latest_scenario=7855b86b76ca25d2c374
[eval test w13] 875/2,512 ( 34.8%) | elapsed=3s | eta=5s | latest_scenario=a9aab2de862c38d7efb8
[eval test w17] 750/2,517 ( 29.8%) | elapsed=3s | eta=6s | latest_scenario=7afec83df722a8f26b65
[eval test w20] 756/2,520 ( 30.0%) | elapsed=3s | eta=6s | latest_scenario=1d99d0fe6e2118a3267f
[eval test w21] 750/2,513 ( 29.8%) | elapsed=3s | eta=6s | latest_scenario=61896cd63c138f3edf7e
[eval test w18] 875/2,514 ( 34.8%) | elapsed=3s | eta=5s | latest_scenario=18fbb68fa685c9f5b2d5
[eval test w16] 875/2,517 ( 34.8%) | elapsed=3s | eta=5s | latest_scenario=323cf28c4c959c826633
[eval test w22] 750/2,509 ( 29.9%) | elapsed=3s | eta=6s | latest_scenario=9bad71934bd284466b75
[eval test w12] 1,000/2,516 ( 39.7%) | elapsed=3s | eta=5s | latest_scenario=cbe802b199b082c7face
[eval test w23] 756/2,535 ( 29.8%) | elapsed=2s | eta=6s | latest_scenario=da92780906da7171eb41
[eval test w15] 882/2,525 ( 34.9%) | elapsed=3s | eta=6s | latest_scenario=61d396378dd5047dfb9c
[eval test w14] 882/2,529 ( 34.9%) | elapsed=3s | eta=6s | latest_scenario=e7aa8f5945cf587484ca
[eval test w19] 875/2,505 ( 34.9%) | elapsed=3s | eta=5s | latest_scenario=a4dcf8d1851608d07efd
[eval test w13] 1,000/2,512 ( 39.8%) | elapsed=3s | eta=5s | latest_scenario=be167df709c69b121c33
[eval test w17] 875/2,517 ( 34.8%) | elapsed=3s | eta=6s | latest_scenario=28d29e6c4747d83c6621
[eval test w20] 882/2,520 ( 35.0%) | elapsed=3s | eta=5s | latest_scenario=8c4c48bbac9c67dc69f1
[eval test w18] 1,000/2,514 ( 39.8%) | elapsed=3s | eta=5s | latest_scenario=3ef43d20eb2ed4e3704f
[eval test w21] 875/2,513 ( 34.8%) | elapsed=3s | eta=5s | latest_scenario=651e588fd450df9ed80f
[eval test w12] 1,125/2,516 ( 44.7%) | elapsed=4s | eta=5s | latest_scenario=2e947870f1bd25f41f99
[eval test w16] 1,000/2,517 ( 39.7%) | elapsed=3s | eta=5s | latest_scenario=1ef9ae28f308d29519a6
[eval test w22] 875/2,509 ( 34.9%) | elapsed=3s | eta=5s | latest_scenario=3e822ecb5f112ec99e81
[eval test w23] 882/2,535 ( 34.8%) | elapsed=3s | eta=5s | latest_scenario=60f2599bc0750dbfdd69
[eval test w15] 1,008/2,525 ( 39.9%) | elapsed=3s | eta=5s | latest_scenario=9cdf14a986f8f79300a5
[eval test w14] 1,008/2,529 ( 39.9%) | elapsed=3s | eta=5s | latest_scenario=e9172e66527bd52feced
[eval test w19] 1,000/2,505 ( 39.9%) | elapsed=3s | eta=5s | latest_scenario=acd705ee8f906eed271a
[eval test w13] 1,125/2,512 ( 44.8%) | elapsed=4s | eta=5s | latest_scenario=8b92b661b05bfa26fe85
[eval test w17] 1,000/2,517 ( 39.7%) | elapsed=3s | eta=5s | latest_scenario=e0e7e87020f7aaa41b72
[eval test w20] 1,008/2,520 ( 40.0%) | elapsed=3s | eta=5s | latest_scenario=39fa6011da535a07010d
[eval test w18] 1,125/2,514 ( 44.7%) | elapsed=4s | eta=4s | latest_scenario=c41bc7de4a406bc9c49c
[eval test w21] 1,000/2,513 ( 39.8%) | elapsed=3s | eta=5s | latest_scenario=75d8f1b5b1f3bbd73618
[eval test w12] 1,250/2,516 ( 49.7%) | elapsed=4s | eta=4s | latest_scenario=a2ea56a4ae9a0112e231
[eval test w16] 1,125/2,517 ( 44.7%) | elapsed=4s | eta=5s | latest_scenario=f5e4d6f0037560d4f0d1
[eval test w22] 1,000/2,509 ( 39.9%) | elapsed=3s | eta=5s | latest_scenario=37f189bba01167c8a3eb
[eval test w15] 1,134/2,525 ( 44.9%) | elapsed=4s | eta=5s | latest_scenario=1652fd73b3bce52b4ad6
[eval test w23] 1,008/2,535 ( 39.8%) | elapsed=3s | eta=5s | latest_scenario=e00a6cb4f16e6366b66a
[eval test w14] 1,134/2,529 ( 44.8%) | elapsed=4s | eta=5s | latest_scenario=95a5bf3118851aa3bc1a
[eval test w19] 1,125/2,505 ( 44.9%) | elapsed=4s | eta=5s | latest_scenario=cc57a3e81d325236f3ff
[eval test w13] 1,250/2,512 ( 49.8%) | elapsed=4s | eta=4s | latest_scenario=f47ea1c6afab14799aa2
[eval test w17] 1,125/2,517 ( 44.7%) | elapsed=4s | eta=5s | latest_scenario=3101f5961a5fdc073e2c
[eval test w20] 1,134/2,520 ( 45.0%) | elapsed=4s | eta=5s | latest_scenario=ac3ee30682af46547cc0
[WARN] could not set torch thread counts: Error: cannot set number of interop threads after parallel work has started or set_num_interop_threads called
[compute] torch_threads=8 | blas_threads=4
[WARN] could not set torch thread counts: Error: cannot set number of interop threads after parallel work has started or set_num_interop_threads called
[compute] torch_threads=8 | blas_threads=4
[WARN] could not set torch thread counts: Error: cannot set number of interop threads after parallel work has started or set_num_interop_threads called
[compute] torch_threads=8 | blas_threads=4
[WARN] could not set torch thread counts: Error: cannot set number of interop threads after parallel work has started or set_num_interop_threads called
[compute] torch_threads=8 | blas_threads=4

====================================================================================================
[RUN 1/8] type=policy_grid weights=(0.45, 0.35, 0.2) delta_h=0.05 gamma=0.99 timesteps=1000
python -m rl.train_sac_portfolio_costaware --asset OPEC --rolling-windows --timesteps 1000 --parallel-windows 12 --torch-threads 8 --blas-threads 4 --h-min 0.0 --h-max 1.25 --delta-h 0.05 --gamma 0.99 --reward-weight-lpm 0.45 --reward-weight-volatility 0.35 --reward-weight-decision-cost 0.2 --plot-fraction 0.0
====================================================================================================
