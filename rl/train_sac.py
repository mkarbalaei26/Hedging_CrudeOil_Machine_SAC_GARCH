"""
train_sac.py - SAC training with walk-forward expanding window.

Based on train_walkforward.py but adapted for SAC algorithm.
Minimal changes to preserve existing infrastructure.
"""

import sys
import os
from pathlib import Path
import yaml
import numpy as np
import torch
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# Stable-baselines3 SAC
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.monitor import Monitor

# Import v2 environment
from rl.env_daily_v2 import OilHedgingDailyEnv_v2, EnvConfigV2

# Import existing utilities (scenario loader, data adapter, etc.)
# TODO: incompatible with current project scenario_loader
# from rl.scenario_loader import ScenarioSpec, load_scenario
# TODO: incompatible with current project data_adapter
# from data_adapter import load_precomputed_data


class WalkForwardSACTrainer:
    """
    Walk-forward trainer for SAC with expanding window.
    
    Structure mirrors train_walkforward.py but uses SAC instead of PPO.
    """
    
    def __init__(self, config_path: str):
        """Load config and initialize trainer."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.project_root = Path(self.config['project']['root_dir'])
        self.results_dir = self.project_root / 'results' / 'sac'
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Device
        self.device = torch.device(
            self.config['training'].get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        )
        
        print(f"[SAC Trainer] Using device: {self.device}")
        print(f"[SAC Trainer] Results dir: {self.results_dir}")
    
    def create_env(
        self,
        scenario_spec: ScenarioSpec,
        train_start: str,
        train_end: str,
        is_eval: bool = False
    ) -> OilHedgingDailyEnv_v2:
        """Create a single environment instance."""
        
        # Load precomputed data
        data = load_precomputed_data(
            data_dir=self.project_root / 'data',
            start_date=train_start,
            end_date=train_end
        )
        
        # Build env config
        env_cfg = EnvConfigV2(
            # Data
            spot=data['spot'],
            dS=data['dS'],
            f_mark=data['f_mark'],
            pnl_1c=data['pnl_1c'],
            roll_flag=data['roll_flag'],
            tradable=data['tradable'],
            feature_matrix=data['feature_matrix'],
            dates=data['dates'],
            
            # Scenario
            Q=scenario_spec.Q,
            base_policy=scenario_spec.base_policy,
            naive_hedge_ratio=scenario_spec.naive_hedge_ratio,
            
            # Action space (v2 bounds)
            action_mode=self.config['env']['action_mode'],
            delta_h_bounds=[-0.5, 0.5],  # Changed from [-2.0, 2.0]
            
            # Reward v2 params
            mu_paper=self.config['reward_v2']['mu_paper'],
            lambda_down=self.config['reward_v2']['lambda_down'],
            downside_order=self.config['reward_v2']['downside_order'],
            dd_threshold=self.config['reward_v2']['dd_threshold'],
            dd_beta=self.config['reward_v2']['dd_beta'],
            eta_cost=self.config['reward_v2']['eta_cost'],
            kappa_stability=self.config['reward_v2']['kappa_stability'],
            omega_inaction=self.config['reward_v2']['omega_inaction'],
            alpha_inaction=self.config['reward_v2']['alpha_inaction'],
            n_loss_max=self.config['reward_v2']['n_loss_max'],
            dd_panic_threshold=self.config['reward_v2']['dd_panic_threshold'],
            
            # Other env settings
            include_position=self.config['env'].get('include_position', True),
            include_time=self.config['env'].get('include_time', True),
            include_equity=self.config['env'].get('include_equity', True),
        )
        
        env = OilHedgingDailyEnv_v2(cfg=env_cfg)
        
        if not is_eval:
            env = Monitor(env)
        
        return env
    
    def create_vec_env(
        self,
        scenario_spec: ScenarioSpec,
        train_start: str,
        train_end: str,
        n_envs: int = 1,
        is_eval: bool = False
    ):
        """Create vectorized environment."""
        
        if n_envs == 1:
            env = self.create_env(scenario_spec, train_start, train_end, is_eval)
            return DummyVecEnv([lambda: env])
        else:
            # Parallel envs
            env_fns = [
                lambda: self.create_env(scenario_spec, train_start, train_end, is_eval)
                for _ in range(n_envs)
            ]
            return SubprocVecEnv(env_fns)
    
    def train_window(
        self,
        window_idx: int,
        train_start: str,
        train_end: str,
        val_start: str,
        val_end: str,
        prev_model_path: Optional[Path] = None
    ) -> Tuple[Path, Dict]:
        """
        Train SAC on one expanding window.
        
        Args:
            window_idx: Window index
            train_start, train_end: Training period
            val_start, val_end: Validation period
            prev_model_path: Path to previous window's model for warm-start
        
        Returns:
            (best_model_path, val_metrics)
        """
        
        print(f"\n{'='*60}")
        print(f"Window {window_idx}: Train [{train_start} → {train_end}]")
        print(f"                Val   [{val_start} → {val_end}]")
        print(f"{'='*60}\n")
        
        # Create window dir
        window_dir = self.results_dir / f'window_{window_idx:02d}'
        window_dir.mkdir(parents=True, exist_ok=True)
        
        # Load scenario
        scenario_spec = load_scenario(self.config['scenario']['name'])
        
        # Create train env
        train_env = self.create_vec_env(
            scenario_spec,
            train_start,
            train_end,
            n_envs=self.config['training'].get('n_envs', 1),
            is_eval=False
        )
        
        # Create eval env
        eval_env = self.create_vec_env(
            scenario_spec,
            val_start,
            val_end,
            n_envs=1,
            is_eval=True
        )
        
        # SAC hyperparameters
        sac_params = self.config['sac_params']
        
        # Initialize or load model
        if prev_model_path is not None and self.config['training'].get('warm_start', True):
            print(f"[SAC] Warm-starting from: {prev_model_path}")
            model = SAC.load(
                str(prev_model_path),
                env=train_env,
                device=self.device,
                # Reset some params for new window
                learning_rate=sac_params['learning_rate'],
                buffer_size=sac_params['buffer_size'],
            )
            # Note: replay buffer is NOT transferred (SAC limitation in SB3)
            # Each window starts with empty buffer
        else:
            print("[SAC] Training from scratch")
            model = SAC(
                policy="MlpPolicy",
                env=train_env,
                learning_rate=sac_params['learning_rate'],
                buffer_size=sac_params['buffer_size'],
                learning_starts=sac_params['learning_starts'],
                batch_size=sac_params['batch_size'],
                tau=sac_params['tau'],
                gamma=sac_params['gamma'],
                train_freq=sac_params['train_freq'],
                gradient_steps=sac_params['gradient_steps'],
                ent_coef=sac_params['ent_coef'],
                policy_kwargs=dict(net_arch=sac_params['net_arch']),
                verbose=1,
                device=self.device,
                tensorboard_log=str(window_dir / 'tensorboard'),
            )
        
        # Callbacks
        checkpoint_callback = CheckpointCallback(
            save_freq=sac_params.get('checkpoint_freq', 10000),
            save_path=str(window_dir / 'checkpoints'),
            name_prefix='sac_model',
        )
        
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=str(window_dir),
            log_path=str(window_dir / 'eval_logs'),
            eval_freq=sac_params.get('eval_freq', 10000),
            n_eval_episodes=sac_params.get('n_eval_episodes', 5),
            deterministic=True,
            render=False,
        )
        
        # Train
        total_timesteps = sac_params['total_timesteps']
        print(f"[SAC] Training for {total_timesteps} timesteps...")
        
        model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_callback, eval_callback],
            log_interval=10,
            tb_log_name=f'window_{window_idx:02d}',
        )
        
        # Save final model
        final_model_path = window_dir / 'model_final.zip'
        model.save(str(final_model_path))
        print(f"[SAC] Saved final model: {final_model_path}")
        
        # Best model path (from eval callback)
        best_model_path = window_dir / 'best_model.zip'
        
        # Validation metrics (read from eval callback logs)
        val_metrics = self._load_val_metrics(window_dir / 'eval_logs')
        
        # Save window summary
        summary = {
            'window_idx': window_idx,
            'train_period': [train_start, train_end],
            'val_period': [val_start, val_end],
            'total_timesteps': total_timesteps,
            'val_metrics': val_metrics,
        }
        
        with open(window_dir / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        train_env.close()
        eval_env.close()
        
        return best_model_path, val_metrics
    
    def _load_val_metrics(self, eval_log_dir: Path) -> Dict:
        """Load validation metrics from eval callback logs."""
        # Placeholder: parse evaluations.npz or results.json
        # For now, return empty dict
        return {}
    
    def run_walkforward(self):
        """Run full walk-forward training."""
        
        # Define windows (example: yearly expanding)
        windows = self.config['walkforward']['windows']
        
        prev_model_path = None
        all_results = []
        
        for i, window in enumerate(windows):
            best_model_path, val_metrics = self.train_window(
                window_idx=i,
                train_start=window['train_start'],
                train_end=window['train_end'],
                val_start=window['val_start'],
                val_end=window['val_end'],
                prev_model_path=prev_model_path,
            )
            
            all_results.append({
                'window': i,
                'val_metrics': val_metrics,
            })
            
            # Update for next window
            prev_model_path = best_model_path
        
        # Save overall summary
        with open(self.results_dir / 'walkforward_summary.json', 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\n{'='*60}")
        print("Walk-forward training complete!")
        print(f"Results saved to: {self.results_dir}")
        print(f"{'='*60}\n")


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python train_sac.py <config_path>")
        sys.exit(1)
    
    config_path = sys.argv[1]
    
    trainer = WalkForwardSACTrainer(config_path)
    trainer.run_walkforward()


if __name__ == '__main__':
    main()
