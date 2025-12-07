import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import yfinance as yf
import ta
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, DummyVecEnv
from stable_baselines3.common.noise import NormalActionNoise
import torch
import multiprocessing
import time
import os
from typing import Dict, Tuple, List

def set_all_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class TradingEnv(gym.Env):
    def __init__(self, df: pd.DataFrame, initial_balance: float = 10000):
        super().__init__()
        
        self.prices = df['Close'].values
        self.features = df[['RSI', 'MACD', 'Volatility', 'Ratio', 'Body']].values
        self.data_len = len(self.prices)
        
        self.initial_balance = initial_balance
        self.transaction_cost = 0.001
        
        self.action_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32
        )
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Buffer logic to avoid ValueError on short data
        buffer = 300 if self.data_len > 500 else 10
        high = self.data_len - buffer
        
        if high <= 50:
            self.current_step = 50
        else:
            self.current_step = np.random.randint(50, high)
            
        self.max_steps = 250
        self.steps_taken = 0
        
        self.cash = self.initial_balance
        self.shares = 0
        self.prev_value = self.initial_balance
        
        return self._get_obs(), {}
    
    def _get_obs(self):
        market = self.features[self.current_step]
        price = self.prices[self.current_step]
        portfolio_value = self.cash + self.shares * price
        weight = (self.shares * price) / portfolio_value if portfolio_value > 0 else 0
        pnl = (portfolio_value - self.prev_value) / self.prev_value
        
        return np.concatenate([market, [weight, pnl, price / 1000]]).astype(np.float32)
    
    def step(self, action):
        target_weight = float(np.clip(action[0], 0, 1))
        
        price = self.prices[self.current_step]
        portfolio_value = self.cash + self.shares * price
        
        target_shares_value = portfolio_value * target_weight
        current_shares_value = self.shares * price
        diff = target_shares_value - current_shares_value
        
        if diff > 0:  # Buy
            cost = diff * (1 + self.transaction_cost)
            if self.cash >= cost:
                self.cash -= cost
                self.shares += diff / price
        elif diff < 0:  # Sell
            sell_value = abs(diff)
            if self.shares * price >= sell_value:
                self.shares -= sell_value / price
                self.cash += sell_value * (1 - self.transaction_cost)
        
        self.current_step += 1
        self.steps_taken += 1
        
        new_price = self.prices[self.current_step]
        new_value = self.cash + self.shares * new_price
        
        reward = ((new_value - portfolio_value) / portfolio_value) * 100
        self.prev_value = new_value
        
        done = self.steps_taken >= self.max_steps or self.current_step >= self.data_len - 1
        
        return self._get_obs(), reward, done, False, {'value': new_value}

def load_data(ticker: str = "SPY") -> Tuple[pd.DataFrame, pd.DataFrame]:
    print(f"Downloading {ticker}")
    
    df = yf.download(ticker, start="2015-04-28", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    df['Body'] = df['Close'] - df['Open']
    df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
    df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
    df['MACD'] = ta.trend.macd_diff(df['Close'])
    df['Volatility'] = df['Close'].rolling(20).std()
    df['Ratio'] = (df['Close'] - df['SMA_20']) / (df['SMA_20'] + 1e-8)
    
    df = df.dropna().reset_index(drop=True)
    
    split_idx = len(df) - 500
    train = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:].copy()
    
    print(f"Train: {len(train)}, Test: {len(test)}")
    return train, test

def train_model(name, model_class, train_df, n_envs=16, timesteps=200000, **kwargs):
    start = time.time()
    
    def make_env():
        return lambda: TradingEnv(train_df)
    
    env = SubprocVecEnv([make_env() for _ in range(n_envs)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    
    model = model_class("MlpPolicy", env, verbose=0, device='cuda', **kwargs)
    
    model.learn(total_timesteps=timesteps, log_interval=100)
    
    elapsed = time.time() - start
    
    obs_rms = env.obs_rms
    env.close()
    
    return model, obs_rms, elapsed

def evaluate_model(model, obs_rms, test_df):
    class EvalEnv(TradingEnv):
        def reset(self, seed=None, options=None):
            super(TradingEnv, self).reset(seed=seed)
            self.current_step = 0
            self.max_steps = len(test_df) - 1
            self.steps_taken = 0
            self.cash = self.initial_balance
            self.shares = 0
            self.prev_value = self.initial_balance
            return self._get_obs(), {}
    
    eval_env = DummyVecEnv([lambda: EvalEnv(test_df)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    eval_env.training = False
    eval_env.obs_rms = obs_rms
    
    obs = eval_env.reset()
    portfolio_values = [10000]
    done = False
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, dones, info = eval_env.step(action)
        done = dones[0]
        portfolio_values.append(info[0]['value'])
    
    final_return = (portfolio_values[-1] - 10000) / 10000 * 100
    eval_env.close()
    return portfolio_values, final_return

def run_competition(seed, train_df, test_df, n_envs, timesteps):
    print(f"Seed: {seed}")
    set_all_seeds(seed)
    
    if not os.path.exists('models'):
        os.makedirs('models')
    
    results = {}
    timings = {}
    
    print("  Training PPO")
    model_ppo, obs_rms_ppo, time_ppo = train_model(
        "PPO", PPO, train_df, n_envs, timesteps,
        learning_rate=3e-4, n_steps=2048, batch_size=256, n_epochs=10, ent_coef=0.01, gamma=0.99
    )
    save_path = 'models/ppo_competition.pt'
    torch.save({
        'model_state': model_ppo.policy.state_dict(),
        'obs_mean': obs_rms_ppo.mean,
        'obs_var': obs_rms_ppo.var,
        'obs_count': obs_rms_ppo.count
    }, save_path)
    
    vals_ppo, ret_ppo = evaluate_model(model_ppo, obs_rms_ppo, test_df)
    results['PPO'] = ret_ppo
    timings['PPO'] = time_ppo
    print(f"PPO: {ret_ppo:.2f}% ({time_ppo:.1f}s)")
    
    print("  Training SAC")
    model_sac, obs_rms_sac, time_sac = train_model(
        "SAC", SAC, train_df, n_envs, timesteps,
        learning_rate=3e-4, buffer_size=100000, batch_size=512, ent_coef='auto', gamma=0.99, tau=0.005
    )
    save_path = 'models/sac_competition.pt'
    torch.save({
        'model_state': model_sac.policy.state_dict(),
        'obs_mean': obs_rms_sac.mean,
        'obs_var': obs_rms_sac.var,
        'obs_count': obs_rms_sac.count
    }, save_path)

    vals_sac, ret_sac = evaluate_model(model_sac, obs_rms_sac, test_df)
    results['SAC'] = ret_sac
    timings['SAC'] = time_sac
    print(f"SAC: {ret_sac:.2f}% ({time_sac:.1f}s)")
    
    print("  Training TD3")
    n_actions = 1
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
    
    model_td3, obs_rms_td3, time_td3 = train_model(
        "TD3", TD3, train_df, n_envs, timesteps,
        learning_rate=1e-3, buffer_size=100000, batch_size=512, action_noise=action_noise, gamma=0.99, tau=0.005
    )
    save_path = 'models/td3_competition.pt'
    torch.save({
        'model_state': model_td3.policy.state_dict(),
        'obs_mean': obs_rms_td3.mean,
        'obs_var': obs_rms_td3.var,
        'obs_count': obs_rms_td3.count
    }, save_path)

    vals_td3, ret_td3 = evaluate_model(model_td3, obs_rms_td3, test_df)
    results['TD3'] = ret_td3
    timings['TD3'] = time_td3
    print(f"TD3: {ret_td3:.2f}% ({time_td3:.1f}s)")
    
    total_time = time_ppo + time_sac + time_td3
    print(f"Run time: {total_time:.1f}s ({total_time/60:.1f} min)")
    
    return results, timings

# ======================================================
# Main
# ======================================================
if __name__ == '__main__':
    multiprocessing.freeze_support()
    torch.backends.cudnn.benchmark = True
    
    print("MULTI-SEED competition")
    
    # 데이터를 메인 프로세스에서 한 번 로드
    train_df, test_df = load_data("SPY")
    
    n_runs = 5
    n_envs = 16
    timesteps = 200000
    
    np.random.seed(int(time.time()))
    seeds = [np.random.randint(1, 100000) for _ in range(n_runs)]
    
    print(f"\nSettings:")
    print(f"  Runs: {n_runs}")
    print(f"  Random Seeds: {seeds}")
    print(f"  Envs: {n_envs}")
    print(f"  Timesteps: {timesteps:,}")
    
    all_results = {'PPO': [], 'SAC': [], 'TD3': []}
    all_timings = {'PPO': [], 'SAC': [], 'TD3': []}
    
    competition_start = time.time()
    
    for i, seed in enumerate(seeds, 1):
        print(f"\n{'#'*60}")
        print(f"# RUN {i}/{n_runs}")
        print(f"{'#'*60}")
        
        results, timings = run_competition(seed, train_df, test_df, n_envs, timesteps)
        
        for model_name in ['PPO', 'SAC', 'TD3']:
            all_results[model_name].append(results[model_name])
            all_timings[model_name].append(timings[model_name])
    
    total_time = time.time() - competition_start
    
    # Statistics
    print("\nFINAL STATISTICS")
    stats = {}
    for model_name in ['PPO', 'SAC', 'TD3']:
        returns = np.array(all_results[model_name])
        times = np.array(all_timings[model_name])
        
        stats[model_name] = {
            'mean': returns.mean(),
            'std': returns.std(),
            'min': returns.min(),
            'max': returns.max(),
            'avg_time': times.mean()
        }
        
        print(f"\n{model_name}:")
        print(f"  Mean: {stats[model_name]['mean']:.2f}% (±{stats[model_name]['std']:.2f}%)")
        print(f"  Range: {stats[model_name]['min']:.2f}% to {stats[model_name]['max']:.2f}%")
        print(f"  All Results: {[f'{r:.2f}%' for r in returns]}")
    
    winner = max(stats.items(), key=lambda x: x[1]['mean'])
    print(f"\nMOST CONSISTENT: {winner[0]}")
    print(f"  Average: {winner[1]['mean']:.2f}% (±{winner[1]['std']:.2f}%)")
    
    print(f"\nTotal Time: {total_time/60:.1f} minutes")
    print(f"  Per Run: {total_time/n_runs/60:.1f} minutes")
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Box plot
    ax = axes[0]
    data = [all_results['PPO'], all_results['SAC'], all_results['TD3']]
    bp = ax.boxplot(data, labels=['PPO', 'SAC', 'TD3'], patch_artist=True)
    
    colors = ['blue', 'green', 'red']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    ax.set_ylabel('Return (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'Return Distribution ({n_runs} seeds)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Bar chart
    ax = axes[1]
    models = ['PPO', 'SAC', 'TD3']
    means = [stats[m]['mean'] for m in models]
    stds = [stats[m]['std'] for m in models]
    
    bars = ax.bar(models, means, yerr=stds, capsize=10, 
                  color=colors, edgecolor='black', alpha=0.7, linewidth=2)
    
    ax.set_ylabel('Mean Return (%)', fontsize=12, fontweight='bold')
    ax.set_title('Average Performance (±std)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std,
                f'{mean:.2f}%\n±{std:.2f}%',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    if not os.path.exists('results'):
        os.makedirs('results')
    plt.savefig('results/multi_seed_comparison.png', dpi=300)
    print(f"\nSaved: results/multi_seed_comparison.png")
    
    # Save CSV
    results_df = pd.DataFrame(all_results)
    results_df['seed'] = seeds
    results_df.to_csv('results/multi_seed_results.csv', index=False)
    print(f"Saved: results/multi_seed_results.csv")