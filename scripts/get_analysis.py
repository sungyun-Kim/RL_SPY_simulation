import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import torch
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.running_mean_std import RunningMeanStd
import yfinance as yf
import ta
import warnings
warnings.filterwarnings('ignore')

import gymnasium as gym
from gymnasium import spaces

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
        self.current_step = 0
        self.max_steps = len(self.prices) - 1
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
        
        if diff > 0:
            cost = diff * (1 + self.transaction_cost)
            if self.cash >= cost:
                self.cash -= cost
                self.shares += diff / price
        elif diff < 0:
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

def load_data():
    print("Loading data")
    df = yf.download("SPY", start="2015-04-28", progress=False)
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
    return df.iloc[split_idx:].copy()

def load_competition_model(name):
    
    checkpoint = torch.load(f'models/{name.lower()}_competition.pt', weights_only=False)
    
    # Load test data to create env
    test_df = load_data()
    
    # Create model
    env = DummyVecEnv([lambda: TradingEnv(test_df)])
    
    if name == 'PPO':
        model = PPO("MlpPolicy", env, device='cuda')
    elif name == 'SAC':
        model = SAC("MlpPolicy", env, device='cuda')
    elif name == 'TD3':
        model = TD3("MlpPolicy", env, device='cuda')
    
    # Load weights
    model.policy.load_state_dict(checkpoint['model_state'])
    model.policy.eval()
    
    # Create obs_rms
    obs_rms = RunningMeanStd(shape=(8,))
    obs_rms.mean = checkpoint['obs_mean']
    obs_rms.var = checkpoint['obs_var']
    obs_rms.count = checkpoint['obs_count']
    
    env.close()
    
    print(f"Loaded {name}")
    return model, obs_rms

def collect_observations(model, obs_rms, test_df, n_samples=200):
    
    env = DummyVecEnv([lambda: TradingEnv(test_df)])
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    env.training = False
    env.obs_rms = obs_rms
    
    obs_list = []
    action_list = []
    
    obs = env.reset()
    
    for _ in range(min(n_samples, len(test_df) - 2)):
        obs_list.append(obs[0].copy())
        action, _ = model.predict(obs, deterministic=True)
        action_list.append(action[0])
        obs, _, done, _ = env.step(action)
        
        if done[0]:
            break
    
    env.close()
    
    return np.array(obs_list), np.array(action_list)

def calculate_feature_importance(model, test_df, obs_rms):
    print(f"Calculating feature importance...")
    
    env = DummyVecEnv([lambda: TradingEnv(test_df)])
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    env.training = False
    env.obs_rms = obs_rms
    
    obs_list = []
    obs = env.reset()
    for _ in range(200):
        obs_list.append(obs[0])
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _ = env.step(action)
        if done[0]: break
    
    obs_tensor = torch.FloatTensor(np.array(obs_list)).to(model.device)
    obs_tensor.requires_grad = True
    
    if hasattr(model.policy, 'actor'): # SAC, TD3
        actions = model.policy.actor(obs_tensor)
    else: # PPO
        dist = model.policy.get_distribution(obs_tensor)
        actions = dist.mode() 
        
    actions.sum().backward()
    
    importance = obs_tensor.grad.abs().mean(dim=0).cpu().numpy()
    
    env.close()
    return importance / importance.sum()

FEATURE_NAMES = ['RSI', 'MACD', 'Volatility', 'Ratio', 'Body', 'Weight', 'PnL', 'Price/1000']

def visualize_comparison(results):
    
    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    models = ['TD3', 'SAC', 'PPO']
    colors_model = {'TD3': '#e74c3c', 'SAC': '#27ae60', 'PPO': '#3498db'}
    
    # Feature colors (consistent across models)
    feature_colors = plt.cm.Set3(np.linspace(0, 1, len(FEATURE_NAMES)))
    
    # Row 1: Bar charts (horizontal)
    for i, model_name in enumerate(models):
        ax = fig.add_subplot(gs[0, i])
        
        importance = results[model_name]['importance']
        sorted_idx = np.argsort(importance)
        
        bars = ax.barh(range(len(FEATURE_NAMES)), importance[sorted_idx], 
                    color=[feature_colors[idx] for idx in sorted_idx],
                    edgecolor='black', alpha=0.8, linewidth=1.5)
        
        ax.set_yticks(range(len(FEATURE_NAMES)))
        ax.set_yticklabels([FEATURE_NAMES[idx] for idx in sorted_idx], fontsize=10, fontweight='bold')
        ax.set_xlabel('Importance (%)', fontsize=11, fontweight='bold')
        ax.set_title(f'{model_name} Feature Importance\n(Return: {results[model_name]["return"]:.2f}%)', 
                    fontsize=13, fontweight='bold')
        ax.set_xlim(0, max(importance) * 1.15)
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add percentage labels
        for j, (idx, val) in enumerate(zip(sorted_idx, importance[sorted_idx])):
            ax.text(val + max(importance)*0.02, j, f'{val*100:.1f}%',
                va='center', fontsize=9, fontweight='bold')
    
    for i, model_name in enumerate(models):
        ax = fig.add_subplot(gs[1, i])
        
        importance = results[model_name]['importance']
        
        # Normalize to sum to 1
        importance_norm = importance / importance.sum()
        
        # Only show features > 10%
        threshold = 0.10
        mask = importance_norm > threshold
        
        pie_values = importance_norm[mask].tolist()
        pie_labels = [FEATURE_NAMES[j] for j in range(len(FEATURE_NAMES)) if mask[j]]
        pie_colors = [feature_colors[j] for j in range(len(FEATURE_NAMES)) if mask[j]]
        
        if not mask.all():
            others = importance_norm[~mask].sum()
            pie_values.append(others)
            pie_labels.append('Others')
            pie_colors.append('lightgray')
        
        sorted_indices = np.argsort(pie_values)[::-1]
        pie_values = [pie_values[idx] for idx in sorted_indices]
        pie_labels = [pie_labels[idx] for idx in sorted_indices]
        pie_colors = [pie_colors[idx] for idx in sorted_indices]
        
        wedges, texts, autotexts = ax.pie(
            pie_values, 
            labels=pie_labels, 
            colors=pie_colors,
            autopct='%1.1f%%',
            startangle=90,
            textprops={'fontsize': 10, 'fontweight': 'bold'},
            counterclock=False
        )
        
        ax.set_title(f'{model_name} Distribution', fontsize=13, fontweight='bold')
    
    plt.suptitle('Feature Importance Analysis (Normalized to 100%)', 
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.savefig('results/feature_analysis.png', dpi=300, bbox_inches='tight')
    print("\nSaved: results/feature_analysis.png")
    
    print("Feature Importance Summary (Normalized)")
    
    for model_name in models:
        importance = results[model_name]['importance']
        total = importance.sum()
        
        print(f"\n{model_name} (Total: {total:.4f}):")
        sorted_idx = np.argsort(importance)[::-1]
        
        for rank, idx in enumerate(sorted_idx, 1):
            print(f"  {rank}. {FEATURE_NAMES[idx]:15s}: {importance[idx]*100:5.2f}%")

if __name__ == '__main__':
    test_df = load_data()
    
    try:
        csv_df = pd.read_csv('results/multi_seed_results.csv')
        mean_returns = csv_df[['PPO', 'SAC', 'TD3']].mean()
        print("Mean Returns loaded from CSV:")
        print(mean_returns)
    except:
        print("CSV not found. Using default returns.")
        mean_returns = {'PPO': 0.0, 'SAC': 0.0, 'TD3': 0.0}

    results = {}
    for name in ['TD3', 'SAC', 'PPO']:
        model, obs_rms = load_competition_model(name)
        if model:
            imp = calculate_feature_importance(model, test_df, obs_rms)
            results[name] = {
                'importance': imp,
                'return': mean_returns[name]
            }
            
    if len(results) == 3:
        visualize_comparison(results)