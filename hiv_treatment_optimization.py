"""
HIV长期治疗策略优化系统
基于深度强化学习(DQN)的个性化治疗方案生成
"""
import sys
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 设置标准输出为UTF-8编码
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 第一部分：环境模拟器 ====================

class HIVTreatmentEnv:
    """
    HIV治疗环境模拟器
    模拟患者免疫系统在不同治疗方案下的动态演变
    """
    def __init__(self, patient_id=0):
        self.patient_id = patient_id
        # 状态空间：[病毒载量(log), CD4计数, CD8计数, 免疫1, 免疫2, 效应器]
        self.state_dim = 6
        # 动作空间：4种药物组合 [无治疗, 单药, 双药, 三药联合]
        self.action_dim = 4
        
        # 临床参数（来自文献的典型值）
        self.viral_threshold = 500  # 病毒抑制目标
        self.cd4_danger = 200       # CD4危险阈值
        self.cd4_healthy = 500      # CD4健康水平
        
        self.reset()
    
    def reset(self):
        """
        重置环境到初始状态
        为何这么写：初始状态模拟未治疗的HIV感染患者典型指标
        """
        # 初始病毒载量：10^4 - 10^5 copies/mL（未治疗患者典型值）
        initial_viral = np.random.uniform(4.0, 5.0)  # log10尺度
        # 初始CD4：200-500 cells/μL（感染但未严重免疫抑制）
        initial_cd4 = np.random.uniform(200, 500)
        # 其他免疫指标随机初始化
        self.state = np.array([
            initial_viral,
            initial_cd4,
            np.random.uniform(800, 1200),  # CD8计数
            np.random.uniform(0.1, 0.3),   # 免疫效应1
            np.random.uniform(0.1, 0.3),   # 免疫效应2
            np.random.uniform(0.05, 0.15)  # 效应器细胞
        ])
        self.week = 0
        return self.state.copy()
    
    def step(self, action):
        """
        执行治疗动作，返回新状态和奖励
        
        参数：
            action: 0=无治疗, 1=单药, 2=双药, 3=三药联合
        
        为何这么写：使用简化的HIV动力学模型（基于Perelson模型）
        真实Health Gym使用更复杂的微分方程，这里用近似更新规则
        """
        # 药物效力系数
        drug_efficacy = [0.0, 0.5, 0.75, 0.9][action]  # 越多药物，效力越强
        
        # 1. 病毒载量更新（指数衰减 + 复制）
        viral_decay = drug_efficacy * 0.5  # 治疗导致的病毒衰减
        viral_replication = (1 - drug_efficacy) * 0.3  # 残余复制
        self.state[0] += -viral_decay + viral_replication + np.random.normal(0, 0.1)
        self.state[0] = np.clip(self.state[0], 1.0, 6.0)  # 限制在合理范围
        
        # 2. CD4计数更新（受病毒载量和治疗影响）
        viral_damage = -0.01 * (10 ** self.state[0]) / 10000  # 病毒杀伤CD4
        treatment_benefit = drug_efficacy * 5  # 治疗促进CD4恢复
        self.state[1] += treatment_benefit + viral_damage + np.random.normal(0, 10)
        self.state[1] = np.clip(self.state[1], 50, 1500)
        
        # 3. 其他免疫指标更新（简化处理）
        self.state[2:] += np.random.normal(0, 0.05, size=4)  # 随机波动
        self.state[2:] = np.clip(self.state[2:], 0, 2)
        
        self.week += 1
        
        # 计算奖励
        reward = self._compute_reward(action)
        
        # 判断是否结束（96周或CD4过低）
        done = (self.week >= 96) or (self.state[1] < 50)
        
        info = {
            'viral_load': 10 ** self.state[0],  # 转回真实尺度
            'cd4_count': self.state[1],
            'week': self.week
        }
        
        return self.state.copy(), reward, done, info
    
    def _compute_reward(self, action):
        """
        计算综合奖励
        为何这么写：多目标优化需要加权组合，权重基于临床优先级
        """
        # 1. 病毒抑制奖励（最高优先级 40%）
        viral_log = self.state[0]
        if 10**viral_log < 50:  # 完全抑制
            viral_reward = 10.0
        elif 10**viral_log < self.viral_threshold:  # 达标
            viral_reward = 5.0
        elif 10**viral_log < 10000:  # 可接受
            viral_reward = 0.0
        else:  # 失败
            viral_reward = -5.0
        
        # 2. CD4维持奖励（次优先级 30%）
        cd4 = self.state[1]
        if cd4 >= self.cd4_healthy:
            cd4_reward = 10.0
        elif cd4 >= 350:
            cd4_reward = 5.0
        elif cd4 >= self.cd4_danger:
            cd4_reward = 0.0
        else:  # 危险区域
            cd4_reward = -10.0
        
        # 3. 稳定性奖励（20%）- 惩罚剧烈波动
        # 为何用getattr：首次调用时prev_state不存在，避免报错
        prev_state = getattr(self, 'prev_state', self.state)
        viral_change_rate = abs(self.state[0] - prev_state[0])
        cd4_change_rate = abs(self.state[1] - prev_state[1]) / (prev_state[1] + 1e-6)
        
        if viral_change_rate > 0.5 or cd4_change_rate > 0.3:
            stability_reward = -5.0  # 大幅波动
        elif viral_change_rate > 0.2 or cd4_change_rate > 0.15:
            stability_reward = -2.0  # 中度波动
        else:
            stability_reward = 2.0   # 稳定
        
        self.prev_state = self.state.copy()
        
        # 4. 治疗负担惩罚（10%）
        treatment_burden = -action * 0.5  # 越多药物，负担越大
        
        # 额外惩罚频繁切换
        prev_action = getattr(self, 'prev_action', action)
        if action != prev_action and self.week > 0:
            treatment_burden -= 2.0  # 切换成本
        self.prev_action = action
        
        # 加权组合（总和为1.0，确保可比性）
        total_reward = (
            0.4 * viral_reward +
            0.3 * cd4_reward +
            0.2 * stability_reward +
            0.1 * treatment_burden
        )
        
        return total_reward


# ==================== 第二部分：深度Q网络 ====================

class DQN(nn.Module):
    """
    Deep Q-Network
    为何用这个结构：
    - 两层隐藏层足够拟合HIV动力学
    - Dropout防止过拟合（数据量可能有限）
    - ReLU激活避免梯度消失
    """
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),  # 20%dropout率，平衡正则化和容量
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, x):
        return self.network(x)


# ==================== 第三部分：强化学习智能体 ====================

class HIVTreatmentAgent:
    """
    DQN智能体，学习最优治疗策略
    """
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # 设备选择（优先GPU）
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️  使用设备: {self.device}")
        
        # 双网络架构（策略网络 + 目标网络）
        # 为何用双网络：稳定训练，避免Q值估计震荡
        self.policy_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # 目标网络不参与训练
        
        # 优化器（Adam自适应学习率，适合非平稳问题）
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)
        
        # 经验回放缓冲区（打破数据相关性）
        # 为何用deque：自动丢弃旧数据，保持固定容量
        self.memory = deque(maxlen=10000)
        
        # 超参数
        self.batch_size = 64
        self.gamma = 0.99  # 折扣因子，接近1重视长期回报
        self.epsilon = 1.0  # 初始探索率
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.target_update_freq = 10  # 每10轮更新目标网络
    
    def select_action(self, state, training=True):
        """
        ε-贪心策略选择动作
        为何这么写：平衡探索(随机)和利用(最优)
        """
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)  # 探索
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                return q_values.argmax().item()  # 利用
    
    def store_transition(self, state, action, reward, next_state, done):
        """存储经验到回放缓冲区"""
        self.memory.append((state, action, reward, next_state, done))
    
    def train_step(self):
        """
        从经验回放中采样并训练
        为何批量训练：提高样本利用率，稳定梯度
        """
        if len(self.memory) < self.batch_size:
            return 0.0  # 样本不足，跳过
        
        # 随机采样batch（打破时间相关性）
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # 转换为张量（为何用stack：保持batch维度）
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # 计算当前Q值
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        # 计算目标Q值（Bellman方程）
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            # 为何乘(1-dones)：终止状态无未来回报
            target_q = rewards + self.gamma * next_q * (1 - dones)
        
        # MSE损失（为何用MSE：Q值回归问题）
        loss = nn.MSELoss()(current_q.squeeze(), target_q)
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪（为何需要：防止梯度爆炸）
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self):
        """将策略网络权重复制到目标网络"""
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def decay_epsilon(self):
        """衰减探索率"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


# ==================== 第四部分：训练流程 ====================

def train_agent(env, agent, num_episodes=500):
    """
    训练智能体
    返回：训练历史记录
    """
    history = {
        'episode_rewards': [],
        'viral_suppression_rates': [],
        'cd4_safety_rates': [],
        'losses': [],
        'epsilons': []
    }
    
    print("\n🚀 开始训练...")
    for episode in tqdm(range(num_episodes), desc="训练进度"):
        state = env.reset()
        episode_reward = 0
        episode_losses = []
        viral_suppressed = 0
        cd4_safe = 0
        
        for week in range(96):
            # 选择并执行动作
            action = agent.select_action(state, training=True)
            next_state, reward, done, info = env.step(action)
            
            # 存储经验
            agent.store_transition(state, action, reward, next_state, done)
            
            # 训练
            loss = agent.train_step()
            if loss > 0:
                episode_losses.append(loss)
            
            # 统计指标
            episode_reward += reward
            if info['viral_load'] < 500:
                viral_suppressed += 1
            if info['cd4_count'] >= 200:
                cd4_safe += 1
            
            state = next_state
            if done:
                break
        
        # 更新目标网络
        if episode % agent.target_update_freq == 0:
            agent.update_target_network()
        
        # 衰减探索率
        agent.decay_epsilon()
        
        # 记录历史
        history['episode_rewards'].append(episode_reward)
        history['viral_suppression_rates'].append(viral_suppressed / 96)
        history['cd4_safety_rates'].append(cd4_safe / 96)
        history['losses'].append(np.mean(episode_losses) if episode_losses else 0)
        history['epsilons'].append(agent.epsilon)
        
        # 每50轮打印进度
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(history['episode_rewards'][-50:])
            avg_viral_supp = np.mean(history['viral_suppression_rates'][-50:])
            print(f"\n📊 Episode {episode+1}: "
                  f"平均奖励={avg_reward:.2f}, "
                  f"病毒抑制率={avg_viral_supp*100:.1f}%, "
                  f"ε={agent.epsilon:.3f}")
    
    print("\n✅ 训练完成！")
    return history


# ==================== 第五部分：评估与对比 ====================

def evaluate_policy(env, agent, num_episodes=100):
    """
    评估训练好的策略（无探索）
    """
    results = {
        'episode_rewards': [],
        'viral_loads': [],
        'cd4_counts': [],
        'actions': [],
        'viral_suppression_rates': [],
        'cd4_safety_rates': [],
        'treatment_switches': []
    }
    
    print("\n🔍 评估策略性能...")
    for episode in tqdm(range(num_episodes), desc="评估进度"):
        state = env.reset()
        episode_data = {
            'rewards': [],
            'viral_loads': [],
            'cd4_counts': [],
            'actions': []
        }
        viral_suppressed = 0
        cd4_safe = 0
        switches = 0
        prev_action = None
        
        for week in range(96):
            action = agent.select_action(state, training=False)  # 无探索
            next_state, reward, done, info = env.step(action)
            
            episode_data['rewards'].append(reward)
            episode_data['viral_loads'].append(info['viral_load'])
            episode_data['cd4_counts'].append(info['cd4_count'])
            episode_data['actions'].append(action)
            
            if info['viral_load'] < 500:
                viral_suppressed += 1
            if info['cd4_count'] >= 200:
                cd4_safe += 1
            if prev_action is not None and action != prev_action:
                switches += 1
            
            prev_action = action
            state = next_state
            if done:
                break
        
        results['episode_rewards'].append(np.sum(episode_data['rewards']))
        results['viral_loads'].append(episode_data['viral_loads'])
        results['cd4_counts'].append(episode_data['cd4_counts'])
        results['actions'].append(episode_data['actions'])
        results['viral_suppression_rates'].append(viral_suppressed / 96)
        results['cd4_safety_rates'].append(cd4_safe / 96)
        results['treatment_switches'].append(switches)
    
    return results


def evaluate_baseline(env, strategy_name, num_episodes=100):
    """
    评估基线策略
    strategy_name: 'fixed' (固定方案) 或 'cycling' (循环方案)
    """
    results = {
        'episode_rewards': [],
        'viral_suppression_rates': [],
        'cd4_safety_rates': [],
        'treatment_switches': []
    }
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        viral_suppressed = 0
        cd4_safe = 0
        switches = 0
        prev_action = None
        
        for week in range(96):
            # 基线策略选择
            if strategy_name == 'fixed':
                action = 3  # 固定使用三药联合
            elif strategy_name == 'cycling':
                action = week % 4  # 每周循环切换
            
            next_state, reward, done, info = env.step(action)
            
            episode_reward += reward
            if info['viral_load'] < 500:
                viral_suppressed += 1
            if info['cd4_count'] >= 200:
                cd4_safe += 1
            if prev_action is not None and action != prev_action:
                switches += 1
            
            prev_action = action
            state = next_state
            if done:
                break
        
        results['episode_rewards'].append(episode_reward)
        results['viral_suppression_rates'].append(viral_suppressed / 96)
        results['cd4_safety_rates'].append(cd4_safe / 96)
        results['treatment_switches'].append(switches)
    
    return results


# ==================== 第六部分：可视化模块 ====================

def plot_training_history(history):
    """
    绘制训练历史曲线
    为何用4子图：全面展示训练动态（奖励、指标、损失、探索）
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. 奖励曲线（平滑处理）
    # 为何用移动平均：减少噪声，更清晰展示趋势
    window = 20
    smoothed_rewards = pd.Series(history['episode_rewards']).rolling(window).mean()
    axes[0, 0].plot(smoothed_rewards, 'b-', linewidth=2, label='平滑奖励')
    axes[0, 0].plot(history['episode_rewards'], 'b-', alpha=0.3, label='原始奖励')
    axes[0, 0].set_xlabel('训练轮次', fontsize=12)
    axes[0, 0].set_ylabel('累积奖励', fontsize=12)
    axes[0, 0].set_title('训练奖励曲线', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 病毒抑制率
    axes[0, 1].plot(history['viral_suppression_rates'], 'g-', linewidth=2)
    axes[0, 1].axhline(y=0.85, color='r', linestyle='--', label='优秀目标(85%)')
    axes[0, 1].set_xlabel('训练轮次', fontsize=12)
    axes[0, 1].set_ylabel('病毒抑制率', fontsize=12)
    axes[0, 1].set_title('病毒抑制率演变', fontsize=14, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim([0, 1])
    
    # 3. 训练损失
    axes[1, 0].plot(history['losses'], 'orange', linewidth=2)
    axes[1, 0].set_xlabel('训练轮次', fontsize=12)
    axes[1, 0].set_ylabel('平均损失', fontsize=12)
    axes[1, 0].set_title('训练损失曲线', fontsize=14, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_yscale('log')  # 对数刻度更清晰
    
    # 4. 探索率衰减
    axes[1, 1].plot(history['epsilons'], 'purple', linewidth=2)
    axes[1, 1].set_xlabel('训练轮次', fontsize=12)
    axes[1, 1].set_ylabel('探索率 ε', fontsize=12)
    axes[1, 1].set_title('探索率衰减', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    print("📈 训练历史图已保存: training_history.png")
    plt.show()


def plot_treatment_trajectory(results, episode_idx=0):
    """
    可视化单个患者的治疗轨迹
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    weeks = range(len(results['viral_loads'][episode_idx]))
    viral_loads = results['viral_loads'][episode_idx]
    cd4_counts = results['cd4_counts'][episode_idx]
    actions = results['actions'][episode_idx]
    
    # 1. 病毒载量演变
    axes[0, 0].plot(weeks, viral_loads, 'b-', linewidth=2, marker='o', markersize=3)
    axes[0, 0].axhline(y=500, color='r', linestyle='--', linewidth=2, label='抑制目标')
    axes[0, 0].axhline(y=50, color='g', linestyle='--', linewidth=2, label='完全抑制')
    axes[0, 0].set_xlabel('周数', fontsize=12)
    axes[0, 0].set_ylabel('病毒载量 (copies/mL)', fontsize=12)
    axes[0, 0].set_title('病毒载量演变', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_yscale('log')  # 对数刻度（病毒载量跨度大）
    
    # 2. CD4计数演变
    axes[0, 1].plot(weeks, cd4_counts, 'g-', linewidth=2, marker='s', markersize=3)
    axes[0, 1].axhline(y=500, color='g', linestyle='--', linewidth=2, label='健康水平')
    axes[0, 1].axhline(y=200, color='r', linestyle='--', linewidth=2, label='危险阈值')
    axes[0, 1].set_xlabel('周数', fontsize=12)
    axes[0, 1].set_ylabel('CD4计数 (cells/μL)', fontsize=12)
    axes[0, 1].set_title('CD4细胞演变', fontsize=14, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 治疗方案时间线
    action_names = ['无治疗', '单药', '双药', '三药联合']
    # 为何用步阶图：清晰显示离散治疗方案切换
    axes[1, 0].step(weeks, actions, 'purple', linewidth=2, where='post')
    axes[1, 0].set_xlabel('周数', fontsize=12)
    axes[1, 0].set_ylabel('治疗方案', fontsize=12)
    axes[1, 0].set_title('治疗方案时间线', fontsize=14, fontweight='bold')
    axes[1, 0].set_yticks(range(4))
    axes[1, 0].set_yticklabels(action_names)
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 双指标散点图（状态空间轨迹）
    # 为何用渐变色：显示时间维度
    scatter = axes[1, 1].scatter(viral_loads, cd4_counts, c=weeks, 
                                  cmap='viridis', s=50, alpha=0.6)
    axes[1, 1].axvline(x=500, color='r', linestyle='--', alpha=0.5)
    axes[1, 1].axhline(y=200, color='r', linestyle='--', alpha=0.5)
    axes[1, 1].set_xlabel('病毒载量 (log scale)', fontsize=12)
    axes[1, 1].set_ylabel('CD4计数', fontsize=12)
    axes[1, 1].set_title('状态空间轨迹', fontsize=14, fontweight='bold')
    axes[1, 1].set_xscale('log')
    axes[1, 1].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes[1, 1], label='周数')
    
    plt.tight_layout()
    plt.savefig('treatment_trajectory.png', dpi=300, bbox_inches='tight')
    print("📊 治疗轨迹图已保存: treatment_trajectory.png")
    plt.show()


def plot_comparison(our_results, baseline_fixed, baseline_cycling):
    """
    对比我们的策略与基线方法
    为何用柱状图+误差线：直观比较多个指标的均值和方差
    """
    metrics = ['平均奖励', '病毒抑制率(%)', 'CD4安全率(%)', '治疗切换次数']
    
    our_scores = [
        np.mean(our_results['episode_rewards']),
        np.mean(our_results['viral_suppression_rates']) * 100,
        np.mean(our_results['cd4_safety_rates']) * 100,
        np.mean(our_results['treatment_switches'])
    ]
    
    fixed_scores = [
        np.mean(baseline_fixed['episode_rewards']),
        np.mean(baseline_fixed['viral_suppression_rates']) * 100,
        np.mean(baseline_fixed['cd4_safety_rates']) * 100,
        np.mean(baseline_fixed['treatment_switches'])
    ]
    
    cycling_scores = [
        np.mean(baseline_cycling['episode_rewards']),
        np.mean(baseline_cycling['viral_suppression_rates']) * 100,
        np.mean(baseline_cycling['cd4_safety_rates']) * 100,
        np.mean(baseline_cycling['treatment_switches'])
    ]
    
    # 计算标准误差（为何用标准误：评估结果的可靠性）
    our_std = [
        np.std(our_results['episode_rewards']) / np.sqrt(len(our_results['episode_rewards'])),
        np.std(our_results['viral_suppression_rates']) * 100 / np.sqrt(len(our_results['viral_suppression_rates'])),
        np.std(our_results['cd4_safety_rates']) * 100 / np.sqrt(len(our_results['cd4_safety_rates'])),
        np.std(our_results['treatment_switches']) / np.sqrt(len(our_results['treatment_switches']))
    ]
    
    # 绘制分组柱状图
    x = np.arange(len(metrics))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # 为何用不同颜色：区分不同策略
    bars1 = ax.bar(x - width, our_scores, width, label='我们的策略(DQN)', 
                   color='#2ecc71', alpha=0.8, yerr=our_std, capsize=5)
    bars2 = ax.bar(x, fixed_scores, width, label='固定三药联合', 
                   color='#3498db', alpha=0.8)
    bars3 = ax.bar(x + width, cycling_scores, width, label='循环方案', 
                   color='#e74c3c', alpha=0.8)
    
    ax.set_xlabel('评估指标', fontsize=13, fontweight='bold')
    ax.set_ylabel('得分', fontsize=13, fontweight='bold')
    ax.set_title('治疗策略性能对比', fontsize=15, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 在柱子上标注数值
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}',
                    ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('strategy_comparison.png', dpi=300, bbox_inches='tight')
    print("📊 策略对比图已保存: strategy_comparison.png")
    plt.show()
    
    # 打印详细统计
    print("\n" + "="*60)
    print("📋 详细性能对比")
    print("="*60)
    for i, metric in enumerate(metrics):
        print(f"\n{metric}:")
        print(f"  我们的策略: {our_scores[i]:.2f} ± {our_std[i]:.2f}")
        print(f"  固定方案:   {fixed_scores[i]:.2f}")
        print(f"  循环方案:   {cycling_scores[i]:.2f}")
        
        # 计算相对提升（为何用百分比：更直观）
        if fixed_scores[i] != 0:
            improvement = (our_scores[i] - fixed_scores[i]) / abs(fixed_scores[i]) * 100
            print(f"  相对固定方案提升: {improvement:+.1f}%")


def plot_action_distribution(results):
    """
    分析治疗方案选择分布
    为何需要：了解策略的治疗偏好
    """
    # 统计所有episode的动作分布
    all_actions = []
    for actions in results['actions']:
        all_actions.extend(actions)
    
    action_names = ['无治疗', '单药', '双药', '三药联合']
    action_counts = [all_actions.count(i) for i in range(4)]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. 饼图
    colors = ['#95a5a6', '#f39c12', '#3498db', '#2ecc71']
    ax1.pie(action_counts, labels=action_names, autopct='%1.1f%%',
            colors=colors, startangle=90, textprops={'fontsize': 11})
    ax1.set_title('治疗方案分布', fontsize=14, fontweight='bold')
    
    # 2. 柱状图
    ax2.bar(action_names, action_counts, color=colors, alpha=0.8)
    ax2.set_xlabel('治疗方案', fontsize=12)
    ax2.set_ylabel('使用次数', fontsize=12)
    ax2.set_title('治疗方案使用频次', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 在柱子上标注数值
    for i, (name, count) in enumerate(zip(action_names, action_counts)):
        ax2.text(i, count, str(count), ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('action_distribution.png', dpi=300, bbox_inches='tight')
    print("📊 动作分布图已保存: action_distribution.png")
    plt.show()


def generate_report(our_results, baseline_fixed, baseline_cycling):
    """
    生成综合评估报告
    为何需要：提供可解释的量化结果
    """
    print("\n" + "="*70)
    print("📄 HIV治疗策略优化 - 综合评估报告")
    print("="*70)
    
    print("\n【1. 整体性能指标】")
    print("-" * 70)
    
    metrics_dict = {
        '平均累积奖励': [
            np.mean(our_results['episode_rewards']),
            np.mean(baseline_fixed['episode_rewards']),
            np.mean(baseline_cycling['episode_rewards'])
        ],
        '病毒抑制率(%)': [
            np.mean(our_results['viral_suppression_rates']) * 100,
            np.mean(baseline_fixed['viral_suppression_rates']) * 100,
            np.mean(baseline_cycling['viral_suppression_rates']) * 100
        ],
        'CD4安全维持率(%)': [
            np.mean(our_results['cd4_safety_rates']) * 100,
            np.mean(baseline_fixed['cd4_safety_rates']) * 100,
            np.mean(baseline_cycling['cd4_safety_rates']) * 100
        ],
        '平均治疗切换次数': [
            np.mean(our_results['treatment_switches']),
            np.mean(baseline_fixed['treatment_switches']),
            np.mean(baseline_cycling['treatment_switches'])
        ]
    }
    
    df = pd.DataFrame(metrics_dict, index=['DQN策略', '固定方案', '循环方案']).T
    print(df.to_string())
    
    print("\n【2. 临床意义解读】")
    print("-" * 70)
    
    viral_supp = np.mean(our_results['viral_suppression_rates']) * 100
    cd4_safe = np.mean(our_results['cd4_safety_rates']) * 100
    switches = np.mean(our_results['treatment_switches'])
    
    if viral_supp >= 85:
        print(f"✅ 病毒抑制率 {viral_supp:.1f}% 达到优秀标准(≥85%)")
    elif viral_supp >= 70:
        print(f"⚠️  病毒抑制率 {viral_supp:.1f}% 达到合格标准(70-85%)")
    else:
        print(f"❌ 病毒抑制率 {viral_supp:.1f}% 未达标(<70%)")
    
    if cd4_safe >= 90:
        print(f"✅ CD4安全率 {cd4_safe:.1f}% 表明免疫功能维持良好")
    elif cd4_safe >= 75:
        print(f"⚠️  CD4安全率 {cd4_safe:.1f}% 需关注免疫波动")
    else:
        print(f"❌ CD4安全率 {cd4_safe:.1f}% 存在免疫风险")
    
    if switches < 10:
        print(f"✅ 平均切换 {switches:.1f} 次，治疗方案稳定性好")
    elif switches < 20:
        print(f"⚠️  平均切换 {switches:.1f} 次，适度调整")
    else:
        print(f"❌ 平均切换 {switches:.1f} 次，可能增加依从性负担")
    
    print("\n【3. 相对改进分析】")
    print("-" * 70)
    
    for metric, values in metrics_dict.items():
        dqn_val = values[0]
        fixed_val = values[1]
        if fixed_val != 0:
            improvement = (dqn_val - fixed_val) / abs(fixed_val) * 100
            direction = "↑" if improvement > 0 else "↓"
            print(f"{metric}: {direction} {abs(improvement):.1f}% (相比固定方案)")
    
    print("\n【4. 关键发现】")
    print("-" * 70)
    
    # 分析动作偏好
    all_actions = []
    for actions in our_results['actions']:
        all_actions.extend(actions)
    action_dist = [all_actions.count(i)/len(all_actions)*100 for i in range(4)]
    action_names = ['无治疗', '单药', '双药', '三药联合']
    dominant_action = action_names[np.argmax(action_dist)]
    
    print(f"• 策略最常使用: {dominant_action} ({max(action_dist):.1f}%)")
    print(f"• 治疗负担均衡性: {'良好' if action_dist[3] < 60 else '偏高'}")
    
    # 稳定性分析
    reward_std = np.std(our_results['episode_rewards'])
    print(f"• 性能稳定性: 标准差={reward_std:.2f} ({'稳定' if reward_std < 50 else '波动较大'})")
    
    print("\n【5. 临床应用建议】")
    print("-" * 70)
    print("• 建议在病毒载量>10,000时使用三药联合快速抑制")
    print("• CD4>500且病毒<50时可考虑简化为双药方案")
    print("• 每4-8周评估一次，根据指标动态调整")
    print("• 切换方案时需评估患者依从性和耐受性")
    
    print("\n" + "="*70)
    print("报告生成完成！")
    print("="*70 + "\n")


# ==================== 第七部分：主函数 ====================

def main():
    """
    主执行流程
    为何分步骤：模块化设计，便于调试和扩展
    """
    print("="*70)
    print("🏥 HIV长期治疗策略优化系统")
    print("基于深度强化学习(DQN)的个性化治疗方案生成")
    print("="*70)
    
    # 设置随机种子（为何需要：保证结果可复现）
    np.random.seed(42)
    torch.manual_seed(42)
    random.seed(42)
    
    # 第1步：初始化环境和智能体
    print("\n【步骤1】初始化环境和智能体...")
    env = HIVTreatmentEnv()
    agent = HIVTreatmentAgent(
        state_dim=env.state_dim,
        action_dim=env.action_dim
    )
    print(f"✅ 环境状态维度: {env.state_dim}")
    print(f"✅ 动作空间大小: {env.action_dim}")
    
    # 第2步：训练智能体
    print("\n【步骤2】训练智能体...")
    num_train_episodes = 500  # 可根据需要调整
    history = train_agent(env, agent, num_episodes=num_train_episodes)
    
    # 第3步：可视化训练过程
    print("\n【步骤3】可视化训练历史...")
    plot_training_history(history)
    
    # 第4步：评估训练好的策略
    print("\n【步骤4】评估训练好的策略...")
    num_eval_episodes = 100
    our_results = evaluate_policy(env, agent, num_episodes=num_eval_episodes)
    
    # 第5步：评估基线方法
    print("\n【步骤5】评估基线方法...")
    print("  - 评估固定方案...")
    baseline_fixed = evaluate_baseline(env, 'fixed', num_episodes=num_eval_episodes)
    print("  - 评估循环方案...")
    baseline_cycling = evaluate_baseline(env, 'cycling', num_episodes=num_eval_episodes)
    
    # 第6步：可视化对比结果
    print("\n【步骤6】生成对比可视化...")
    plot_comparison(our_results, baseline_fixed, baseline_cycling)
    
    # 第7步：可视化单个患者轨迹
    print("\n【步骤7】可视化典型患者治疗轨迹...")
    plot_treatment_trajectory(our_results, episode_idx=0)
    
    # 第8步：分析动作分布
    print("\n【步骤8】分析治疗方案选择分布...")
    plot_action_distribution(our_results)
    
    # 第9步：生成综合报告
    print("\n【步骤9】生成综合评估报告...")
    generate_report(our_results, baseline_fixed, baseline_cycling)
    
    # 第10步：保存模型（可选）
    print("\n【步骤10】保存训练好的模型...")
    torch.save({
        'policy_net_state_dict': agent.policy_net.state_dict(),
        'target_net_state_dict': agent.target_net.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
    }, 'hiv_treatment_model.pth')
    print("✅ 模型已保存到: hiv_treatment_model.pth")
    
    print("\n" + "="*70)
    print("🎉 所有任务完成！")
    print("="*70)
    
    # 返回结果供进一步分析
    return {
        'agent': agent,
        'env': env,
        'history': history,
        'our_results': our_results,
        'baseline_fixed': baseline_fixed,
        'baseline_cycling': baseline_cycling
    }


# ==================== 执行入口 ====================

if __name__ == "__main__":
    results = main()
    
    # 可选：交互式分析
    print("\n💡 提示：你可以通过以下方式进一步分析：")
    print("   - results['agent']: 访问训练好的智能体")
    print("   - results['env']: 访问环境")
    print("   - results['history']: 查看训练历史")
    print("   - results['our_results']: 查看评估结果")
