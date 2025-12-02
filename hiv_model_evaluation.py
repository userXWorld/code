"""
HIV治疗模型评估框架
对hiv_treatment_optimization.py系统进行全面验证和评估
"""
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 导入原始环境和智能体
try:
    from hiv_treatment_optimization import HIVTreatmentEnv, HIVTreatmentAgent, train_agent
except ImportError:
    print("❌ 无法导入hiv_treatment_optimization模块，请确保文件存在于同一目录")
    sys.exit(1)


# ==================== 第一部分：扰动环境 ====================

class PerturbedHIVEnv(HIVTreatmentEnv):
    """
    扰动的HIV治疗环境
    用于测试策略的鲁棒性
    继承自HIVTreatmentEnv，添加扰动参数
    """
    def __init__(self, patient_id=0, viral_replication_factor=1.0, observation_noise=0.0):
        """
        参数：
            patient_id: 患者ID
            viral_replication_factor: 病毒复制因子（模拟耐药性），>1表示更强复制能力
            observation_noise: 观察噪声水平（模拟测量误差）
        """
        super().__init__(patient_id)
        self.viral_replication_factor = viral_replication_factor
        self.observation_noise = observation_noise
    
    def step(self, action):
        """
        执行治疗动作，返回新状态和奖励（带扰动）
        """
        # 药物效力系数
        drug_efficacy = [0.0, 0.5, 0.75, 0.9][action]
        
        # 1. 病毒载量更新（添加复制因子扰动）
        viral_decay = drug_efficacy * 0.5
        # 应用病毒复制因子（模拟耐药性）
        viral_replication = (1 - drug_efficacy) * 0.3 * self.viral_replication_factor
        self.state[0] += -viral_decay + viral_replication + np.random.normal(0, 0.1)
        self.state[0] = np.clip(self.state[0], 1.0, 6.0)
        
        # 2. CD4计数更新（受病毒载量和治疗影响）
        viral_damage = -0.01 * (10 ** self.state[0]) / 10000
        treatment_benefit = drug_efficacy * 5
        self.state[1] += treatment_benefit + viral_damage + np.random.normal(0, 10)
        self.state[1] = np.clip(self.state[1], 50, 1500)
        
        # 3. 其他免疫指标更新
        self.state[2:] += np.random.normal(0, 0.05, size=4)
        self.state[2:] = np.clip(self.state[2:], 0, 2)
        
        self.week += 1
        
        # 计算奖励
        reward = self._compute_reward(action)
        
        # 判断是否结束
        done = (self.week >= 96) or (self.state[1] < 50)
        
        info = {
            'viral_load': 10 ** self.state[0],
            'cd4_count': self.state[1],
            'week': self.week
        }
        
        # 添加观察噪声（在返回状态之前）
        observed_state = self.state.copy()
        if self.observation_noise > 0:
            noise = np.random.normal(0, self.observation_noise, size=self.state.shape)
            observed_state += noise
            observed_state = np.clip(observed_state, [1.0, 50, 0, 0, 0, 0], [6.0, 1500, 2, 2, 2, 2])
        
        return observed_state, reward, done, info


# ==================== 第二部分：评估器类 ====================

class HIVModelEvaluator:
    """
    HIV模型评估器
    执行全面的有效性、鲁棒性和改进诊断
    """
    def __init__(self, env, agent):
        """
        参数：
            env: HIV治疗环境实例
            agent: 训练好的智能体实例
        """
        self.env = env
        self.agent = agent
        self.evaluation_results = {}
    
    def evaluate_validity(self, num_episodes=100):
        """
        有效性/效果检查
        测量：
        1. 病毒抑制率（目标>85%）
        2. CD4安全率（目标>90%）
        3. 治疗稳定性（动作切换频率）
        """
        print("\n" + "="*70)
        print("📊 【有效性评估】")
        print("="*70)
        
        viral_suppression_counts = []
        cd4_safety_counts = []
        treatment_switches = []
        episode_rewards = []
        
        for episode in range(num_episodes):
            state = self.env.reset()
            viral_suppressed = 0
            cd4_safe = 0
            switches = 0
            prev_action = None
            episode_reward = 0
            
            for week in range(96):
                action = self.agent.select_action(state, training=False)
                next_state, reward, done, info = self.env.step(action)
                
                episode_reward += reward
                
                # 统计病毒抑制（<500 copies/mL）
                if info['viral_load'] < 500:
                    viral_suppressed += 1
                
                # 统计CD4安全（>=200 cells/μL）
                if info['cd4_count'] >= 200:
                    cd4_safe += 1
                
                # 统计治疗切换
                if prev_action is not None and action != prev_action:
                    switches += 1
                
                prev_action = action
                state = next_state
                
                if done:
                    break
            
            viral_suppression_counts.append(viral_suppressed / 96)
            cd4_safety_counts.append(cd4_safe / 96)
            treatment_switches.append(switches)
            episode_rewards.append(episode_reward)
        
        # 计算统计指标
        viral_suppression_rate = np.mean(viral_suppression_counts) * 100
        cd4_safety_rate = np.mean(cd4_safety_counts) * 100
        avg_switches = np.mean(treatment_switches)
        avg_reward = np.mean(episode_rewards)
        
        # 保存结果
        self.evaluation_results['validity'] = {
            'viral_suppression_rate': viral_suppression_rate,
            'cd4_safety_rate': cd4_safety_rate,
            'avg_treatment_switches': avg_switches,
            'avg_reward': avg_reward,
            'std_reward': np.std(episode_rewards)
        }
        
        # 打印结果
        print(f"\n病毒抑制率: {viral_suppression_rate:.2f}% (目标: >85%)")
        if viral_suppression_rate >= 85:
            print("  ✅ 达到优秀标准")
        elif viral_suppression_rate >= 70:
            print("  ⚠️  达到合格标准")
        else:
            print("  ❌ 未达标")
        
        print(f"\nCD4安全率: {cd4_safety_rate:.2f}% (目标: >90%)")
        if cd4_safety_rate >= 90:
            print("  ✅ 达到优秀标准")
        elif cd4_safety_rate >= 75:
            print("  ⚠️  达到合格标准")
        else:
            print("  ❌ 未达标")
        
        print(f"\n平均治疗切换次数: {avg_switches:.2f}")
        if avg_switches < 10:
            print("  ✅ 治疗稳定性良好")
        elif avg_switches < 20:
            print("  ⚠️  治疗稳定性一般")
        else:
            print("  ❌ 治疗稳定性差，频繁切换")
        
        print(f"\n平均累积奖励: {avg_reward:.2f} ± {np.std(episode_rewards):.2f}")
        
        return self.evaluation_results['validity']
    
    def evaluate_no_treatment_baseline(self, num_episodes=100):
        """
        评估"无治疗"基线
        用于量化净收益
        """
        print("\n" + "="*70)
        print("📊 【无治疗基线评估】")
        print("="*70)
        
        viral_suppression_counts = []
        cd4_safety_counts = []
        episode_rewards = []
        
        for episode in range(num_episodes):
            state = self.env.reset()
            viral_suppressed = 0
            cd4_safe = 0
            episode_reward = 0
            
            for week in range(96):
                # 始终选择动作0（无治疗）
                action = 0
                next_state, reward, done, info = self.env.step(action)
                
                episode_reward += reward
                
                if info['viral_load'] < 500:
                    viral_suppressed += 1
                if info['cd4_count'] >= 200:
                    cd4_safe += 1
                
                state = next_state
                
                if done:
                    break
            
            viral_suppression_counts.append(viral_suppressed / 96)
            cd4_safety_counts.append(cd4_safe / 96)
            episode_rewards.append(episode_reward)
        
        # 计算统计指标
        baseline_viral_suppression = np.mean(viral_suppression_counts) * 100
        baseline_cd4_safety = np.mean(cd4_safety_counts) * 100
        baseline_reward = np.mean(episode_rewards)
        
        # 保存结果
        self.evaluation_results['no_treatment_baseline'] = {
            'viral_suppression_rate': baseline_viral_suppression,
            'cd4_safety_rate': baseline_cd4_safety,
            'avg_reward': baseline_reward
        }
        
        # 计算净收益
        if 'validity' in self.evaluation_results:
            validity = self.evaluation_results['validity']
            viral_benefit = validity['viral_suppression_rate'] - baseline_viral_suppression
            cd4_benefit = validity['cd4_safety_rate'] - baseline_cd4_safety
            reward_benefit = validity['avg_reward'] - baseline_reward
            
            print(f"\n无治疗基线指标:")
            print(f"  病毒抑制率: {baseline_viral_suppression:.2f}%")
            print(f"  CD4安全率: {baseline_cd4_safety:.2f}%")
            print(f"  平均奖励: {baseline_reward:.2f}")
            
            print(f"\n净收益（相比无治疗）:")
            print(f"  病毒抑制率提升: +{viral_benefit:.2f}%")
            print(f"  CD4安全率提升: +{cd4_benefit:.2f}%")
            print(f"  奖励提升: +{reward_benefit:.2f}")
            
            self.evaluation_results['net_benefit'] = {
                'viral_suppression_benefit': viral_benefit,
                'cd4_safety_benefit': cd4_benefit,
                'reward_benefit': reward_benefit
            }
        
        return self.evaluation_results['no_treatment_baseline']
    
    def evaluate_robustness(self, num_episodes=50):
        """
        鲁棒性分析
        测试在扰动条件下的性能下降
        """
        print("\n" + "="*70)
        print("📊 【鲁棒性评估】")
        print("="*70)
        
        # 测试不同的扰动水平
        perturbation_configs = [
            {'name': '正常条件', 'viral_factor': 1.0, 'noise': 0.0},
            {'name': '轻度耐药', 'viral_factor': 1.2, 'noise': 0.05},
            {'name': '中度耐药', 'viral_factor': 1.5, 'noise': 0.1},
            {'name': '重度耐药', 'viral_factor': 2.0, 'noise': 0.15},
            {'name': '高测量误差', 'viral_factor': 1.0, 'noise': 0.2}
        ]
        
        robustness_results = []
        
        for config in perturbation_configs:
            print(f"\n测试场景: {config['name']}")
            print(f"  病毒复制因子: {config['viral_factor']:.2f}")
            print(f"  观察噪声: {config['noise']:.2f}")
            
            # 创建扰动环境
            perturbed_env = PerturbedHIVEnv(
                patient_id=0,
                viral_replication_factor=config['viral_factor'],
                observation_noise=config['noise']
            )
            
            viral_suppression_counts = []
            cd4_safety_counts = []
            episode_rewards = []
            
            for episode in range(num_episodes):
                state = perturbed_env.reset()
                viral_suppressed = 0
                cd4_safe = 0
                episode_reward = 0
                
                for week in range(96):
                    action = self.agent.select_action(state, training=False)
                    next_state, reward, done, info = perturbed_env.step(action)
                    
                    episode_reward += reward
                    
                    if info['viral_load'] < 500:
                        viral_suppressed += 1
                    if info['cd4_count'] >= 200:
                        cd4_safe += 1
                    
                    state = next_state
                    
                    if done:
                        break
                
                viral_suppression_counts.append(viral_suppressed / 96)
                cd4_safety_counts.append(cd4_safe / 96)
                episode_rewards.append(episode_reward)
            
            result = {
                'scenario': config['name'],
                'viral_factor': config['viral_factor'],
                'noise': config['noise'],
                'viral_suppression_rate': np.mean(viral_suppression_counts) * 100,
                'cd4_safety_rate': np.mean(cd4_safety_counts) * 100,
                'avg_reward': np.mean(episode_rewards)
            }
            
            robustness_results.append(result)
            
            print(f"  结果: 病毒抑制率={result['viral_suppression_rate']:.2f}%, "
                  f"CD4安全率={result['cd4_safety_rate']:.2f}%, "
                  f"平均奖励={result['avg_reward']:.2f}")
        
        # 保存结果
        self.evaluation_results['robustness'] = robustness_results
        
        # 计算性能下降
        if 'validity' in self.evaluation_results:
            baseline_performance = self.evaluation_results['validity']
            print("\n性能下降分析（相比正常条件）:")
            
            for result in robustness_results[1:]:  # 跳过正常条件
                viral_drop = baseline_performance['viral_suppression_rate'] - result['viral_suppression_rate']
                cd4_drop = baseline_performance['cd4_safety_rate'] - result['cd4_safety_rate']
                reward_drop = baseline_performance['avg_reward'] - result['avg_reward']
                
                print(f"\n  {result['scenario']}:")
                print(f"    病毒抑制率下降: {viral_drop:.2f}%")
                print(f"    CD4安全率下降: {cd4_drop:.2f}%")
                print(f"    奖励下降: {reward_drop:.2f}")
        
        return robustness_results
    
    def diagnose_improvements(self):
        """
        改进诊断
        基于评估结果提供改进建议
        """
        print("\n" + "="*70)
        print("🔧 【改进诊断】")
        print("="*70)
        
        if 'validity' not in self.evaluation_results:
            print("❌ 请先运行有效性评估")
            return
        
        validity = self.evaluation_results['validity']
        suggestions = []
        
        # 1. 病毒抑制率诊断
        print("\n【病毒抑制率分析】")
        if validity['viral_suppression_rate'] < 85:
            print(f"⚠️  当前病毒抑制率 {validity['viral_suppression_rate']:.2f}% 低于目标")
            print("\n建议改进措施:")
            print("  1️⃣  使用 Double DQN")
            print("     - 减少Q值过高估计，提高策略质量")
            print("     - 实现方式: 用策略网络选择动作，用目标网络评估Q值")
            print("  2️⃣  使用 Prioritized Experience Replay")
            print("     - 优先回放高TD误差的经验，加速学习")
            print("     - 实现方式: 使用优先级队列替代均匀采样")
            suggestions.append("Double DQN")
            suggestions.append("Prioritized Replay")
        else:
            print(f"✅ 病毒抑制率 {validity['viral_suppression_rate']:.2f}% 达标")
        
        # 2. 治疗稳定性诊断
        print("\n【治疗稳定性分析】")
        if validity['avg_treatment_switches'] > 15:
            print(f"⚠️  平均切换次数 {validity['avg_treatment_switches']:.2f} 偏高")
            print("\n建议改进措施:")
            print("  1️⃣  添加 Action Smoothing Reward")
            print("     - 在奖励函数中增加连续性惩罚项")
            print("     - 实现方式: reward -= lambda * |action_t - action_{t-1}|")
            print("  2️⃣  使用 RNN/LSTM 策略网络")
            print("     - 捕捉时序依赖，产生更连贯的动作序列")
            print("     - 实现方式: 用LSTM替换当前的前馈网络")
            suggestions.append("Action Smoothing Reward")
            suggestions.append("RNN/LSTM Policy")
        else:
            print(f"✅ 治疗稳定性良好（平均切换 {validity['avg_treatment_switches']:.2f} 次）")
        
        # 3. 鲁棒性诊断
        if 'robustness' in self.evaluation_results:
            print("\n【鲁棒性分析】")
            robustness = self.evaluation_results['robustness']
            
            # 计算平均性能下降
            performance_drops = []
            for result in robustness[1:]:  # 跳过正常条件
                drop = validity['viral_suppression_rate'] - result['viral_suppression_rate']
                performance_drops.append(drop)
            
            avg_drop = np.mean(performance_drops)
            
            if avg_drop > 10:  # 平均下降超过10%
                print(f"⚠️  扰动条件下平均性能下降 {avg_drop:.2f}%")
                print("\n建议改进措施:")
                print("  1️⃣  训练时使用 Domain Randomization")
                print("     - 在训练期间随机化环境参数")
                print("     - 实现方式: 每个episode随机采样病毒复制因子和噪声水平")
                print("  2️⃣  使用 Robust MDP 框架")
                print("     - 优化最坏情况下的性能")
                print("     - 实现方式: 在Bellman更新中考虑不确定性")
                suggestions.append("Domain Randomization")
            else:
                print(f"✅ 鲁棒性良好（平均性能下降 {avg_drop:.2f}%）")
        
        # 4. CD4安全率诊断
        print("\n【CD4安全率分析】")
        if validity['cd4_safety_rate'] < 90:
            print(f"⚠️  CD4安全率 {validity['cd4_safety_rate']:.2f}% 低于目标")
            print("\n建议改进措施:")
            print("  1️⃣  调整奖励函数权重")
            print("     - 增加CD4维持奖励的权重（当前30% -> 35-40%）")
            print("  2️⃣  添加安全约束")
            print("     - 使用Constrained RL，硬约束CD4不低于阈值")
            suggestions.append("Reward Reweighting")
            suggestions.append("Constrained RL")
        else:
            print(f"✅ CD4安全率 {validity['cd4_safety_rate']:.2f}% 达标")
        
        # 保存建议
        self.evaluation_results['improvement_suggestions'] = suggestions
        
        print("\n" + "="*70)
        print(f"总结: 发现 {len(suggestions)} 个潜在改进方向")
        print("="*70)
        
        return suggestions
    
    def generate_comprehensive_report(self):
        """
        生成综合评估报告
        """
        print("\n" + "="*80)
        print("📄 HIV治疗模型 - 综合评估报告")
        print("="*80)
        
        if 'validity' in self.evaluation_results:
            print("\n【有效性评估结果】")
            print("-" * 80)
            validity = self.evaluation_results['validity']
            print(f"病毒抑制率:        {validity['viral_suppression_rate']:.2f}% (目标: >85%)")
            print(f"CD4安全率:         {validity['cd4_safety_rate']:.2f}% (目标: >90%)")
            print(f"平均治疗切换次数:  {validity['avg_treatment_switches']:.2f}")
            print(f"平均累积奖励:      {validity['avg_reward']:.2f} ± {validity['std_reward']:.2f}")
        
        if 'net_benefit' in self.evaluation_results:
            print("\n【净收益分析】")
            print("-" * 80)
            benefit = self.evaluation_results['net_benefit']
            print(f"病毒抑制率提升:    +{benefit['viral_suppression_benefit']:.2f}%")
            print(f"CD4安全率提升:     +{benefit['cd4_safety_benefit']:.2f}%")
            print(f"奖励提升:          +{benefit['reward_benefit']:.2f}")
        
        if 'robustness' in self.evaluation_results:
            print("\n【鲁棒性评估结果】")
            print("-" * 80)
            robustness = self.evaluation_results['robustness']
            
            # 创建表格
            data = []
            for result in robustness:
                data.append([
                    result['scenario'],
                    f"{result['viral_suppression_rate']:.2f}%",
                    f"{result['cd4_safety_rate']:.2f}%",
                    f"{result['avg_reward']:.2f}"
                ])
            
            df = pd.DataFrame(data, columns=['场景', '病毒抑制率', 'CD4安全率', '平均奖励'])
            print(df.to_string(index=False))
        
        if 'improvement_suggestions' in self.evaluation_results:
            print("\n【改进建议】")
            print("-" * 80)
            suggestions = self.evaluation_results['improvement_suggestions']
            if suggestions:
                for i, suggestion in enumerate(suggestions, 1):
                    print(f"{i}. {suggestion}")
            else:
                print("当前模型表现良好，无需特别改进")
        
        print("\n" + "="*80)
        print("报告生成完成")
        print("="*80)
    
    def plot_robustness_analysis(self):
        """
        可视化鲁棒性分析结果
        """
        if 'robustness' not in self.evaluation_results:
            print("❌ 请先运行鲁棒性评估")
            return
        
        robustness = self.evaluation_results['robustness']
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        scenarios = [r['scenario'] for r in robustness]
        viral_rates = [r['viral_suppression_rate'] for r in robustness]
        cd4_rates = [r['cd4_safety_rate'] for r in robustness]
        rewards = [r['avg_reward'] for r in robustness]
        
        # 1. 病毒抑制率
        bars1 = axes[0].bar(scenarios, viral_rates, color='steelblue', alpha=0.8)
        axes[0].axhline(y=85, color='red', linestyle='--', linewidth=2, label='目标(85%)')
        axes[0].set_ylabel('病毒抑制率 (%)', fontsize=12)
        axes[0].set_title('不同场景下的病毒抑制率', fontsize=13, fontweight='bold')
        axes[0].tick_params(axis='x', rotation=15)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # 标注数值
        for bar in bars1:
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
        
        # 2. CD4安全率
        bars2 = axes[1].bar(scenarios, cd4_rates, color='green', alpha=0.8)
        axes[1].axhline(y=90, color='red', linestyle='--', linewidth=2, label='目标(90%)')
        axes[1].set_ylabel('CD4安全率 (%)', fontsize=12)
        axes[1].set_title('不同场景下的CD4安全率', fontsize=13, fontweight='bold')
        axes[1].tick_params(axis='x', rotation=15)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3, axis='y')
        
        for bar in bars2:
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
        
        # 3. 平均奖励
        bars3 = axes[2].bar(scenarios, rewards, color='orange', alpha=0.8)
        axes[2].set_ylabel('平均奖励', fontsize=12)
        axes[2].set_title('不同场景下的平均奖励', fontsize=13, fontweight='bold')
        axes[2].tick_params(axis='x', rotation=15)
        axes[2].grid(True, alpha=0.3, axis='y')
        
        for bar in bars3:
            height = bar.get_height()
            axes[2].text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig('robustness_analysis.png', dpi=300, bbox_inches='tight')
        print("📊 鲁棒性分析图已保存: robustness_analysis.png")
        plt.show()


# ==================== 第三部分：主函数 ====================

def main():
    """
    主执行流程
    创建环境和智能体，训练并执行完整评估
    """
    print("="*80)
    print("🏥 HIV治疗模型评估框架")
    print("对hiv_treatment_optimization.py系统进行全面验证和评估")
    print("="*80)
    
    # 设置随机种子
    np.random.seed(42)
    torch.manual_seed(42)
    
    # 第1步：初始化环境和智能体
    print("\n【步骤1】初始化环境和智能体...")
    env = HIVTreatmentEnv()
    agent = HIVTreatmentAgent(
        state_dim=env.state_dim,
        action_dim=env.action_dim
    )
    print(f"✅ 环境状态维度: {env.state_dim}")
    print(f"✅ 动作空间大小: {env.action_dim}")
    
    # 第2步：简短训练（演示用）
    print("\n【步骤2】训练智能体（演示模式：100轮）...")
    num_train_episodes = 100  # 演示用，实际应该更多
    history = train_agent(env, agent, num_episodes=num_train_episodes)
    print(f"✅ 训练完成，最终探索率: {agent.epsilon:.3f}")
    
    # 第3步：创建评估器
    print("\n【步骤3】创建评估器...")
    evaluator = HIVModelEvaluator(env, agent)
    print("✅ 评估器已初始化")
    
    # 第4步：执行有效性评估
    print("\n【步骤4】执行有效性评估...")
    evaluator.evaluate_validity(num_episodes=50)
    
    # 第5步：评估无治疗基线
    print("\n【步骤5】评估无治疗基线...")
    evaluator.evaluate_no_treatment_baseline(num_episodes=50)
    
    # 第6步：执行鲁棒性分析
    print("\n【步骤6】执行鲁棒性分析...")
    evaluator.evaluate_robustness(num_episodes=30)
    
    # 第7步：可视化鲁棒性结果
    print("\n【步骤7】可视化鲁棒性分析...")
    evaluator.plot_robustness_analysis()
    
    # 第8步：改进诊断
    print("\n【步骤8】执行改进诊断...")
    evaluator.diagnose_improvements()
    
    # 第9步：生成综合报告
    print("\n【步骤9】生成综合评估报告...")
    evaluator.generate_comprehensive_report()
    
    print("\n" + "="*80)
    print("🎉 评估完成！")
    print("="*80)
    
    return evaluator


# ==================== 执行入口 ====================

if __name__ == "__main__":
    evaluator = main()
    
    # 可选：交互式分析
    print("\n💡 提示：你可以通过以下方式进一步分析：")
    print("   - evaluator.evaluation_results: 查看所有评估结果")
    print("   - evaluator.evaluate_validity(): 重新运行有效性评估")
    print("   - evaluator.evaluate_robustness(): 重新运行鲁棒性分析")
