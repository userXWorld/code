# HIV模型评估框架使用指南

## 概述

`hiv_model_evaluation.py` 提供了一个全面的评估框架，用于验证和评估 `hiv_treatment_optimization.py` 中的HIV治疗优化系统。

## 主要功能

### 1. HIVModelEvaluator 类

评估器类提供以下方法：

- **evaluate_validity()**: 有效性评估
  - 病毒抑制率（目标 >85%）
  - CD4安全率（目标 >90%）
  - 治疗稳定性（动作切换频率）

- **evaluate_no_treatment_baseline()**: 无治疗基线对比
  - 量化治疗的净收益

- **evaluate_robustness()**: 鲁棒性分析
  - 测试在耐药性和测量误差下的性能

- **diagnose_improvements()**: 改进诊断
  - 基于评估结果提供具体改进建议

### 2. PerturbedHIVEnv 类

扰动环境类用于鲁棒性测试：

- `viral_replication_factor`: 病毒复制因子（模拟耐药性）
- `observation_noise`: 观察噪声（模拟测量误差）

## 使用示例

### 独立运行

```bash
python hiv_model_evaluation.py
```

这将：
1. 训练一个演示智能体（100轮）
2. 执行完整的评估套件
3. 生成中文报告和可视化图表

### 作为模块使用

```python
from hiv_model_evaluation import HIVModelEvaluator, PerturbedHIVEnv
from hiv_treatment_optimization import HIVTreatmentEnv, HIVTreatmentAgent

# 创建环境和智能体
env = HIVTreatmentEnv()
agent = HIVTreatmentAgent(env.state_dim, env.action_dim)

# 训练智能体（可选，如果已有训练好的模型）
# train_agent(env, agent, num_episodes=500)

# 创建评估器
evaluator = HIVModelEvaluator(env, agent)

# 执行评估
evaluator.evaluate_validity(num_episodes=100)
evaluator.evaluate_no_treatment_baseline(num_episodes=100)
evaluator.evaluate_robustness(num_episodes=50)

# 诊断改进
suggestions = evaluator.diagnose_improvements()

# 生成报告
evaluator.generate_comprehensive_report()

# 可视化鲁棒性
evaluator.plot_robustness_analysis()
```

### 自定义扰动测试

```python
from hiv_model_evaluation import PerturbedHIVEnv

# 创建高耐药性环境
high_resistance_env = PerturbedHIVEnv(
    viral_replication_factor=2.5,  # 2.5倍病毒复制
    observation_noise=0.2          # 20%测量误差
)

# 在扰动环境中测试智能体
state = high_resistance_env.reset()
for _ in range(96):
    action = agent.select_action(state, training=False)
    state, reward, done, info = high_resistance_env.step(action)
    if done:
        break
```

## 输出

### 生成的文件

- `robustness_analysis.png`: 鲁棒性分析可视化图表

### 报告内容

1. **有效性评估结果**：各项指标的达标情况
2. **净收益分析**：相比无治疗的改进
3. **鲁棒性评估结果**：不同扰动场景下的性能
4. **改进建议**：基于诊断的具体建议

## 改进建议类型

根据评估结果，系统可能建议：

- **病毒抑制率低** → Double DQN, Prioritized Replay
- **治疗稳定性差** → Action Smoothing Reward, RNN/LSTM
- **鲁棒性差** → Domain Randomization
- **CD4安全率低** → Reward Reweighting, Constrained RL

## 依赖项

- numpy
- pandas
- matplotlib
- torch
- tqdm

## 注意事项

1. 评估需要一定计算时间，可根据需要调整 `num_episodes` 参数
2. 独立运行时使用100轮训练作为演示，实际应用建议使用500+轮
3. 所有输出使用中文，匹配原始代码风格
