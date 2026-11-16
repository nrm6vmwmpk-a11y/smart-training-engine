# 智训引擎项目 - 快速启动指南

## 项目结构

```
smart-training-engine/
├── README.md                          # 项目说明
├── TEAM_TASKS.md                      # 团队任务分配（已创建）
├── requirements.txt                   # Python依赖
├── Dockerfile                         # Docker镜像配置
│
├── step1_baseline.py                  # ✅ 步骤1：静态基线（已完成）
├── baseline_performance.png           # ✅ 生成的性能图
├── baseline_results.csv               # ✅ 结果数据
│
├── data/                              # 数据模块
│   ├── __init__.py
│   ├── gas_sensor_loader.py          # Gas Sensor数据加载器
│   ├── bearing_loader.py              # XJTU Bearing数据加载器
│   ├── cmapss_loader.py               # C-MAPSS数据加载器
│   ├── stream_interface.py            # 统一数据流接口
│   ├── quality_monitor.py             # 数据质量监控
│   └── augmentation.py                # 数据增强
│
├── models/                            # 模型模块
│   ├── __init__.py
│   ├── pwpae.py                       # PWPAE集成模型
│   ├── base_learners.py               # 基学习器（ARF、SRP等）
│   └── model_pool.py                  # 模型池管理
│
├── drift/                             # 漂移检测模块
│   ├── __init__.py
│   ├── detector.py                    # 漂移监测器
│   ├── diagnosis.py                   # 漂移诊断
│   └── reports.py                     # 诊断报告生成
│
├── strategy/                          # 策略控制模块
│   ├── __init__.py
│   ├── controller.py                  # 策略控制器
│   ├── levels.py                      # 五级响应策略
│   ├── resource_manager.py            # 资源管理
│   └── rsi_optimizer.py               # 策略层RSI
│
├── agent/                             # Agent决策模块
│   ├── __init__.py
│   ├── decision_agent.py              # Agent决策器
│   ├── prompts.py                     # Prompt模板
│   └── pytrio_client.py               # PyTRiO API客户端
│
├── experiment/                        # 实验管理模块
│   ├── __init__.py
│   ├── swanlab_logger.py              # SwanLab记录器
│   ├── shadow_evaluator.py            # 影子评估
│   ├── gate_keeper.py                 # 评估门禁
│   ├── version_manager.py             # 版本管理与回滚
│   └── resource_tracker.py            # 资源追踪
│
├── system/                            # 系统集成模块
│   ├── __init__.py
│   ├── orchestrator.py                # 系统编排器
│   ├── config.py                      # 配置管理
│   └── utils.py                       # 工具函数
│
├── main.py                            # 主程序入口
├── config.yaml                        # 系统配置文件
│
├── tests/                             # 测试代码
│   ├── test_data_loaders.py
│   ├── test_pwpae.py
│   └── test_drift_detection.py
│
└── docs/                              # 文档
    ├── architecture.md                # 架构文档
    ├── api_reference.md               # API文档
    └── experiment_log.md              # 实验日志
```

---

## 当前状态

### ✅ 已完成
- 项目目录创建
- 步骤1：静态基线模型
  - 代码：`step1_baseline.py`
  - 结果：`baseline_performance.png`, `baseline_results.csv`
  - 验证了数据漂移导致的性能下降

### 🎯 进行中
- 团队任务分配（`TEAM_TASKS.md`）

### 📋 待办
- 数据集下载与预处理
- 在线学习系统（步骤2）
- PWPAE集成
- Agent决策层
- SwanLab实验记录

---

## 立即开始（3个人的分工）

### 👤 成员A：技术负责人（你）
**现在就做**：
```bash
# 1. 安装核心依赖
pip install river scikit-learn pandas numpy matplotlib

# 2. 测试River安装
python3 -c "from river import ensemble, drift; print('River安装成功')"

# 3. 开始写步骤2代码
# 我会在下一步提供代码模板
```

**本周目标**：
- 完成在线学习基础框架
- 在模拟数据上看到效果

---

### 👤 成员B：数据工程师
**现在就做**：
```bash
# 1. 创建数据目录
mkdir -p ~/smart-training-engine/data/raw

# 2. 下载Gas Sensor数据集
# 方式1：浏览器下载
# 访问：https://archive.ics.uci.edu/dataset/224/gas
# 点击"Download"，保存到 data/raw/

# 方式2：命令行（如果可以访问）
cd ~/smart-training-engine/data/raw
wget https://archive.ics.uci.edu/static/public/224/data.zip

# 3. 查看数据格式
unzip data.zip
ls -lh
head batch1.dat  # 查看第一个批次的数据格式
```

**本周目标**：
- Gas Sensor数据集下载并解析
- 编写`data/gas_sensor_loader.py`
- 提供统一的数据迭代器接口

---

### 👤 成员C：实验工程师
**现在就做**：
```bash
# 1. 安装SwanLab
pip install swanlab

# 2. 注册账号
# 访问：https://swanlab.cn
# 注册并创建项目："smart-training-engine"

# 3. 测试SwanLab
python3 -c "
import swanlab

# 初始化项目
swanlab.init(project='smart-training-engine', experiment_name='test')

# 记录一条数据
swanlab.log({'test_metric': 0.95})

print('SwanLab测试成功')
"

# 4. 查看实验记录
# 在SwanLab网页端查看是否有记录
```

**本周目标**：
- SwanLab账号配置完成
- 设计实验记录规范（参考赛题要求）
- 将step1的结果记录到SwanLab

---

## 关键配置文件

### requirements.txt
```
# 核心依赖
river>=0.21.0              # 在线学习框架
swanlab>=0.3.0             # 实验记录
scikit-learn>=1.3.0        # 传统ML
pandas>=2.0.0              # 数据处理
numpy>=1.24.0              # 数值计算
matplotlib>=3.7.0          # 可视化
scipy>=1.11.0              # 科学计算

# 信号处理（XJTU数据集需要）
pywavelets>=1.4.0          # 小波变换

# API与工具
requests>=2.31.0           # HTTP请求
pyyaml>=6.0                # 配置文件
tqdm>=4.65.0               # 进度条

# 可选：大模型（如果用本地模型）
# transformers>=4.35.0
# torch>=2.0.0
```

---

## 数据集下载指南

### 1. Gas Sensor Array Drift Dataset
- **官方链接**：https://archive.ics.uci.edu/dataset/224/gas
- **大小**：约70MB
- **格式**：多个.dat文件，每个文件是一个batch
- **内容**：13,910个样本，128个特征（16个传感器×8个特征），6个类别
- **时间跨度**：36个月（2007.01-2011.02）

**下载步骤**：
1. 访问官网
2. 点击"Download"按钮
3. 保存到`data/raw/gas_sensor/`
4. 解压

### 2. XJTU-SY Bearing Dataset
- **GitHub链接**：https://github.com/WangBiaoXJTU/xjtu-sy-bearing-datasets
- **大小**：约4GB
- **格式**：CSV文件，每个文件是一个轴承的完整寿命数据
- **内容**：15个轴承，每个轴承包含水平和垂直方向的振动信号
- **采样频率**：25.6 kHz

**下载步骤**：
1. 访问GitHub页面
2. 点击README中的百度网盘链接
3. 下载并解压到`data/raw/xjtu_bearing/`

### 3. C-MAPSS Jet Engine Dataset
- **官方链接**：https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data
- **大小**：约50MB
- **格式**：txt文件，空格分隔
- **内容**：4个子数据集，每个包含训练集和测试集
- **特征数**：21个传感器读数 + 3个运行设置

**下载步骤**：
1. 访问NASA数据门户
2. 下载"CMAPSSData.zip"
3. 解压到`data/raw/cmapss/`

---

## 开发规范

### Git提交规范
```bash
# 功能开发
git commit -m "feat: 添加PWPAE集成模型"

# Bug修复
git commit -m "fix: 修复漂移检测器内存泄漏"

# 文档更新
git commit -m "docs: 更新API文档"

# 测试
git commit -m "test: 添加数据加载器单元测试"
```

### 代码注释规范
```python
def pwpae_ensemble(base_learners, window_size=100):
    """
    PWPAE动态加权集成预测
    
    Args:
        base_learners: dict, 基学习器字典，格式 {name: model}
        window_size: int, 计算错误率的窗口大小
        
    Returns:
        prediction: 集成预测结果
        
    Example:
        >>> learners = {'ARF-ADWIN': model1, 'SRP-DDM': model2}
        >>> pred = pwpae_ensemble(learners)
    """
    pass
```

---

## 调试技巧

### 1. 先用小数据集测试
```python
# 不要一开始就用全部13910个样本
# 先用前500个样本快速验证逻辑

df_small = df.head(500)
# 跑通流程后再用全量数据
```

### 2. 打印中间结果
```python
print(f"[DEBUG] 当前样本 {i}, 漂移检测: {drift_detected}, 错误率: {error_rate:.3f}")
```

### 3. 可视化每一步
```python
# 每个模块都应该有可视化输出
# 帮助理解系统行为

plt.plot(timestamps, accuracies)
plt.title("PWPAE Performance Over Time")
plt.show()
```

---

## 常见问题排查

### Q: River安装失败
```bash
# 确保Python版本>=3.8
python3 --version

# 升级pip
pip install --upgrade pip

# 重新安装
pip install river -v
```

### Q: 数据集下载太慢
```bash
# 使用代理（如果有）
export HTTP_PROXY=http://127.0.0.1:10809
export HTTPS_PROXY=http://127.0.0.1:10809

# 或者用迅雷/IDM等下载工具
```

### Q: SwanLab连接失败
```python
# 检查网络连接
import requests
response = requests.get('https://swanlab.cn')
print(response.status_code)  # 应该是200

# 如果公司网络有限制，考虑用手机热点
```

---

## 下一步计划

### 明天（技术负责人）
1. 我会提供`step2_online_learning.py`的代码模板
2. 你运行并测试
3. 对比step1和step2的性能差异

### 本周末
- 三个人完成各自的第一个任务
- 周日晚上开会汇报进度
- 确定下周的具体分工

### 下周
- 集成PWPAE
- 接入SwanLab
- 至少在Gas Sensor上跑通完整流程

---

**项目启动时间**：今天
**第一次评审**：本周日晚上
**技术负责人**：你
