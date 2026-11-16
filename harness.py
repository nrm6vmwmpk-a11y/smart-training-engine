"""
智训引擎 Harness - 主控程序
任务：让Harness自动优化官方baseline模型，支持任务A/B/C

作者：技术负责人
日期：2026-11-11
"""

# 标准库
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# ──────────────────────────────────────────────
# 可选依赖：SwanLab（没装也能跑）
# ──────────────────────────────────────────────
try:
    import swanlab
    _SWANLAB_AVAILABLE = True
except ImportError:
    _SWANLAB_AVAILABLE = False


# ──────────────────────────────────────────────
# 任务元信息
# ──────────────────────────────────────────────
TASK_META = {
    # task_id -> (is_regression, primary_metric, higher_is_better)
    'a': (False, 'macro_f1',  True),   # 分类，Macro-F1 越高越好
    'b': (True,  'rmse',      False),  # 回归，RMSE 越低越好
    'c': (True,  'rmse',      False),  # 回归，RMSE 越低越好
}


class SmartTrainingHarness:
    """
    智能训练Harness

    核心功能：
    1. 读取当前模型状态
    2. 分析性能并决策
    3. 修改超参数
    4. 执行续训
    5. 评估效果，回滚劣化轮次
    6. 从历史学习（策略层RSI）
    """

    def __init__(self, task='a', max_rounds=10):
        # 验证任务ID
        task = task.lower()
        if task not in TASK_META:
            raise ValueError(f"任务必须是 a/b/c，收到: {task}")

        self.task = task
        self.task_dir = Path(f'tasks/{task}')
        self.max_rounds = max_rounds
        self.history = []

        # 任务类型
        self.is_regression, self.metric_name, self.higher_is_better = TASK_META[task]

        # 追踪最佳轮次（用于回滚）
        # best_score 内部统一用"越高越好"表示（回归取负值）
        self.best_score = None
        self.best_round_id = 0  # 0 表示使用baseline checkpoint

        print("=" * 60)
        print("智训引擎 Harness 启动")
        print(f"任务: {task.upper()} | 指标: {self.metric_name} "
              f"({'越高越好' if self.higher_is_better else '越低越好'})")
        print(f"最大轮数: {max_rounds}")
        print("=" * 60)

    # ──────────────────────────────────────────
    # 主循环
    # ──────────────────────────────────────────
    def run(self):
        """主循环：执行多轮优化"""
        # 初始化SwanLab实验（可选）
        swanlab_run = None
        if _SWANLAB_AVAILABLE:
            try:
                swanlab_run = swanlab.init(
                    project="smart-training-engine",
                    experiment_name=f"task-{self.task.upper()}-{self.max_rounds}rounds",
                    config={
                        "task": self.task.upper(),
                        "metric": self.metric_name,
                        "max_rounds": self.max_rounds,
                        "harness_version": "v2",
                    }
                )
                print("[SwanLab] 实验已初始化")
            except Exception as e:
                print(f"[SwanLab] 初始化失败（跳过）: {e}")
                swanlab_run = None

        for round_id in range(1, self.max_rounds + 1):
            print(f"\n{'=' * 60}")
            print(f"第 {round_id}/{self.max_rounds} 轮优化")
            print(f"{'=' * 60}")

            try:
                # 1. 分析当前状态
                print("\n[步骤1] 分析当前模型状态...")
                analysis = self.analyze_current_state(round_id)

                # 2. 生成优化方案
                print("\n[步骤2] 生成优化方案...")
                plan = self.generate_optimization_plan(analysis, round_id)

                # 3. 执行优化
                print("\n[步骤3] 执行优化方案...")
                result = self.execute_plan(plan, round_id)

                # 4. 计算内部标准分（统一为越高越好）
                raw_score = result[f'{self.metric_name}_after']
                internal_score = raw_score if self.higher_is_better else -raw_score

                # 5. 判断是否需要回滚
                improved = (self.best_score is None) or (internal_score > self.best_score)
                if improved:
                    self.best_score = internal_score
                    self.best_round_id = round_id
                    result['rolled_back'] = False
                else:
                    result['rolled_back'] = True
                    print(f"\n  ⚠️  性能退化，最佳checkpoint保持第{self.best_round_id}轮")

                # 6. 记录历史
                self.history.append({
                    'round': round_id,
                    'analysis': analysis,
                    'plan': plan,
                    'result': result,
                    'timestamp': time.time(),
                    'best_round_so_far': self.best_round_id,
                })

                # 7. 写SwanLab
                if swanlab_run:
                    self._log_swanlab(round_id, result, plan)

                # 8. 显示结果
                self.print_round_summary(round_id, result)

                # 9. 策略层学习（每3轮一次）
                if round_id % 3 == 0:
                    print(f"\n[策略反思] 分析前{round_id}轮经验...")
                    self.learn_from_history()

            except Exception as e:
                print(f"\n❌ 第{round_id}轮执行失败: {e}")
                import traceback
                traceback.print_exc()
                continue

        # 最终总结
        self.print_final_summary()

        if swanlab_run:
            try:
                swanlab.finish()
                print("[SwanLab] 实验已结束，请前往 https://swanlab.cn 查看")
            except Exception:
                pass

    # ──────────────────────────────────────────
    # 状态分析
    # ──────────────────────────────────────────
    def analyze_current_state(self, round_id):
        """
        读取 training.json 提取当前模型关键指标

        Returns:
            dict: 分析结果（统一含 current_score 字段，越高越好）
        """
        if round_id == 1:
            training_file = self.task_dir / 'training.json'
        else:
            # 回滚：从最佳轮次读，而不是上一轮
            best = self.best_round_id
            if best == 0:
                training_file = self.task_dir / 'training.json'
            else:
                training_file = self.task_dir / f'round-{best}' / 'training.json'

        with open(training_file, encoding='utf-8') as f:
            info = json.load(f)

        val = info['validation_after']
        lr  = info['configuration']['learning_rate']
        epochs_so_far = info.get('final_epochs', info.get('configuration', {}).get('requested_epochs', 0))

        # 主指标
        raw_score = val[self.metric_name]
        internal_score = raw_score if self.higher_is_better else -raw_score

        analysis = {
            'current_score': internal_score,   # 越高越好（内部统一）
            'raw_score': raw_score,            # 原始值
            'learning_rate': lr,
            'epochs_so_far': epochs_so_far,
            'parameter_count': info.get('parameter_count', 0),
        }

        # 额外打印副指标
        if self.is_regression:
            analysis['mae'] = val.get('mae')
            print(f"  当前 RMSE: {val['rmse']:.4f}  MAE: {val.get('mae', 'N/A')}")
        else:
            analysis['accuracy'] = val.get('accuracy')
            print(f"  当前 Macro-F1: {val['macro_f1']:.4f}  Accuracy: {val.get('accuracy', 'N/A'):.4f}")
        print(f"  学习率: {lr}  累计epochs: {epochs_so_far}")

        return analysis

    # ──────────────────────────────────────────
    # 优化策略
    # ──────────────────────────────────────────
    def generate_optimization_plan(self, analysis, round_id):
        """
        生成超参数调整方案

        当前实现：基于规则的策略（后续可替换为LLM Agent）
        策略逻辑：前3轮多训、中段调学习率、后段保守微调

        Returns:
            dict: { strategy, epochs, learning_rate, expected_improvement }
        """
        score = analysis['current_score']   # 越高越好
        lr    = analysis['learning_rate']
        is_low_score  = score < (0.65 if not self.is_regression else -200)
        is_mid_score  = score < (0.78 if not self.is_regression else -100)

        if round_id <= 3:
            strategy, epochs = "增加训练轮数（观察趋势）", 5
            new_lr = lr
        elif is_low_score:
            strategy, epochs = "提高学习率（性能偏低）", 3
            new_lr = min(lr * 1.5, 0.01)
        elif is_mid_score:
            strategy, epochs = "微调学习率（接近目标）", 5
            new_lr = lr * 0.8
        else:
            strategy, epochs = "保持稳定（性能良好）", 2
            new_lr = lr * 0.9  # 轻微退火

        plan = {
            'strategy': strategy,
            'epochs': epochs,
            'learning_rate': new_lr,
            'expected_improvement': 0.01,
        }

        print(f"  策略: {plan['strategy']}")
        print(f"  epochs: {plan['epochs']}  学习率: {plan['learning_rate']:.6f}")
        return plan

    # ──────────────────────────────────────────
    # 执行训练
    # ──────────────────────────────────────────
    def execute_plan(self, plan, round_id):
        """
        调用官方 baselines/train.py 执行续训

        Args:
            plan: 优化计划
            round_id: 当前轮次

        Returns:
            dict: 训练前后指标 + 训练时长
        """
        # 清理旧输出目录
        output_dir = self.task_dir / f'round-{round_id}'
        if output_dir.exists():
            print(f"  清理旧目录: {output_dir}")
            shutil.rmtree(output_dir)

        # 确定起始checkpoint（支持回滚：使用最佳轮次）
        if self.best_round_id == 0:
            checkpoint = 'checkpoint.npz'
            flag = '--from-baseline'
        else:
            checkpoint = f'round-{self.best_round_id}/checkpoint.npz'
            flag = '--resume'

        # 构建基础命令
        cmd = [
            'python', '-m', 'baselines.train',
            '--profile', self.task.upper(),
            '--train',      'public/train.jsonl',
            '--validation', 'public/validation.jsonl',
            flag, checkpoint,
            '--epochs',        str(plan['epochs']),
            '--learning-rate', str(plan['learning_rate']),
            '--output',   f'round-{round_id}',
            '--stage-id', f'round-{round_id}',
        ]

        # 任务B额外需要 label_provenance
        if self.task == 'b':
            lp_file = self.task_dir / 'label_provenance.json'
            if not lp_file.exists():
                # 自动创建（draft状态）
                lp_data = {
                    "status": "draft",
                    "method": (
                        "RUL labels derived from official competition dataset. "
                        "Time-to-failure countdown from known end-of-life timestamps."
                    )
                }
                lp_file.write_text(json.dumps(lp_data, ensure_ascii=False, indent=2), encoding='utf-8')
                print(f"  已生成 label_provenance.json")
            cmd += ['--label-provenance', str(lp_file.name)]

        print(f"  执行: {' '.join(cmd)}")

        result_proc = subprocess.run(
            cmd,
            cwd=self.task_dir,
            capture_output=True,
            text=True,
            encoding='utf-8',
        )

        if result_proc.returncode != 0:
            raise RuntimeError(
                f"训练失败 (exit {result_proc.returncode}):\n{result_proc.stderr[-2000:]}"
            )

        # 读取结果
        result_file = self.task_dir / f'round-{round_id}' / 'training.json'
        with open(result_file, encoding='utf-8') as f:
            tr = json.load(f)

        before = tr['validation_before']
        after  = tr['validation_after']

        result = {
            f'{self.metric_name}_before': before[self.metric_name],
            f'{self.metric_name}_after':  after[self.metric_name],
            'training_time': tr['training_seconds'],
        }

        # 保留副指标
        if self.is_regression:
            result['mae_after'] = after.get('mae')
        else:
            result['accuracy_after'] = after.get('accuracy')

        return result

    # ──────────────────────────────────────────
    # SwanLab日志
    # ──────────────────────────────────────────
    def _log_swanlab(self, round_id, result, plan):
        """把单轮指标推送到SwanLab（失败不影响主流程）"""
        try:
            log_data = {
                'round': round_id,
                f'{self.metric_name}_before': result[f'{self.metric_name}_before'],
                f'{self.metric_name}_after':  result[f'{self.metric_name}_after'],
                'training_seconds': result['training_time'],
                'learning_rate': plan['learning_rate'],
                'epochs': plan['epochs'],
                'rolled_back': int(result.get('rolled_back', False)),
            }
            if 'accuracy_after' in result:
                log_data['accuracy'] = result['accuracy_after']
            if 'mae_after' in result:
                log_data['mae'] = result['mae_after']
            swanlab.log(log_data)
        except Exception as e:
            print(f"  [SwanLab] 日志写入失败（跳过）: {e}")

    # ──────────────────────────────────────────
    # 策略层学习
    # ──────────────────────────────────────────
    def learn_from_history(self):
        """每3轮分析历史ROI，输出经验总结（策略层RSI）"""
        if len(self.history) < 2:
            return

        print("\n  【策略学习】")
        for record in self.history:
            r = record['result']
            before = r[f'{self.metric_name}_before']
            after  = r[f'{self.metric_name}_after']
            # 统一用内部分（越高越好）计算提升
            delta = (after - before) if self.higher_is_better else (before - after)
            roi   = delta / (r['training_time'] + 0.001)
            rolled = "⚠️回滚" if r.get('rolled_back') else ""
            print(f"    第{record['round']}轮: {record['plan']['strategy']} | "
                  f"Δ{self.metric_name}={delta:+.4f} ROI={roi:.3f} {rolled}")

        # 最佳策略
        best = max(self.history, key=lambda x: (
            (x['result'][f'{self.metric_name}_after'] - x['result'][f'{self.metric_name}_before'])
            if self.higher_is_better
            else (x['result'][f'{self.metric_name}_before'] - x['result'][f'{self.metric_name}_after'])
        ))
        print(f"\n  最佳策略: 第{best['round']}轮 — {best['plan']['strategy']}")

    # ──────────────────────────────────────────
    # 打印函数
    # ──────────────────────────────────────────
    def print_round_summary(self, round_id, result):
        before = result[f'{self.metric_name}_before']
        after  = result[f'{self.metric_name}_after']
        delta  = (after - before) if self.higher_is_better else (before - after)

        print(f"\n{'=' * 60}")
        print(f"第 {round_id} 轮完成")
        print(f"{'=' * 60}")
        print(f"  {self.metric_name} 前: {before:.4f}")
        print(f"  {self.metric_name} 后: {after:.4f}")
        print(f"  变化:  {delta:+.4f}  {'[+提升]' if delta > 0 else '[-退化]'}")
        print(f"  耗时:  {result['training_time']:.2f}秒")
        if result.get('rolled_back'):
            print(f"  ** 已标记退化，下轮仍从第{self.best_round_id}轮checkpoint续训")

    def print_final_summary(self):
        print(f"\n{'=' * 60}")
        print(f"全部 {self.max_rounds} 轮优化完成")
        print(f"{'=' * 60}")

        if not self.history:
            print("没有完成的轮次")
            return

        initial = self.history[0]['result'][f'{self.metric_name}_before']
        final   = self.history[-1]['result'][f'{self.metric_name}_after']
        delta   = (final - initial) if self.higher_is_better else (initial - final)

        print(f"\n{self.metric_name} 变化:")
        print(f"  初始: {initial:.4f}")
        print(f"  最终: {final:.4f}")
        print(f"  净提升: {delta:+.4f} ({delta / (abs(initial) + 1e-9) * 100:+.1f}%)")
        print(f"\n最佳checkpoint: 第{self.best_round_id}轮 (round-{self.best_round_id}/checkpoint.npz)")

        # 保存历史
        history_file = self.task_dir / 'optimization_history.json'
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, indent=2, ensure_ascii=False, default=str)
        print(f"历史记录已保存: {history_file}")


# ──────────────────────────────────────────────
# 入口
# ──────────────────────────────────────────────
def main():
    # Windows GBK 终端不支持 emoji，统一改成 UTF-8
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

    task   = sys.argv[1] if len(sys.argv) > 1 else 'a'
    rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    harness = SmartTrainingHarness(task=task, max_rounds=rounds)
    harness.run()

    print("\n✅ Harness执行完成！")


if __name__ == '__main__':
    main()
