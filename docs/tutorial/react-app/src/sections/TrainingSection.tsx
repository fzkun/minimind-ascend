import { useState, useMemo, useRef, useEffect, useCallback } from 'react';
import Card from '../components/Card';
import SourcePanel from '../components/SourcePanel';
import { useTheme } from '../context/ThemeContext';
import { useCanvas } from '../hooks/useCanvas';

interface Stage {
  name: string;
  x: number;
  color: string;
  desc: string;
}

const STAGES: Stage[] = [
  { name: 'Pretrain', x: 20, color: '#3b82f6', desc: '预训练：在大规模文本语料上学习语言模式。lr=5e-4, 损失函数=CrossEntropy(next token prediction)。输出: pretrain_512.pth' },
  { name: 'Full SFT', x: 170, color: '#10b981', desc: '监督微调：在对话数据上训练，学习指令跟随能力。lr=1e-6, 只对 assistant 回复计算损失。输出: full_sft_512.pth' },
  { name: 'LoRA', x: 310, color: '#f59e0b', desc: 'LoRA 轻量微调：冻结主干，只训练低秩适配器。rank=16, alpha=16, 可训练参数仅 0.5M。输出: lora_512.pth' },
  { name: 'DPO', x: 440, color: '#ef4444', desc: 'DPO 对齐：通过 chosen/rejected 对比优化。β=0.1, lr=4e-8。无需训练奖励模型。输出: dpo_512.pth' },
  { name: 'GRPO', x: 560, color: '#8b5cf6', desc: 'GRPO 在线 RL：Group Relative Policy Optimization，采样多个回复计算组内相对奖励。输出: grpo_512.pth' },
  { name: 'Inference', x: 690, color: '#06b6d4', desc: '推理部署：加载训练好的权重，KV Cache + 采样策略 (top-p, temperature) 生成文本。' },
];

const CONVERSATION = [
  { role: 'system', tokens: ['<|im_start|>', 'system', '\\n', '你', '是', 'Mini', 'Mind', '<|im_end|>'] },
  { role: 'user', tokens: ['<|im_start|>', 'user', '\\n', '你', '好', '吗', '？', '<|im_end|>'] },
  { role: 'assistant', tokens: ['<|im_start|>', 'assistant', '\\n', '我', '很', '好', '！', '<|im_end|>'] },
];

export default function TrainingSection() {
  const { isDark } = useTheme();
  const [selectedStage, setSelectedStage] = useState<number | null>(null);
  const [beta, setBeta] = useState(0.1);
  const svgRef = useRef<SVGSVGElement>(null);

  // Pipeline SVG
  const pipelineSvg = useMemo(() => {
    const fg = isDark ? '#e2e8f0' : '#1a1a2e';
    const fg2 = isDark ? '#94a3b8' : '#555';
    let html = `<defs><marker id="arrPipe" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0 L10,5 L0,10 Z" fill="${fg2}"/></marker></defs>`;
    STAGES.forEach((s, i) => {
      const opacity = selectedStage === i ? 0.4 : 0.15;
      html += `<rect x="${s.x}" y="35" width="110" height="42" rx="8" fill="${s.color}" opacity="${opacity}" stroke="${s.color}" stroke-width="2" style="cursor:pointer" data-stage="${i}"/>`;
      html += `<text x="${s.x + 55}" y="61" text-anchor="middle" fill="${fg}" font-size="12" font-weight="bold" style="pointer-events:none">${s.name}</text>`;
      if (i < STAGES.length - 1) {
        html += `<line x1="${s.x + 110}" y1="56" x2="${STAGES[i + 1].x}" y2="56" stroke="${fg2}" stroke-width="1.5" marker-end="url(#arrPipe)"/>`;
      }
    });
    html += `<text x="400" y="105" text-anchor="middle" fill="${fg2}" font-size="10">每个阶段产出 out/{stage}_512.pth 权重文件，下一阶段通过 --from_weight 加载</text>`;
    return html;
  }, [isDark, selectedStage]);

  // Attach click handlers to SVG rects
  const handleSvgClick = useCallback((e: React.MouseEvent<SVGSVGElement>) => {
    const target = e.target as SVGElement;
    const rect = target.closest('rect[data-stage]');
    if (rect) {
      const idx = parseInt(rect.getAttribute('data-stage')!);
      setSelectedStage(idx);
    }
  }, []);

  // DPO canvas
  const dpoCanvasRef = useCanvas(
    (ctx, w, h) => {
      ctx.fillStyle = isDark ? '#1e293b' : '#f8f8f8';
      ctx.fillRect(0, 0, w, h);

      const padL = 50, padR = 20, padT = 20, padB = 30;
      const plotW = w - padL - padR, plotH = h - padT - padB;
      const fg = isDark ? '#e2e8f0' : '#1a1a2e';
      const fg2c = isDark ? '#475569' : '#ddd';

      ctx.strokeStyle = fg2c;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(padL, padT);
      ctx.lineTo(padL, padT + plotH);
      ctx.lineTo(padL + plotW, padT + plotH);
      ctx.stroke();

      const xRange = [-4, 4];
      const toX = (v: number) => padL + ((v - xRange[0]) / (xRange[1] - xRange[0])) * plotW;

      // σ(β·x)
      const green = isDark ? '#34d399' : '#10b981';
      ctx.strokeStyle = green;
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      for (let px = 0; px <= plotW; px++) {
        const x = xRange[0] + (px / plotW) * (xRange[1] - xRange[0]);
        const sig = 1 / (1 + Math.exp(-beta * x));
        const sy = padT + plotH - sig * plotH;
        if (px === 0) ctx.moveTo(padL + px, sy);
        else ctx.lineTo(padL + px, sy);
      }
      ctx.stroke();

      // -log σ(β·x)
      const red = isDark ? '#f87171' : '#ef4444';
      ctx.strokeStyle = red;
      ctx.lineWidth = 2;
      ctx.beginPath();
      for (let px = 0; px <= plotW; px++) {
        const x = xRange[0] + (px / plotW) * (xRange[1] - xRange[0]);
        const sig = 1 / (1 + Math.exp(-beta * x));
        const loss = -Math.log(Math.max(sig, 1e-7));
        const normedLoss = Math.min(loss / 4, 1);
        const sy = padT + plotH - normedLoss * plotH;
        if (px === 0) ctx.moveTo(padL + px, sy);
        else ctx.lineTo(padL + px, sy);
      }
      ctx.stroke();

      ctx.fillStyle = fg;
      ctx.font = '10px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('log π_ratio(chosen) - log π_ratio(rejected)', padL + plotW / 2, h - 3);
      ctx.textAlign = 'right';
      ctx.fillText('1.0', padL - 5, padT + 5);
      ctx.fillText('0', padL - 5, padT + plotH + 3);
      ctx.textAlign = 'left';
      ctx.fillStyle = green;
      ctx.fillText('σ(β·x)', padL + plotW - 60, padT + 15);
      ctx.fillStyle = red;
      ctx.fillText('-log σ (loss)', padL + plotW - 82, padT + 28);

      ctx.strokeStyle = fg2c;
      ctx.lineWidth = 0.5;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(toX(0), padT);
      ctx.lineTo(toX(0), padT + plotH);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = fg;
      ctx.font = '9px monospace';
      ctx.textAlign = 'center';
      ctx.fillText('0', toX(0), padT + plotH + 12);
    },
    [isDark, beta],
    400,
    160,
  );

  // LR canvas
  const lrCanvasRef = useCanvas(
    (ctx, w, h) => {
      ctx.fillStyle = isDark ? '#1e293b' : '#f8f8f8';
      ctx.fillRect(0, 0, w, h);

      const padL = 60, padR = 20, padT = 20, padB = 35;
      const plotW = w - padL - padR, plotH = h - padT - padB;
      const fg = isDark ? '#e2e8f0' : '#1a1a2e';
      const fg2c = isDark ? '#475569' : '#ddd';

      ctx.strokeStyle = fg2c;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(padL, padT);
      ctx.lineTo(padL, padT + plotH);
      ctx.lineTo(padL + plotW, padT + plotH);
      ctx.stroke();

      const getLR = (step: number, total: number, lr: number) => lr * (0.1 + 0.45 * (1 + Math.cos(Math.PI * step / total)));
      const curves = [
        { lr: 5e-4, color: isDark ? '#60a5fa' : '#3b82f6' },
        { lr: 1e-6, color: isDark ? '#34d399' : '#10b981' },
        { lr: 4e-8, color: isDark ? '#fbbf24' : '#f59e0b' },
      ];
      const totalSteps = 100;
      curves.forEach(curve => {
        ctx.strokeStyle = curve.color;
        ctx.lineWidth = 2;
        ctx.beginPath();
        for (let s = 0; s <= totalSteps; s++) {
          const lr = getLR(s, totalSteps, curve.lr);
          const normed = lr / curve.lr;
          const x = padL + (s / totalSteps) * plotW;
          const y = padT + plotH - normed * plotH;
          if (s === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }
        ctx.stroke();
      });

      ctx.fillStyle = fg;
      ctx.font = '10px monospace';
      ctx.textAlign = 'right';
      ctx.fillText('lr_max', padL - 5, padT + 8);
      ctx.fillText('0.1×lr', padL - 5, padT + plotH + 3);
      ctx.textAlign = 'center';
      ctx.fillText('0', padL, padT + plotH + 15);
      ctx.fillText('T/2', padL + plotW / 2, padT + plotH + 15);
      ctx.fillText('T', padL + plotW, padT + plotH + 15);
      ctx.fillText('训练步数 (归一化)', padL + plotW / 2, h - 3);
      ctx.textAlign = 'left';
      ctx.fillText('lr(t) = lr₀ × (0.1 + 0.45 × (1 + cos(πt/T)))', padL + 10, padT + 15);
    },
    [isDark],
    500,
    200,
  );

  return (
    <>
      <h2>6. 训练流程</h2>
      <p className="desc">
        MiniMind 的完整训练流程包含预训练、监督微调 (SFT)、对齐训练 (DPO/PPO/GRPO) 等阶段，每个阶段产出独立的权重文件。
      </p>

      <Card title="Pipeline 流程图">
        <p style={{ marginBottom: 10, fontSize: '0.9rem', color: 'var(--fg2)' }}>
          MiniMind 的训练分为多个阶段，数据从左到右流经各阶段。每个阶段产出独立的 .pth 权重文件，
          下一阶段通过 --from_weight 参数加载上一阶段的权重继续训练。点击各阶段方块查看详细说明：
        </p>
        <svg
          ref={svgRef}
          width="100%"
          height={120}
          viewBox="0 0 800 120"
          onClick={handleSvgClick}
          dangerouslySetInnerHTML={{ __html: pipelineSvg }}
        />
        <div style={{ marginTop: 10, fontSize: '0.85rem', color: 'var(--fg2)', minHeight: 40 }}>
          {selectedStage !== null && (
            <span>
              <strong style={{ color: STAGES[selectedStage].color }}>{STAGES[selectedStage].name}</strong>
              : {STAGES[selectedStage].desc}
            </span>
          )}
        </div>
      </Card>

      <Card title="SFT 损失掩码">
        <p style={{ marginBottom: 10, fontSize: '0.9rem', color: 'var(--fg2)' }}>
          监督微调（SFT）时，我们只希望模型学习生成 assistant 的回复，而不是复述 system 提示或 user 的问题。
          实现方式：将 system/user 部分的 labels 设为 -100（PyTorch CrossEntropyLoss 自动忽略该值），
          只有 assistant 回复处的 labels 保留真实 token ID，从而只对绿色部分计算梯度和损失。
        </p>
        <div style={{ padding: 10, background: 'var(--bg)', borderRadius: 'var(--radius)', border: '1px solid var(--border)', overflowX: 'auto' }}>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 3, alignItems: 'center' }}>
            {CONVERSATION.map((turn, ti) => {
              const isAssistant = turn.role === 'assistant';
              return turn.tokens.map((t, i) => (
                <span
                  key={`${ti}-${i}`}
                  className="token-box"
                  style={{
                    background: isAssistant ? 'var(--green)' : 'var(--bg3)',
                    color: isAssistant ? '#fff' : 'var(--fg3)',
                    fontSize: '0.8rem',
                  }}
                  title={isAssistant ? 'labels = token_id (计算损失)' : 'labels = -100 (忽略)'}
                >
                  {t}
                </span>
              ));
            })}
          </div>
        </div>
        <div style={{ marginTop: 8, display: 'flex', gap: 16, flexWrap: 'wrap', fontSize: '0.85rem' }}>
          <span><span className="token-box" style={{ background: 'var(--bg3)', color: 'var(--fg3)' }}>灰色</span> = labels=-100（不计算损失）</span>
          <span><span className="token-box" style={{ background: 'var(--green)', color: '#fff' }}>绿色</span> = 计算损失的 token</span>
        </div>
        <SourcePanel
          title="对照源码：dataset/lm_dataset.py:74-90 (generate_labels)"
          code={`def generate_labels(self, input_ids):
    """为 SFT 生成损失掩码：只对 assistant 回复部分计算损失"""
    # 初始化全部为 -100（CrossEntropyLoss 忽略此标签）
    labels = [-100] * len(input_ids)
    i = 0
    while i < len(input_ids):
        # 寻找 assistant 回复的起始标记 <|im_start|>assistant
        if input_ids[i:i+len(self.bos_id)] == self.bos_id:
            start = i + len(self.bos_id)  # 跳过 BOS 标记本身
            end = start
            # 向后扫描直到找到结束标记 <|im_end|>
            while end < len(input_ids):
                if input_ids[end:end+len(self.eos_id)] == self.eos_id:
                    break
                end += 1
            # 将 assistant 回复区间的 labels 设为真实 token ID
            # 这样只有这些位置会产生梯度，system/user 部分不参与训练
            for j in range(start, min(end+len(self.eos_id), self.max_length)):
                labels[j] = input_ids[j]  # 只有 assistant 回复被标记
            i = end + len(self.eos_id)
        else:
            i += 1
    return labels`}
        />
      </Card>

      <Card title="DPO 对比训练">
        <p style={{ marginBottom: 10, fontSize: '0.9rem', color: 'var(--fg2)' }}>
          DPO（Direct Preference Optimization）是一种无需训练奖励模型的对齐方法。
          核心思想：给定同一个问题的好回复（chosen）和差回复（rejected），通过调整模型概率使好回复的似然比差回复更高。
          β 参数控制偏离参考模型的惩罚力度——β 越大曲线越陡，模型更快学会区分好坏回复但也更容易过拟合。
          拖动滑块观察 σ(β·x) 曲线和 loss 曲线的变化：
        </p>
        <div className="viz-grid">
          <div>
            <div className="label" style={{ color: 'var(--green)' }}>✓ Chosen（好回复）</div>
            <div style={{ padding: 8, background: 'var(--bg)', border: '2px solid var(--green)', borderRadius: 'var(--radius)', fontSize: '0.85rem', minHeight: 60 }}>
              中国的首都是北京，位于华北平原北部。
            </div>
          </div>
          <div>
            <div className="label" style={{ color: 'var(--red)' }}>✗ Rejected（差回复）</div>
            <div style={{ padding: 8, background: 'var(--bg)', border: '2px solid var(--red)', borderRadius: 'var(--radius)', fontSize: '0.85rem', minHeight: 60 }}>
              中国的首都是上海，是最大的城市。
            </div>
          </div>
        </div>
        <div style={{ marginTop: 12 }}>
          <div className="label">DPO Loss = -log σ(β × (log π(chosen)/π₀(chosen) - log π(rejected)/π₀(rejected)))</div>
          <div style={{ marginTop: 6 }}>
            <span className="label">β = </span>
            <input
              type="range"
              min="0.01"
              max="0.5"
              step="0.01"
              value={beta}
              onChange={e => setBeta(parseFloat(e.target.value))}
              style={{ width: 200, verticalAlign: 'middle' }}
            />
            <span className="value">{beta.toFixed(2)}</span>
          </div>
          <canvas ref={dpoCanvasRef} style={{ marginTop: 8 }} />
        </div>
        <SourcePanel
          title="对照源码：trainer/train_dpo.py:33-51"
          code={`def dpo_loss(ref_log_probs, policy_log_probs, mask, beta):
    """DPO 损失函数：通过对比好/坏回复的概率比来优化模型"""
    # 计算序列级平均 log P(seq)：对每个 token 的 log_prob 求和后除以有效长度
    ref_log_probs = (ref_log_probs * mask).sum(dim=1) / seq_lengths
    policy_log_probs = (policy_log_probs * mask).sum(dim=1) / seq_lengths
    # 分别计算策略模型（正在训练的模型）和参考模型（冻结的初始模型）的 log 概率比
    # pi_logratios = log π(chosen) - log π(rejected)，策略模型对好坏回复的偏好差
    pi_logratios = chosen_policy - reject_policy
    # ref_logratios = log π₀(chosen) - log π₀(rejected)，参考模型的偏好差（基线）
    ref_logratios = chosen_ref - reject_ref
    # logits = 策略偏好 - 参考偏好，衡量模型相对于参考的改进量
    logits = pi_logratios - ref_logratios
    # DPO 核心：-log σ(β × logits)
    # 当策略模型比参考更偏好 chosen 时 logits > 0，loss 小
    # 当策略模型反而偏好 rejected 时 logits < 0，loss 大（惩罚）
    # β 控制偏离参考的惩罚强度
    loss = -F.logsigmoid(beta * logits)
    return loss.mean()`}
        />
      </Card>

      <Card title="学习率调度">
        <p style={{ marginBottom: 10, fontSize: '0.9rem', color: 'var(--fg2)' }}>
          学习率调度对训练稳定性至关重要。MiniMind 使用余弦退火策略：lr 从最大值平滑下降到 0.1×lr_max。
          公式为 lr(t) = lr₀ × (0.1 + 0.45 × (1 + cos(π·t/T)))。
          不同训练阶段使用差异极大的初始学习率——预训练 5e-4（需要大步探索），SFT 1e-6（微调，小步调整），DPO 4e-8（对齐，极小扰动）。
          下图以归一化方式叠加展示三条曲线（形状相同，只是纵轴缩放不同）：
        </p>
        <canvas ref={lrCanvasRef} />
        <div style={{ marginTop: 8, display: 'flex', gap: 16, flexWrap: 'wrap', fontSize: '0.85rem' }}>
          <span style={{ color: 'var(--blue)' }}>■ Pretrain: 5e-4</span>
          <span style={{ color: 'var(--green)' }}>■ SFT: 1e-6</span>
          <span style={{ color: 'var(--orange)' }}>■ DPO: 4e-8</span>
        </div>
        <SourcePanel
          title="对照源码：trainer/trainer_utils.py:48-49"
          code={`def get_lr(current_step, total_steps, lr):
    """余弦退火学习率调度
    - current_step=0 时：lr * (0.1 + 0.45 * (1 + cos(0))) = lr * 1.0 = lr_max
    - current_step=T/2 时：lr * (0.1 + 0.45 * (1 + cos(π/2))) = lr * 0.55
    - current_step=T 时：lr * (0.1 + 0.45 * (1 + cos(π))) = lr * 0.1 = lr_min
    最终学习率不会降到 0，而是保持 0.1 × lr_max 的最小值，避免训练末期完全停滞
    """
    return lr * (0.1 + 0.45 * (1 + math.cos(math.pi * current_step / total_steps)))`}
        />
      </Card>
    </>
  );
}
