import { useState, useRef, useEffect, useCallback, useMemo } from 'react';
import Card from '../components/Card';
import SourcePanel from '../components/SourcePanel';
import { useTheme } from '../context/ThemeContext';
import { MM } from '../constants';
import { softmax, mulberry32, tokenColor } from '../utils';

interface Step {
  name: string;
  shape: string;
  desc: string;
}

const STEPS: Step[] = [
  { name: 'Input IDs', shape: '[1, 4]', desc: '输入 token ID 序列，例如 [1, 340, 590, 16] 对应 "<s> 你 好 。"' },
  { name: 'Embedding', shape: '[1, 4, 512]', desc: 'Token IDs 通过 Embedding 矩阵查找，每个 ID 映射为 512 维向量。embed_tokens(input_ids)' },
  { name: 'Dropout', shape: '[1, 4, 512]', desc: '训练时随机丢弃部分激活值防止过拟合（推理时关闭）。' },
  { name: 'RoPE 预计算', shape: 'cos/sin: [4, 64]', desc: '预计算每个位置的旋转频率 cos(mθ) 和 sin(mθ)，后续所有层共享。' },
  { name: 'Block 0: RMSNorm', shape: '[1, 4, 512]', desc: '第一个 Transformer Block 的输入层归一化：x / RMS(x) × γ，eps=1e-5。' },
  { name: 'Block 0: Attention', shape: '[1, 4, 512]', desc: 'GQA 注意力: Q[8头]×K[2头]→scores→softmax→×V[2头]→concat→O_proj。加上残差连接。' },
  { name: 'Block 0: RMSNorm', shape: '[1, 4, 512]', desc: '注意力输出后的第二个 RMSNorm，准备进入 FFN。' },
  { name: 'Block 0: FFN', shape: '[1, 4, 512]', desc: 'SwiGLU FFN: down_proj(SiLU(gate_proj(x)) ⊙ up_proj(x))。中间维度 1408。加上残差连接。' },
  { name: 'Block 1-7: ×7', shape: '[1, 4, 512]', desc: '重复相同结构 7 次（共 8 个 Block），每个 Block 包含 Attention + FFN + 2×RMSNorm + 2×残差。' },
  { name: 'Final RMSNorm', shape: '[1, 4, 512]', desc: '最后一个 RMSNorm 归一化，准备投影到词表空间。' },
  { name: 'LM Head', shape: '[1, 4, 6400]', desc: 'Linear(512→6400) 投影到词表大小，与 Embedding 共享权重。输出每个位置对 6400 个 token 的 logits。' },
  { name: 'Softmax', shape: '[1, 4, 6400]', desc: 'softmax(logits/temperature) 将 logits 转换为概率分布。取最后一个位置的分布用于生成下一个 token。' },
  { name: 'Top-K 采样', shape: '→ token_id', desc: '从概率分布中根据 top-p/top-k 策略采样一个 token，拼接到序列末尾，重复直到遇到 EOS。' },
];

export default function ForwardPassSection() {
  const { isDark } = useTheme();
  const [currentStep, setCurrentStep] = useState(-1);
  const autoTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const [autoPlaying, setAutoPlaying] = useState(false);

  useEffect(() => {
    return () => {
      if (autoTimerRef.current) clearInterval(autoTimerRef.current);
    };
  }, []);

  const stopAuto = useCallback(() => {
    if (autoTimerRef.current) {
      clearInterval(autoTimerRef.current);
      autoTimerRef.current = null;
    }
    setAutoPlaying(false);
  }, []);

  const goToStep = useCallback((idx: number) => {
    setCurrentStep(idx);
  }, []);

  const handleNext = useCallback(() => {
    setCurrentStep(prev => (prev < STEPS.length - 1 ? prev + 1 : prev));
  }, []);

  const handlePrev = useCallback(() => {
    setCurrentStep(prev => (prev > 0 ? prev - 1 : prev));
  }, []);

  const handleReset = useCallback(() => {
    stopAuto();
    setCurrentStep(-1);
  }, [stopAuto]);

  const handleAuto = useCallback(() => {
    if (autoTimerRef.current) {
      stopAuto();
      return;
    }
    setAutoPlaying(true);
    autoTimerRef.current = setInterval(() => {
      setCurrentStep(prev => {
        if (prev < STEPS.length - 1) return prev + 1;
        stopAuto();
        return prev;
      });
    }, 1200);
  }, [stopAuto]);

  // SVG
  const svgHtml = useMemo(() => {
    const fg = isDark ? '#e2e8f0' : '#1a1a2e';
    const fg2 = isDark ? '#94a3b8' : '#555';
    const accent = isDark ? '#818cf8' : '#4f46e5';
    const green = isDark ? '#34d399' : '#10b981';
    const blue = isDark ? '#60a5fa' : '#3b82f6';
    const orange = isDark ? '#fbbf24' : '#f59e0b';

    const boxes = [
      { y: 10, h: 30, label: 'Input IDs [1,4]', color: fg2 },
      { y: 50, h: 30, label: 'Embedding [1,4,512]', color: blue },
      { y: 90, h: 25, label: 'Dropout', color: fg2 },
      { y: 122, h: 25, label: 'RoPE cos/sin', color: orange },
      { y: 158, h: 25, label: 'RMSNorm₁', color: green },
      { y: 192, h: 30, label: 'Multi-Head Attention (GQA)', color: accent },
      { y: 232, h: 25, label: 'RMSNorm₂', color: green },
      { y: 266, h: 30, label: 'FFN (SwiGLU)', color: orange },
      { y: 308, h: 35, label: '× 7 more Blocks', color: fg2 },
      { y: 354, h: 25, label: 'Final RMSNorm', color: green },
      { y: 390, h: 30, label: 'LM Head [512→6400]', color: blue },
      { y: 430, h: 25, label: 'Softmax → Probs', color: accent },
      { y: 466, h: 30, label: 'Top-K Sampling → token', color: green },
    ];

    let html = `<defs><marker id="arrFwd" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0,0 L10,5 L0,10 Z" fill="${fg2}"/></marker></defs>`;

    boxes.forEach((b, i) => {
      const isActive = i === currentStep;
      const isDone = i < currentStep;
      const opacity = isActive ? 0.35 : isDone ? 0.2 : 0.08;
      const strokeW = isActive ? 2.5 : 1.5;
      const strokeDash = !isActive && !isDone ? 'stroke-dasharray="4,2"' : '';

      html += `<rect x="40" y="${b.y}" width="320" height="${b.h}" rx="6" fill="${b.color}" opacity="${opacity}" stroke="${b.color}" stroke-width="${strokeW}" ${strokeDash}/>`;
      html += `<text x="200" y="${b.y + b.h / 2 + 5}" text-anchor="middle" fill="${isActive ? fg : fg2}" font-size="${isActive ? 12 : 11}" ${isActive ? 'font-weight="bold"' : ''}>${b.label}</text>`;

      if (i < boxes.length - 1) {
        html += `<line x1="200" y1="${b.y + b.h}" x2="200" y2="${boxes[i + 1].y}" stroke="${fg2}" stroke-width="1" marker-end="url(#arrFwd)" opacity="0.5"/>`;
      }

      if (i === 5) {
        html += `<path d="M 365 ${boxes[3].y + boxes[3].h / 2} L 380 ${boxes[3].y + boxes[3].h / 2} L 380 ${b.y + b.h + 5} L 365 ${b.y + b.h + 5}" fill="none" stroke="${accent}" stroke-width="1" stroke-dasharray="3,2" opacity="0.5"/>`;
        html += `<text x="388" y="${(boxes[3].y + b.y + b.h) / 2 + 5}" fill="${accent}" font-size="8" opacity="0.7">+残差</text>`;
      }
      if (i === 7) {
        html += `<path d="M 365 ${boxes[5].y + boxes[5].h + 5} L 385 ${boxes[5].y + boxes[5].h + 5} L 385 ${b.y + b.h + 5} L 365 ${b.y + b.h + 5}" fill="none" stroke="${orange}" stroke-width="1" stroke-dasharray="3,2" opacity="0.5"/>`;
        html += `<text x="393" y="${(boxes[5].y + boxes[5].h + b.y + b.h) / 2 + 5}" fill="${orange}" font-size="8" opacity="0.7">+残差</text>`;
      }
    });

    html += `<rect x="25" y="155" width="370" height="150" rx="8" fill="none" stroke="${fg2}" stroke-width="1" stroke-dasharray="6,3" opacity="0.3"/>`;
    html += `<text x="30" y="152" fill="${fg2}" font-size="9" opacity="0.6">TransformerBlock ×${MM.num_layers}</text>`;

    return html;
  }, [isDark, currentStep]);

  // Top-K predictions
  const topKData = useMemo(() => {
    const rng = mulberry32(777);
    const candidates = ['谢', '我', '你', '很', '是', '好', '啊', '的', '了', '不'];
    const logits = candidates.map(() => (rng() - 0.3) * 5);
    const probs = softmax(logits, 0.8);
    const sorted = candidates
      .map((token, i) => ({ token, prob: probs[i] }))
      .sort((a, b) => b.prob - a.prob);
    return sorted;
  }, []);

  const step = currentStep >= 0 ? STEPS[currentStep] : null;

  return (
    <>
      <h2>7. 推理过程（Forward Pass）</h2>
      <p className="desc">
        一次完整的推理就是数据在模型中从头到尾走一遍的过程。用代码概括：<code>logits = lm_head(rms_norm(transformer_blocks(embedding(input_ids))))</code>，
        其中 <code>transformer_blocks</code> 重复 8 次，每次包含 Attention + FFN + 残差连接。
        <br/>
        <small style={{ color: 'var(--fg2)' }}>
          关联源码：<code>model/model_minimind.py:442</code> <code>MiniMindForCausalLM.forward()</code> | <code>:392</code> <code>MiniMindModel.forward()</code> | <code>:365</code> <code>MiniMindBlock.forward()</code>
        </small>
      </p>

      <Card title="逐步穿越动画">
        <p style={{ marginBottom: 10, fontSize: '0.9rem', color: 'var(--fg2)' }}>
          完整的前向传播包含 13 个关键步骤，从输入 token ID 到输出下一个 token 的概率分布。
          点击"下一步"逐步执行，观察数据在模型中的流动过程。左侧流程图高亮当前步骤（深色=已完成，虚线=未执行），
          右侧显示该步骤的 tensor shape 变化和详细说明。也可以点击编号圆点直接跳转到任意步骤。
          到达最后一步时，会展示模拟的 top-10 预测结果（token 概率分布）。
        </p>
        <div style={{ display: 'flex', gap: 8, marginBottom: 10, flexWrap: 'wrap' }}>
          <button className="btn" onClick={handlePrev}>◀ 上一步</button>
          <button className="btn primary" onClick={handleNext}>下一步 ▶</button>
          <button className="btn" onClick={handleAuto}>{autoPlaying ? '⏸ 暂停' : '▶ 自动播放'}</button>
          <button className="btn" onClick={handleReset}>重置</button>
        </div>

        {/* Step indicator dots */}
        <div className="step-indicator">
          {STEPS.map((_, i) => (
            <div
              key={i}
              className={`step-dot${i === currentStep ? ' active' : ''}${i < currentStep ? ' done' : ''}`}
              onClick={() => goToStep(i)}
            >
              {i + 1}
            </div>
          ))}
        </div>

        <div className="viz-grid">
          <div>
            <svg width="100%" height={520} viewBox="0 0 400 520" dangerouslySetInnerHTML={{ __html: svgHtml }} />
          </div>
          <div>
            <div className="label">当前步骤</div>
            <div style={{ fontSize: '1.1rem', fontWeight: 600, marginBottom: 8, color: 'var(--accent)' }}>
              {step ? step.name : '—'}
            </div>
            <div className="label">Tensor Shape</div>
            <div className="shape-badge" style={{ marginBottom: 12 }}>
              {step ? step.shape : '—'}
            </div>
            <div className="label">说明</div>
            <div style={{ fontSize: '0.9rem', color: 'var(--fg2)', minHeight: 80 }}>
              {step ? step.desc : '点击"下一步"开始'}
            </div>
            <div className="label" style={{ marginTop: 12 }}>Top-10 预测 (最终输出)</div>
            <div style={{ minHeight: 100 }}>
              {currentStep === STEPS.length - 1 ? (
                topKData.map((c, i) => (
                  <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 3 }}>
                    <span style={{ width: 30, fontSize: '0.8rem', color: 'var(--fg2)', textAlign: 'right', fontFamily: 'monospace' }}>
                      #{i + 1}
                    </span>
                    <span className="token-box" style={{ background: tokenColor(i), color: '#fff', minWidth: 28, textAlign: 'center' }}>
                      {c.token}
                    </span>
                    <div style={{ flex: 1, height: 14, background: 'var(--bg3)', borderRadius: 3, overflow: 'hidden' }}>
                      <div style={{
                        height: '100%',
                        width: (c.prob / topKData[0].prob * 100) + '%',
                        background: tokenColor(i),
                        borderRadius: 3,
                        transition: 'width 0.5s',
                      }} />
                    </div>
                    <span style={{ fontSize: '0.75rem', fontFamily: 'monospace', color: 'var(--accent)', width: 50 }}>
                      {(c.prob * 100).toFixed(1)}%
                    </span>
                  </div>
                ))
              ) : (
                <span style={{ color: 'var(--fg3)', fontSize: '0.85rem' }}>完成前向传播后显示</span>
              )}
            </div>
          </div>
        </div>

        <SourcePanel
          title="对照源码：model/model_minimind.py:392-468"
          code={`class MiniMindModel(nn.Module):
    """模型主干：Embedding + N × TransformerBlock + RMSNorm"""
    def forward(self, input_ids, ...):
        # Step 1-2: Token ID → Embedding 向量 → Dropout（训练时随机丢弃防止过拟合）
        hidden_states = self.dropout(self.embed_tokens(input_ids))  # [B,S] → [B,S,512]
        # Step 3: 取出预计算的 RoPE cos/sin，所有层共享
        position_embeddings = (freqs_cos[...], freqs_sin[...])
        # Step 4-8: 依次通过 8 个 TransformerBlock
        # 每个 Block: RMSNorm → Attention(+残差) → RMSNorm → FFN(+残差)
        for layer in self.layers:           # 8 × MiniMindBlock
            hidden_states, _ = layer(hidden_states, position_embeddings, ...)
        # Step 9: 最终 RMSNorm 归一化
        hidden_states = self.norm(hidden_states)     # RMSNorm
        return hidden_states  # [B,S,512]

class MiniMindForCausalLM(PreTrainedModel):
    """完整模型：主干 + LM Head（语言模型输出头）"""
    def forward(self, input_ids, labels=None, ...):
        # 通过模型主干获得隐藏状态
        hidden_states, _, aux_loss = self.model(input_ids, ...)
        # Step 10: LM Head 线性投影到词表空间（与 Embedding 共享权重）
        logits = self.lm_head(hidden_states)         # [B,S,512] → [B,S,6400]
        # 训练时：用 logits[:-1] 预测 labels[1:]（next token prediction）
        # labels 中 -100 位置被忽略（PAD 和非训练区域）
        if labels is not None:
            loss = cross_entropy(logits[:-1], labels[1:], ignore_index=-100)
        # 推理时：对 logits 做 softmax → 采样 → 得到下一个 token
        return CausalLMOutputWithPast(loss=loss, logits=logits, ...)`}
        />
      </Card>
    </>
  );
}
