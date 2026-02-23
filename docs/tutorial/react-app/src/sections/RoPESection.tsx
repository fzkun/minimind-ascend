import { useState, useMemo, useRef, useCallback, useEffect } from 'react';
import Card from '../components/Card';
import SourcePanel from '../components/SourcePanel';
import { useTheme } from '../context/ThemeContext';
import { useCanvas } from '../hooks/useCanvas';
import { MM } from '../constants';

const ROPE_STEPS = [
  { name: '原始 Q/K 向量', desc: '从注意力投影得到 Q 和 K，每个头 64 维。这些向量还没有位置信息。', analogy: '拿到一份 64 格的空白卡片，还不知道是第几个位置的', color: '#64748b' },
  { name: '两两配对', desc: '将 64 维拆成 32 组，每组 2 个相邻维度视为一个 2D 向量：(q₀,q₁), (q₂,q₃), ..., (q₆₂,q₆₃)。', analogy: '64 格卡片两两配对，变成 32 对"舞伴"', color: '#3b82f6' },
  { name: '计算频率', desc: 'freq[i] = 1 / (base^(2i/d))，i=0 时频率最高（旋转最快），i=31 时频率最低（旋转最慢）。base=1000000。', analogy: '给 32 对舞伴分配不同的"旋转速度"，第 1 对转最快，最后 1 对转最慢', color: '#10b981' },
  { name: '计算旋转角', desc: 'θᵢ = position × freq[i]。位置越大，角度越大。高频维度角度增长快，低频维度增长慢。', analogy: '座位号 × 旋转速度 = 实际要转多少度', color: '#f59e0b' },
  { name: '旋转变换', desc: "q' = q·cos(θ) + rotate(q)·sin(θ)。每对 2D 向量按各自的角度做旋转矩阵变换，K 同理。", analogy: '每对舞伴按自己的角度原地旋转，Q 和 K 都转', color: '#8b5cf6' },
  { name: '位置感知完成', desc: '旋转后：相同内容在不同位置→不同向量，Q·K 点积只取决于相对距离。V 不参与旋转，不需要位置信息。', analogy: '现在每张卡片都"刻上"了位置信息，相对距离自动编码在点积中', color: '#06b6d4' },
];

export default function RoPESection() {
  const { isDark } = useTheme();
  const [pos, setPos] = useState(0);

  // RoPE step animation state
  const [ropeStep, setRopeStep] = useState(-1);
  const [ropePlaying, setRopePlaying] = useState(false);
  const ropeTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Auto-play rotation
  const [ropeAutoPlaying, setRopeAutoPlaying] = useState(false);
  const ropeAutoTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    return () => {
      if (ropeTimerRef.current) clearInterval(ropeTimerRef.current);
      if (ropeAutoTimerRef.current) clearInterval(ropeAutoTimerRef.current);
    };
  }, []);

  const stopRope = useCallback(() => {
    if (ropeTimerRef.current) { clearInterval(ropeTimerRef.current); ropeTimerRef.current = null; }
    setRopePlaying(false);
  }, []);

  const playRope = useCallback(() => {
    if (ropeTimerRef.current) { stopRope(); return; }
    setRopePlaying(true);
    ropeTimerRef.current = setInterval(() => {
      setRopeStep(prev => {
        if (prev < ROPE_STEPS.length - 1) return prev + 1;
        stopRope();
        return prev;
      });
    }, 1200);
  }, [stopRope]);

  const resetRope = useCallback(() => { stopRope(); setRopeStep(-1); }, [stopRope]);

  const toggleRopeAuto = useCallback(() => {
    if (ropeAutoTimerRef.current) {
      clearInterval(ropeAutoTimerRef.current);
      ropeAutoTimerRef.current = null;
      setRopeAutoPlaying(false);
      return;
    }
    setRopeAutoPlaying(true);
    ropeAutoTimerRef.current = setInterval(() => {
      setPos(prev => (prev >= 127 ? 0 : prev + 1));
    }, 100);
  }, []);

  // RoPE flow SVG
  const ropeFlowSvg = useMemo(() => {
    const fg = isDark ? '#e2e8f0' : '#1a1a2e';
    const fg2 = isDark ? '#94a3b8' : '#555';

    const boxes = ROPE_STEPS.map((s, i) => ({
      y: 10 + i * 46,
      h: 34,
      label: `${i}. ${s.name}`,
      color: s.color,
    }));

    let html = `<defs><marker id="arrRope" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0 L10,5 L0,10 Z" fill="${fg2}"/></marker></defs>`;

    boxes.forEach((b, i) => {
      const isActive = i === ropeStep;
      const isDone = i < ropeStep;
      const opacity = isActive ? 0.35 : isDone ? 0.2 : 0.06;
      const strokeW = isActive ? 2.5 : 1.5;
      const dash = !isActive && !isDone ? 'stroke-dasharray="4,2"' : '';

      html += `<rect x="20" y="${b.y}" width="260" height="${b.h}" rx="6" fill="${b.color}" opacity="${opacity}" stroke="${b.color}" stroke-width="${strokeW}" ${dash} style="cursor:pointer" data-rstep="${i}"/>`;
      html += `<text x="150" y="${b.y + b.h / 2 + 5}" text-anchor="middle" fill="${isActive ? fg : fg2}" font-size="${isActive ? 12 : 10.5}" ${isActive ? 'font-weight="bold"' : ''} style="pointer-events:none">${b.label}</text>`;

      if (isActive) {
        html += `<circle cx="10" cy="${b.y + b.h / 2}" r="5" fill="${b.color}"><animate attributeName="r" values="4;7;4" dur="1s" repeatCount="indefinite"/></circle>`;
      }
      if (isDone) {
        html += `<text x="288" y="${b.y + b.h / 2 + 4}" fill="${b.color}" font-size="12">✓</text>`;
      }
      if (i < boxes.length - 1) {
        html += `<line x1="150" y1="${b.y + b.h}" x2="150" y2="${boxes[i + 1].y}" stroke="${fg2}" stroke-width="1" marker-end="url(#arrRope)" opacity="0.4"/>`;
      }
    });

    return html;
  }, [isDark, ropeStep]);

  // Step detail visualization for RoPE steps
  const ropeStepDetailSvg = useMemo(() => {
    if (ropeStep < 0) return '';
    const fg = isDark ? '#e2e8f0' : '#1a1a2e';
    const fg2 = isDark ? '#94a3b8' : '#555';
    const accent = isDark ? '#818cf8' : '#4f46e5';
    let html = '';

    if (ropeStep === 1) {
      // Pair visualization
      const dims = ['q₀','q₁','q₂','q₃','q₄','q₅','...','q₆₂','q₆₃'];
      dims.forEach((d, i) => {
        const x = 10 + i * 30;
        const isPair = i < 6 || i >= 7;
        const pairIdx = i < 6 ? Math.floor(i / 2) : (i >= 7 ? 30 + Math.floor((i - 7) / 2) : -1);
        const colors = [accent, '#10b981', '#f59e0b', '#8b5cf6'];
        const color = pairIdx >= 0 ? colors[pairIdx % colors.length] : fg2;
        html += `<rect x="${x}" y="5" width="26" height="22" rx="3" fill="${color}" opacity="0.2" stroke="${color}" stroke-width="1"/>`;
        html += `<text x="${x + 13}" y="20" text-anchor="middle" fill="${fg}" font-size="8">${d}</text>`;
      });
      // Brackets for pairs
      html += `<path d="M 11 30 L 11 35 L 63 35 L 63 30" fill="none" stroke="${accent}" stroke-width="1"/>`;
      html += `<text x="37" y="46" text-anchor="middle" fill="${accent}" font-size="8">pair 0</text>`;
      html += `<path d="M 71 30 L 71 35 L 123 35 L 123 30" fill="none" stroke="#10b981" stroke-width="1"/>`;
      html += `<text x="97" y="46" text-anchor="middle" fill="#10b981" font-size="8">pair 1</text>`;
      html += `<path d="M 131 30 L 131 35 L 183 35 L 183 30" fill="none" stroke="#f59e0b" stroke-width="1"/>`;
      html += `<text x="157" y="46" text-anchor="middle" fill="#f59e0b" font-size="8">pair 2</text>`;
    } else if (ropeStep === 2) {
      // Frequency values
      const freqs = [0, 4, 8, 16, 24, 31].map(i => ({
        i,
        freq: 1.0 / Math.pow(1000000, (2 * i) / 64),
      }));
      html += `<text x="5" y="14" fill="${fg}" font-size="9" font-weight="bold">freq[i] = 1/(1000000^(2i/64))</text>`;
      freqs.forEach((f, idx) => {
        const y = 28 + idx * 18;
        const barW = Math.max(2, Math.min(180, f.freq * 180));
        html += `<text x="5" y="${y + 10}" fill="${fg2}" font-size="8">i=${f.i}:</text>`;
        html += `<rect x="40" y="${y}" width="${barW}" height="12" rx="2" fill="${accent}" opacity="0.6"/>`;
        html += `<text x="${45 + barW}" y="${y + 10}" fill="${fg}" font-size="8">${f.freq.toExponential(1)}</text>`;
      });
    } else if (ropeStep >= 3 && ropeStep <= 4) {
      // cos/sin values
      const examplePos = 10;
      html += `<text x="5" y="14" fill="${fg}" font-size="9" font-weight="bold">position = ${examplePos} 时的旋转角</text>`;
      [0, 8, 16, 31].forEach((i, idx) => {
        const freq = 1.0 / Math.pow(1000000, (2 * i) / 64);
        const theta = examplePos * freq;
        const y = 28 + idx * 28;
        html += `<text x="5" y="${y + 10}" fill="${fg2}" font-size="8">pair ${i}:</text>`;
        html += `<text x="50" y="${y + 10}" fill="${fg}" font-size="8">θ=${theta.toFixed(4)}</text>`;
        html += `<text x="120" y="${y + 10}" fill="#10b981" font-size="8">cos=${Math.cos(theta).toFixed(3)}</text>`;
        html += `<text x="190" y="${y + 10}" fill="#f59e0b" font-size="8">sin=${Math.sin(theta).toFixed(3)}</text>`;
      });
    }

    return html;
  }, [isDark, ropeStep]);

  // RoPE rotation canvas 1 (high freq, channel 0)
  const canvas1Ref = useCanvas(
    (ctx, size) => {
      drawRotationCanvas(ctx, size, 0, pos, isDark);
    },
    [isDark, pos],
    250,
    250,
  );

  // RoPE rotation canvas 2 (low freq, channel 31)
  const canvas2Ref = useCanvas(
    (ctx, size) => {
      drawRotationCanvas(ctx, size, 31, pos, isDark);
    },
    [isDark, pos],
    250,
    250,
  );

  // Dot product canvas
  const dotCanvasRef = useCanvas(
    (ctx, w, h) => {
      ctx.fillStyle = isDark ? '#1e293b' : '#f8f8f8';
      ctx.fillRect(0, 0, w, h);

      const padL = 50, padR = 20, padT = 30, padB = 40;
      const plotW = w - padL - padR, plotH = h - padT - padB;
      const maxDist = 64;

      const dots: number[] = [];
      for (let d = 0; d <= maxDist; d++) {
        let dp = 0;
        for (let i = 0; i < MM.head_dim / 2; i++) {
          const freq = 1.0 / Math.pow(MM.rope_base, (2 * i) / MM.head_dim);
          dp += Math.cos(d * freq);
        }
        dp /= MM.head_dim / 2;
        dots.push(dp);
      }
      const maxVal = Math.max(...dots.map(Math.abs));

      const fg = isDark ? '#e2e8f0' : '#1a1a2e';
      const fg2 = isDark ? '#64748b' : '#aaa';
      ctx.strokeStyle = fg2;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(padL, padT);
      ctx.lineTo(padL, padT + plotH);
      ctx.lineTo(padL + plotW, padT + plotH);
      ctx.stroke();

      ctx.fillStyle = fg;
      ctx.font = '10px monospace';
      ctx.textAlign = 'right';
      ctx.fillText('1.0', padL - 5, padT + 5);
      ctx.fillText('0', padL - 5, padT + plotH / 2 + 3);
      ctx.fillText('-1.0', padL - 5, padT + plotH + 3);

      ctx.textAlign = 'center';
      for (let d = 0; d <= maxDist; d += 16) {
        const x = padL + (d / maxDist) * plotW;
        ctx.fillText(String(d), x, padT + plotH + 15);
      }
      ctx.fillText('相对距离 |m-n|', padL + plotW / 2, h - 5);
      ctx.save();
      ctx.translate(12, padT + plotH / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.fillText('点积 (归一化)', 0, 0);
      ctx.restore();

      ctx.strokeStyle = fg2;
      ctx.lineWidth = 0.5;
      ctx.setLineDash([3, 3]);
      const zeroY = padT + plotH / 2;
      ctx.beginPath();
      ctx.moveTo(padL, zeroY);
      ctx.lineTo(padL + plotW, zeroY);
      ctx.stroke();
      ctx.setLineDash([]);

      const accent = isDark ? '#818cf8' : '#4f46e5';
      ctx.strokeStyle = accent;
      ctx.lineWidth = 2;
      ctx.beginPath();
      for (let d = 0; d <= maxDist; d++) {
        const x = padL + (d / maxDist) * plotW;
        const y = padT + plotH / 2 - (dots[d] / maxVal) * (plotH / 2);
        if (d === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();

      ctx.fillStyle = fg;
      ctx.font = '11px sans-serif';
      ctx.textAlign = 'left';
      ctx.fillText('RoPE Q·K 点积随相对距离衰减（维度平均）', padL, 18);
    },
    [isDark],
    500,
    200,
  );

  // YaRN SVG
  const yarnSvg = useMemo(() => {
    const fg = isDark ? '#e2e8f0' : '#1a1a2e';
    const fg2 = isDark ? '#94a3b8' : '#555';
    const green = isDark ? '#34d399' : '#10b981';
    const orange = isDark ? '#fbbf24' : '#f59e0b';
    const red = isDark ? '#f87171' : '#ef4444';
    const bg = isDark ? '#1e293b' : '#f0f0f0';

    const dim = MM.head_dim;
    const inv_dim = (b: number) => (dim * Math.log(MM.yarn_orig_max / (b * 2 * Math.PI))) / (2 * Math.log(MM.rope_base));
    const low = Math.max(Math.floor(inv_dim(MM.yarn_beta_fast)), 0);
    const high = Math.min(Math.ceil(inv_dim(MM.yarn_beta_slow)), dim / 2 - 1);

    const padL = 60, padT = 30, plotW = 520, plotH = 140;

    let bars = '';
    for (let i = 0; i < dim / 2; i++) {
      const ramp = Math.max(0, Math.min(1, (i - low) / Math.max(high - low, 0.001)));
      const scale = 1 - ramp + ramp / MM.yarn_factor;
      const x = padL + (i / (dim / 2)) * plotW;
      const barW = plotW / (dim / 2) - 0.5;
      const barH = scale * plotH;
      let color: string;
      if (ramp <= 0) color = green;
      else if (ramp >= 1) color = red;
      else color = orange;
      bars += `<rect x="${x}" y="${padT + plotH - barH}" width="${Math.max(barW, 1)}" height="${barH}" fill="${color}" opacity="0.7"/>`;
    }

    return `
      <rect x="${padL}" y="${padT}" width="${plotW}" height="${plotH}" fill="${bg}" rx="4"/>
      ${bars}
      <line x1="${padL}" y1="${padT + plotH}" x2="${padL + plotW}" y2="${padT + plotH}" stroke="${fg2}" stroke-width="1"/>
      <line x1="${padL}" y1="${padT}" x2="${padL}" y2="${padT + plotH}" stroke="${fg2}" stroke-width="1"/>
      <text x="${padL + plotW / 2}" y="${padT + plotH + 30}" text-anchor="middle" fill="${fg}" font-size="11">频率维度 i (0 ~ ${dim / 2 - 1})</text>
      <text x="${padL - 5}" y="${padT + 5}" text-anchor="end" fill="${fg}" font-size="10">1.0</text>
      <text x="${padL - 5}" y="${padT + plotH}" text-anchor="end" fill="${fg}" font-size="10">${(1 / MM.yarn_factor).toFixed(3)}</text>
      <text x="10" y="${padT + plotH / 2}" text-anchor="middle" fill="${fg}" font-size="10" transform="rotate(-90, 10, ${padT + plotH / 2})">缩放系数</text>
      <text x="${padL + plotW / 2}" y="18" text-anchor="middle" fill="${fg}" font-size="12" font-weight="bold">YaRN: f'(i) = f(i) × ((1-γ) + γ/s)，s=${MM.yarn_factor}</text>
      <line x1="${padL + (low / (dim / 2)) * plotW}" y1="${padT}" x2="${padL + (low / (dim / 2)) * plotW}" y2="${padT + plotH}" stroke="${fg2}" stroke-width="1" stroke-dasharray="3,3"/>
      <line x1="${padL + (high / (dim / 2)) * plotW}" y1="${padT}" x2="${padL + (high / (dim / 2)) * plotW}" y2="${padT + plotH}" stroke="${fg2}" stroke-width="1" stroke-dasharray="3,3"/>
      <text x="${padL + (low / (dim / 2)) * plotW}" y="${padT + plotH + 15}" text-anchor="middle" fill="${fg2}" font-size="9">low=${low}</text>
      <text x="${padL + (high / (dim / 2)) * plotW}" y="${padT + plotH + 15}" text-anchor="middle" fill="${fg2}" font-size="9">high=${high}</text>
    `;
  }, [isDark]);

  return (
    <>
      <h2>4. RoPE 位置编码</h2>
      <p className="desc">
        旋转位置编码 (RoPE) 把位置信息注入到 Q/K 向量中，让 attention score 自然反映 token 之间的相对距离。
        核心操作是对 Q/K 的每对相邻维度做旋转，角度由 <code>rope_theta=1e6</code> 和位置索引决定。
        MiniMind 还支持 YaRN 频率缩放，实现长上下文外推。
        <br/>
        <small style={{ color: 'var(--fg2)' }}>
          关联源码：<code>model/model_minimind.py:109</code> <code>def precompute_freqs_cis()</code> | <code>:105</code> <code>def forward</code> (apply_rotary_emb)
        </small>
      </p>

      <Card title="RoPE 如何工作 — 逐步动画">
        <p style={{ marginBottom: 10, fontSize: '0.9rem', color: 'var(--fg2)' }}>
          从原始向量到位置感知，6 步理解旋转位置编码的完整过程。
        </p>
        <div style={{ display: 'flex', gap: 8, marginBottom: 10, flexWrap: 'wrap' }}>
          <button className="btn" onClick={() => setRopeStep(prev => Math.max(prev - 1, 0))}>◀ 上一步</button>
          <button className="btn primary" onClick={() => setRopeStep(prev => Math.min(prev + 1, ROPE_STEPS.length - 1))}>下一步 ▶</button>
          <button className="btn" onClick={playRope}>{ropePlaying ? '⏸ 暂停' : '▶ 自动播放'}</button>
          <button className="btn" onClick={resetRope}>重置</button>
        </div>
        <div className="step-indicator">
          {ROPE_STEPS.map((s, i) => (
            <div
              key={i}
              className={`step-dot${i === ropeStep ? ' active' : ''}${i < ropeStep ? ' done' : ''}`}
              onClick={() => setRopeStep(i)}
              title={s.name}
            >
              {i}
            </div>
          ))}
        </div>
        <div className="viz-grid">
          <div>
            <svg
              width="100%"
              height={290}
              viewBox="0 0 310 290"
              onClick={(e) => {
                const t = (e.target as SVGElement).closest('rect[data-rstep]');
                if (t) setRopeStep(parseInt(t.getAttribute('data-rstep')!));
              }}
              dangerouslySetInnerHTML={{ __html: ropeFlowSvg }}
            />
          </div>
          <div>
            {ropeStep >= 0 ? (() => {
              const cur = ROPE_STEPS[ropeStep];
              return (
                <>
                  <div style={{ fontSize: '1.1rem', fontWeight: 700, color: cur.color, marginBottom: 6 }}>
                    {ropeStep}. {cur.name}
                  </div>
                  <div style={{
                    padding: '10px 14px',
                    background: isDark ? '#1e293b' : '#fffbeb',
                    border: `2px solid ${cur.color}`,
                    borderRadius: 'var(--radius)',
                    marginBottom: 10,
                    fontSize: '0.92rem',
                  }}>
                    <strong>大白话：</strong>{cur.analogy}
                  </div>
                  <div className="label">说明</div>
                  <p style={{ fontSize: '0.88rem', color: 'var(--fg2)', marginBottom: 10 }}>{cur.desc}</p>
                  {ropeStepDetailSvg && (
                    <svg width="100%" height={ropeStep === 1 ? 55 : 140} viewBox={`0 0 280 ${ropeStep === 1 ? 55 : 140}`} dangerouslySetInnerHTML={{ __html: ropeStepDetailSvg }} />
                  )}
                </>
              );
            })() : (
              <div style={{ color: 'var(--fg3)', fontSize: '0.9rem', padding: 20 }}>
                点击「下一步」或左侧流程图开始演示
              </div>
            )}
          </div>
        </div>
      </Card>

      <Card title="旋转向量动画">
        <p style={{ marginBottom: 10, fontSize: '0.9rem', color: 'var(--fg2)' }}>
          RoPE 将 Q/K 向量的 64 个维度两两配对成 32 组，每组视为一个 2D 向量并旋转 θ 角度。
          旋转角 θ = position × freq[i]，其中 freq[i] = 1/(base^(2i/d))。低维度组频率高（旋转快），高维度组频率低（旋转慢）。
        </p>
        <p style={{ marginBottom: 10, fontSize: '0.9rem', color: 'var(--fg2)' }}>
          拖动下方滑块改变 position 值，观察两个频道的向量旋转情况：左图是频道 0（高频，每步旋转角度大），
          右图是频道 31（低频，需要很大的 position 差异才能看到明显旋转）。半透明小点表示历史位置的轨迹，绿色虚线是 position=0 时的初始方向。
        </p>
        <div className="viz-grid">
          <div>
            <div className="label">频道 0（高频，快速旋转）</div>
            <canvas ref={canvas1Ref} />
          </div>
          <div>
            <div className="label">频道 31（低频，缓慢旋转）</div>
            <canvas ref={canvas2Ref} />
          </div>
        </div>
        <div style={{ marginTop: 10 }}>
          <div className="label" style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            位置 position = <span className="value">{pos}</span>
            <button className="btn" style={{ fontSize: '0.8rem', padding: '2px 10px' }} onClick={toggleRopeAuto}>
              {ropeAutoPlaying ? '⏸ 停止' : '▶ 自动播放'}
            </button>
          </div>
          <input
            type="range"
            min="0"
            max="127"
            step="1"
            value={pos}
            onChange={e => setPos(parseInt(e.target.value))}
          />
        </div>
        <SourcePanel
          title="对照源码：model/model_minimind.py:109-137"
          code={`def precompute_freqs_cis(dim, end=32768, rope_base=1e6, rope_scaling=None):
    """预计算所有位置的旋转频率 cos/sin，模型初始化时调用一次"""
    # 计算每个维度对的基础频率：freq[i] = 1 / (base ^ (2i/dim))
    # dim=64 时产生 32 个频率，从高频 (i=0) 到低频 (i=31)
    freqs = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[:dim//2].float() / dim))
    # t = [0, 1, 2, ..., end-1]，代表所有可能的位置
    t = torch.arange(end)
    # 外积得到旋转角矩阵：theta[pos][i] = pos * freq[i]
    freqs = torch.outer(t, freqs)  # [seq_len, dim/2]
    # 复制一份拼接，使 cos/sin 维度与 head_dim 对齐（方便后续逐元素相乘）
    freqs_cos = torch.cat([cos(freqs), cos(freqs)], dim=-1)  # [seq_len, dim]
    freqs_sin = torch.cat([sin(freqs), sin(freqs)], dim=-1)  # [seq_len, dim]
    return freqs_cos, freqs_sin

def apply_rotary_pos_emb(q, k, cos, sin):
    """将旋转位置编码应用到 Q 和 K（V 不需要位置信息）"""
    def rotate_half(x):
        # 将向量拆为前后两半并交换，配合负号实现 2D 旋转矩阵的效果
        # [x0, x1, ..., x31, x32, ..., x63] → [-x32, ..., -x63, x0, ..., x31]
        return torch.cat((-x[..., x.shape[-1]//2:], x[..., :x.shape[-1]//2]), dim=-1)
    # 旋转公式：q' = q * cos(θ) + rotate_half(q) * sin(θ)
    # 这等价于对每对 (q_{2i}, q_{2i+1}) 做 2D 旋转矩阵运算
    q_embed = q * cos + rotate_half(q) * sin
    k_embed = k * cos + rotate_half(k) * sin
    return q_embed, k_embed`}
        />
      </Card>

      <Card title="相对位置点积">
        <p style={{ marginBottom: 10, fontSize: '0.9rem', color: 'var(--fg2)' }}>
          RoPE 最关键的数学性质：对位置 m 的 Q 和位置 n 的 K 施加旋转后，它们的点积 Q_m · K_n 只取决于相对距离 |m-n|，
          与绝对位置无关。这使模型天然具备处理不同长度序列的能力。
        </p>
        <p style={{ marginBottom: 10, fontSize: '0.9rem', color: 'var(--fg2)' }}>
          下图展示了归一化后的平均点积随相对距离的变化曲线。可以观察到：距离越近，点积越大（注意力越强）；
          距离越远，点积衰减并振荡趋近于零。这种自然的远近衰减正是注意力机制所需要的。
        </p>
        <canvas ref={dotCanvasRef} />
      </Card>

      <Card title="YaRN 频率缩放">
        <p style={{ marginBottom: 10, fontSize: '0.9rem', color: 'var(--fg2)' }}>
          当推理长度超过训练长度时，高频维度的旋转角过大会导致注意力崩溃。YaRN 通过对频率维度分区缩放来解决这个问题：
        </p>
        <p style={{ marginBottom: 10, fontSize: '0.9rem', color: 'var(--fg2)' }}>
          绿色区域（高频）：这些维度在训练范围内已经充分旋转，不需要缩放，保持原始频率。
          橙色区域（过渡带）：通过线性插值 γ(i) 在不缩放和完全缩放之间平滑过渡。
          红色区域（低频）：这些维度旋转最慢，外推时影响最大，频率除以 factor=16 进行压缩。
          MiniMind 通过这种方式将上下文窗口从 2048 扩展到 32768 token。
        </p>
        <svg width="100%" height={220} viewBox="0 0 600 220" dangerouslySetInnerHTML={{ __html: yarnSvg }} />
        <div style={{ marginTop: 8, display: 'flex', gap: 16, flexWrap: 'wrap', fontSize: '0.85rem' }}>
          <span style={{ color: 'var(--green)' }}>■ 高频（不缩放）</span>
          <span style={{ color: 'var(--orange)' }}>■ 过渡带（线性插值）</span>
          <span style={{ color: 'var(--red)' }}>■ 低频（÷16 缩放）</span>
        </div>
      </Card>
    </>
  );
}

function drawRotationCanvas(
  ctx: CanvasRenderingContext2D,
  size: number,
  freqIdx: number,
  pos: number,
  isDark: boolean,
) {
  const bg = isDark ? '#1e293b' : '#f8f8f8';
  const fg = isDark ? '#e2e8f0' : '#1a1a2e';
  const fg2 = isDark ? '#475569' : '#ddd';
  const accent = isDark ? '#818cf8' : '#4f46e5';
  const green = isDark ? '#34d399' : '#10b981';

  ctx.fillStyle = bg;
  ctx.fillRect(0, 0, size, size);

  const cx = size / 2, cy = size / 2, r = 90;

  ctx.strokeStyle = fg2;
  ctx.lineWidth = 0.5;
  ctx.beginPath(); ctx.moveTo(cx - r - 20, cy); ctx.lineTo(cx + r + 20, cy); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(cx, cy - r - 20); ctx.lineTo(cx, cy + r + 20); ctx.stroke();
  ctx.beginPath(); ctx.arc(cx, cy, r, 0, Math.PI * 2); ctx.stroke();

  const freq = 1.0 / Math.pow(MM.rope_base, (2 * freqIdx) / MM.head_dim);
  const theta = pos * freq;

  ctx.globalAlpha = 0.15;
  for (let p = 0; p < pos; p++) {
    const th = p * freq;
    const px = cx + r * Math.cos(th);
    const py = cy - r * Math.sin(th);
    ctx.fillStyle = accent;
    ctx.beginPath(); ctx.arc(px, py, 2, 0, Math.PI * 2); ctx.fill();
  }
  ctx.globalAlpha = 1;

  const vx = cx + r * Math.cos(theta);
  const vy = cy - r * Math.sin(theta);
  ctx.strokeStyle = accent;
  ctx.lineWidth = 2.5;
  ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(vx, vy); ctx.stroke();

  ctx.fillStyle = accent;
  ctx.beginPath(); ctx.arc(vx, vy, 5, 0, Math.PI * 2); ctx.fill();

  const ox = cx + r;
  ctx.strokeStyle = green;
  ctx.lineWidth = 1.5;
  ctx.setLineDash([4, 3]);
  ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(ox, cy); ctx.stroke();
  ctx.setLineDash([]);

  ctx.fillStyle = fg;
  ctx.font = '11px monospace';
  ctx.textAlign = 'left';
  ctx.fillText(`freq[${freqIdx}] = ${freq.toExponential(2)}`, 8, 18);
  ctx.fillText(`θ = pos × freq = ${theta.toFixed(3)}`, 8, 33);
  ctx.fillText(`cos(θ) = ${Math.cos(theta).toFixed(3)}`, 8, size - 20);
  ctx.fillText(`sin(θ) = ${Math.sin(theta).toFixed(3)}`, 8, size - 6);
}
