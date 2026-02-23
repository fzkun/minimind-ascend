import { useState, useMemo } from 'react';
import Card from '../components/Card';
import SourcePanel from '../components/SourcePanel';
import { useTheme } from '../context/ThemeContext';
import { useCanvas } from '../hooks/useCanvas';
import { MM } from '../constants';

export default function RoPESection() {
  const { isDark } = useTheme();
  const [pos, setPos] = useState(0);

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
        旋转位置编码 (RoPE) 通过旋转向量将位置信息编码到 Q/K 中，使注意力分数自然反映相对位置。MiniMind 还支持 YaRN 频率缩放实现长上下文外推。
      </p>

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
          <div className="label">
            位置 position = <span className="value">{pos}</span>
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
