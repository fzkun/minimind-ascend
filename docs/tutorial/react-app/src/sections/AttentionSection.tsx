import { useState, useMemo, useRef, useCallback, useEffect } from 'react';
import Card from '../components/Card';
import SourcePanel from '../components/SourcePanel';
import { useTheme } from '../context/ThemeContext';
import { useCanvas } from '../hooks/useCanvas';
import { MM } from '../constants';
import { softmax, mulberry32, tokenColor } from '../utils';

const ATTN_STEPS = [
  { name: '输入向量', shape: '[1, 4, 512]', desc: '4 个 token 的隐藏向量进入注意力层，每个 token 是 512 维', analogy: '4 位同学拿着各自的 512 维简历走进面试室', color: '#64748b' },
  { name: 'Q/K/V 投影', shape: 'Q:[1,4,8,64] K/V:[1,4,2,64]', desc: '每个 token 通过三个线性层分别生成 Query、Key、Value 向量。Q 有 8 个头，K/V 只有 2 个头（GQA）', analogy: '每人填写三种卡片："我想问什么"(Q)、"我的标签"(K)、"我的内容"(V)', color: '#3b82f6' },
  { name: 'RoPE 旋转', shape: 'Q/K 形状不变', desc: '对 Q 和 K 施加旋转位置编码，让向量携带位置信息。V 不需要位置信息', analogy: '给 Q 和 K 卡片盖上"座位号"章，V 不需要', color: '#10b981' },
  { name: 'GQA repeat_kv', shape: 'K/V → [1,4,8,64]', desc: '2 个 KV 头各复制 4 次，扩展到 8 个，对齐 8 个 Q 头', analogy: '2 份 KV 卡片各复印 4 份，让 8 个 Q 面试官每人都有一份', color: '#f59e0b' },
  { name: 'Q·K^T 打分', shape: '[1, 8, 4, 4]', desc: '每个 Q 头与对应的 K 做点积，得到 4×4 的注意力原始分数矩阵', analogy: '每个面试官给所有同学打"相关度"分数', color: '#8b5cf6' },
  { name: '因果掩码', shape: '[1, 8, 4, 4]', desc: '将上三角（未来 token 位置）设为 -∞，确保自回归：每个 token 只能看到自身和之前的 token', analogy: '用挡板遮住后面的同学，只能看到前面的', color: '#ef4444' },
  { name: 'Softmax 归一化', shape: '[1, 8, 4, 4]', desc: '对每行做 softmax，将原始分数转化为概率分布（每行和为 1）', analogy: '把分数变成"关注度百分比"，总和 100%', color: '#06b6d4' },
  { name: '加权求和 V', shape: '[1, 8, 4, 64]', desc: '用注意力权重对 V 加权求和，每个 token 获得上下文融合的信息', analogy: '按关注度百分比，从每人的内容卡片中提取信息', color: '#f97316' },
  { name: '拼接 + O_proj', shape: '[1, 4, 512]', desc: '将 8 个头的 64 维结果拼接为 512 维，再通过输出投影回原始维度', analogy: '8 个面试官的笔记合并成一份总结报告', color: '#818cf8' },
];

export default function AttentionSection() {
  const { isDark } = useTheme();
  const [temp, setTemp] = useState(1);

  // Attention step animation state
  const [attnStep, setAttnStep] = useState(-1);
  const [attnPlaying, setAttnPlaying] = useState(false);
  const attnTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    return () => {
      if (attnTimerRef.current) clearInterval(attnTimerRef.current);
    };
  }, []);

  const stopAttn = useCallback(() => {
    if (attnTimerRef.current) { clearInterval(attnTimerRef.current); attnTimerRef.current = null; }
    setAttnPlaying(false);
  }, []);

  const playAttn = useCallback(() => {
    if (attnTimerRef.current) { stopAttn(); return; }
    setAttnPlaying(true);
    attnTimerRef.current = setInterval(() => {
      setAttnStep(prev => {
        if (prev < ATTN_STEPS.length - 1) return prev + 1;
        stopAttn();
        return prev;
      });
    }, 1200);
  }, [stopAttn]);

  const resetAttn = useCallback(() => { stopAttn(); setAttnStep(-1); }, [stopAttn]);

  const tokens = ['我', '喜', '欢', '你'];

  const attnFlowSvg = useMemo(() => {
    const fg = isDark ? '#e2e8f0' : '#1a1a2e';
    const fg2 = isDark ? '#94a3b8' : '#555';

    const boxes = ATTN_STEPS.map((s, i) => ({
      y: 10 + i * 46,
      h: 34,
      label: `${i}. ${s.name}`,
      color: s.color,
    }));

    let html = `<defs><marker id="arrAttn" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0 L10,5 L0,10 Z" fill="${fg2}"/></marker></defs>`;

    boxes.forEach((b, i) => {
      const isActive = i === attnStep;
      const isDone = i < attnStep;
      const opacity = isActive ? 0.35 : isDone ? 0.2 : 0.06;
      const strokeW = isActive ? 2.5 : 1.5;
      const dash = !isActive && !isDone ? 'stroke-dasharray="4,2"' : '';

      html += `<rect x="20" y="${b.y}" width="260" height="${b.h}" rx="6" fill="${b.color}" opacity="${opacity}" stroke="${b.color}" stroke-width="${strokeW}" ${dash} style="cursor:pointer" data-astep="${i}"/>`;
      html += `<text x="150" y="${b.y + b.h / 2 + 5}" text-anchor="middle" fill="${isActive ? fg : fg2}" font-size="${isActive ? 12 : 10.5}" ${isActive ? 'font-weight="bold"' : ''} style="pointer-events:none">${b.label}</text>`;

      if (isActive) {
        html += `<circle cx="10" cy="${b.y + b.h / 2}" r="5" fill="${b.color}"><animate attributeName="r" values="4;7;4" dur="1s" repeatCount="indefinite"/></circle>`;
      }
      if (isDone) {
        html += `<text x="288" y="${b.y + b.h / 2 + 4}" fill="${b.color}" font-size="12">✓</text>`;
      }
      if (i < boxes.length - 1) {
        html += `<line x1="150" y1="${b.y + b.h}" x2="150" y2="${boxes[i + 1].y}" stroke="${fg2}" stroke-width="1" marker-end="url(#arrAttn)" opacity="0.4"/>`;
      }
    });

    return html;
  }, [isDark, attnStep]);

  // Attention matrix mini-visualization for steps 4-6
  const attnMatrixSvg = useMemo(() => {
    if (attnStep < 4 || attnStep > 6) return '';
    const fg = isDark ? '#e2e8f0' : '#1a1a2e';
    const fg2 = isDark ? '#94a3b8' : '#555';
    const tokens = ['我', '喜', '欢', '你'];
    const cellSize = 36;
    const pad = 30;
    let html = '';
    html += `<text x="${pad + cellSize * 2}" y="14" text-anchor="middle" fill="${fg}" font-size="10">4×4 注意力矩阵 (模拟)</text>`;
    for (let j = 0; j < 4; j++) {
      html += `<text x="${pad + j * cellSize + cellSize / 2}" y="${pad - 4}" text-anchor="middle" fill="${fg2}" font-size="9">${tokens[j]}</text>`;
      html += `<text x="${pad - 6}" y="${pad + j * cellSize + cellSize / 2 + 4}" text-anchor="end" fill="${fg2}" font-size="9">${tokens[j]}</text>`;
    }
    const rawScores = [[1.2, 0.3, -0.5, 0.1], [0.8, 1.5, 0.2, -0.3], [0.4, 0.9, 1.8, 0.6], [0.3, 0.5, 1.1, 2.0]];
    for (let i = 0; i < 4; i++) {
      for (let j = 0; j < 4; j++) {
        const x = pad + j * cellSize;
        const y = pad + i * cellSize;
        if (attnStep === 4) {
          // Raw scores
          const val = j <= i ? rawScores[i][j] : 0;
          const intensity = j <= i ? Math.max(0, Math.min(1, (val + 1) / 3)) : 0.1;
          const color = isDark ? `rgba(129,140,248,${intensity})` : `rgba(79,70,229,${intensity})`;
          html += `<rect x="${x}" y="${y}" width="${cellSize - 2}" height="${cellSize - 2}" rx="3" fill="${color}"/>`;
          html += `<text x="${x + cellSize / 2 - 1}" y="${y + cellSize / 2 + 3}" text-anchor="middle" fill="${isDark ? '#e2e8f0' : '#333'}" font-size="9">${j <= i ? val.toFixed(1) : '—'}</text>`;
        } else if (attnStep === 5) {
          // After causal mask
          const val = j <= i ? rawScores[i][j] : -Infinity;
          const masked = j > i;
          const bgColor = masked ? (isDark ? '#334155' : '#e5e7eb') : (isDark ? `rgba(129,140,248,${Math.max(0.1, (rawScores[i][j] + 1) / 3)})` : `rgba(79,70,229,${Math.max(0.1, (rawScores[i][j] + 1) / 3)})`);
          html += `<rect x="${x}" y="${y}" width="${cellSize - 2}" height="${cellSize - 2}" rx="3" fill="${bgColor}"/>`;
          html += `<text x="${x + cellSize / 2 - 1}" y="${y + cellSize / 2 + 3}" text-anchor="middle" fill="${masked ? (isDark ? '#ef4444' : '#dc2626') : (isDark ? '#e2e8f0' : '#333')}" font-size="${masked ? 10 : 9}" font-weight="${masked ? 'bold' : 'normal'}">${masked ? '-∞' : val.toFixed(1)}</text>`;
        } else if (attnStep === 6) {
          // After softmax
          const masked = j > i;
          // Simple softmax simulation
          const rowVals = rawScores[i].slice(0, i + 1);
          const maxV = Math.max(...rowVals);
          const exps = rowVals.map(v => Math.exp(v - maxV));
          const sumExp = exps.reduce((a, b) => a + b, 0);
          const probs = exps.map(e => e / sumExp);
          const prob = masked ? 0 : probs[j];
          const intensity = prob;
          const bgColor = masked ? (isDark ? '#1e293b' : '#f0f0f0') : (isDark ? `rgba(6,182,212,${Math.max(0.1, intensity)})` : `rgba(6,182,212,${Math.max(0.1, intensity)})`);
          html += `<rect x="${x}" y="${y}" width="${cellSize - 2}" height="${cellSize - 2}" rx="3" fill="${bgColor}"/>`;
          html += `<text x="${x + cellSize / 2 - 1}" y="${y + cellSize / 2 + 3}" text-anchor="middle" fill="${masked ? (isDark ? '#475569' : '#ccc') : (isDark ? '#e2e8f0' : '#333')}" font-size="9">${masked ? '0' : prob.toFixed(2)}</text>`;
        }
      }
    }
    return html;
  }, [isDark, attnStep]);

  // QKV SVG
  const qkvSvg = useMemo(() => {
    const fg = isDark ? '#e2e8f0' : '#1a1a2e';
    const fg2 = isDark ? '#94a3b8' : '#555';
    const accent = isDark ? '#818cf8' : '#4f46e5';
    const green = isDark ? '#34d399' : '#10b981';
    const orange = isDark ? '#fbbf24' : '#f59e0b';
    const blue = isDark ? '#60a5fa' : '#3b82f6';
    return `
      <rect x="10" y="20" width="120" height="40" rx="6" fill="none" stroke="${fg2}" stroke-width="1.5"/>
      <text x="70" y="45" text-anchor="middle" fill="${fg}" font-size="12">Input [B,S,512]</text>
      <line x1="130" y1="40" x2="180" y2="40" stroke="${fg2}" stroke-width="1" marker-end="url(#arrowGray)"/>
      <defs><marker id="arrowGray" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0 L10,5 L0,10 Z" fill="${fg2}"/></marker></defs>
      <rect x="185" y="5" width="100" height="30" rx="5" fill="${blue}" opacity="0.15" stroke="${blue}" stroke-width="1.5"/>
      <text x="235" y="24" text-anchor="middle" fill="${blue}" font-size="11" font-weight="bold">Q Proj</text>
      <text x="235" y="52" text-anchor="middle" fill="${fg2}" font-size="10">512→512</text>
      <rect x="185" y="62" width="100" height="30" rx="5" fill="${green}" opacity="0.15" stroke="${green}" stroke-width="1.5"/>
      <text x="235" y="81" text-anchor="middle" fill="${green}" font-size="11" font-weight="bold">K Proj</text>
      <text x="235" y="108" text-anchor="middle" fill="${fg2}" font-size="10">512→128</text>
      <rect x="185" y="118" width="100" height="30" rx="5" fill="${orange}" opacity="0.15" stroke="${orange}" stroke-width="1.5"/>
      <text x="235" y="137" text-anchor="middle" fill="${orange}" font-size="11" font-weight="bold">V Proj</text>
      <text x="235" y="163" text-anchor="middle" fill="${fg2}" font-size="10">512→128</text>
      <line x1="285" y1="20" x2="340" y2="20" stroke="${blue}" stroke-width="1"/>
      <line x1="285" y1="77" x2="340" y2="77" stroke="${green}" stroke-width="1"/>
      <line x1="285" y1="133" x2="340" y2="133" stroke="${orange}" stroke-width="1"/>
      <text x="380" y="12" text-anchor="middle" fill="${fg2}" font-size="9">8 个 Q 头</text>
      ${[0, 1, 2, 3, 4, 5, 6, 7].map(i => `<rect x="${345 + i * 22}" y="16" width="18" height="14" rx="2" fill="${blue}" opacity="${0.4 + i * 0.07}"/>`).join('')}
      <text x="380" y="69" text-anchor="middle" fill="${fg2}" font-size="9">2 个 KV 头</text>
      <rect x="345" y="73" width="40" height="14" rx="2" fill="${green}" opacity="0.6"/>
      <rect x="390" y="73" width="40" height="14" rx="2" fill="${green}" opacity="0.8"/>
      <rect x="345" y="129" width="40" height="14" rx="2" fill="${orange}" opacity="0.6"/>
      <rect x="390" y="129" width="40" height="14" rx="2" fill="${orange}" opacity="0.8"/>
      <text x="530" y="12" text-anchor="middle" fill="${fg}" font-size="11" font-weight="bold">GQA 分组</text>
      <rect x="470" y="20" width="120" height="50" rx="6" fill="${green}" opacity="0.08" stroke="${green}" stroke-width="1" stroke-dasharray="4,2"/>
      <text x="530" y="36" text-anchor="middle" fill="${fg2}" font-size="9">KV Head 0 → Q Head 0,1,2,3</text>
      ${[0, 1, 2, 3].map(i => `<rect x="${485 + i * 26}" y="42" width="22" height="12" rx="2" fill="${blue}" opacity="0.7"/>`).join('')}
      <text x="530" y="64" text-anchor="middle" fill="${green}" font-size="9">repeat_kv ×4</text>
      <rect x="470" y="78" width="120" height="50" rx="6" fill="${orange}" opacity="0.08" stroke="${orange}" stroke-width="1" stroke-dasharray="4,2"/>
      <text x="530" y="94" text-anchor="middle" fill="${fg2}" font-size="9">KV Head 1 → Q Head 4,5,6,7</text>
      ${[0, 1, 2, 3].map(i => `<rect x="${485 + i * 26}" y="100" width="22" height="12" rx="2" fill="${blue}" opacity="0.7"/>`).join('')}
      <text x="530" y="122" text-anchor="middle" fill="${orange}" font-size="9">repeat_kv ×4</text>
      <rect x="470" y="145" width="120" height="30" rx="6" fill="none" stroke="${accent}" stroke-width="1.5"/>
      <text x="530" y="164" text-anchor="middle" fill="${accent}" font-size="11">scores @ V → O Proj</text>
      <text x="300" y="200" text-anchor="middle" fill="${fg}" font-size="12">Attention(Q,K,V) = softmax(QK<tspan baseline-shift="super" font-size="9">T</tspan>/√d<tspan baseline-shift="sub" font-size="9">k</tspan> + mask) × V</text>
      <text x="300" y="220" text-anchor="middle" fill="${fg2}" font-size="10">d_k = head_dim = ${MM.head_dim}，√d_k = ${Math.sqrt(MM.head_dim).toFixed(1)}</text>
      <text x="300" y="245" text-anchor="middle" fill="${fg2}" font-size="10">GQA: ${MM.num_heads} Q heads 共享 ${MM.num_kv_heads} KV heads → 减少 KV 缓存 ${MM.num_heads / MM.num_kv_heads}×</text>
    `;
  }, [isDark]);

  // Attention heatmap
  const canvasRef = useCanvas(
    (ctx, w, h) => {
      const n = tokens.length;
      const rng = mulberry32(123);
      const raw: number[][] = [];
      for (let i = 0; i < n; i++) {
        raw[i] = [];
        for (let j = 0; j < n; j++) {
          raw[i][j] = (rng() - 0.3) * 3;
        }
      }
      const attn: number[][] = [];
      for (let i = 0; i < n; i++) {
        const row: number[] = [];
        for (let j = 0; j < n; j++) {
          row.push(j <= i ? raw[i][j] : -1e9);
        }
        attn.push(softmax(row, temp));
      }

      const pad = 50;
      const cellSize = (w - pad - 10) / n;
      ctx.fillStyle = isDark ? '#1e293b' : '#f8f8f8';
      ctx.fillRect(0, 0, w, h);

      ctx.font = '13px sans-serif';
      ctx.fillStyle = isDark ? '#e2e8f0' : '#1a1a2e';
      ctx.textAlign = 'center';
      for (let j = 0; j < n; j++) {
        ctx.fillText(tokens[j], pad + cellSize * j + cellSize / 2, pad - 10);
        ctx.fillText(tokens[j], pad - 20, pad + cellSize * j + cellSize / 2 + 5);
      }
      ctx.font = '10px sans-serif';
      ctx.fillStyle = isDark ? '#94a3b8' : '#888';
      ctx.fillText('Key →', pad + (cellSize * n) / 2, 15);
      ctx.save();
      ctx.translate(10, pad + (cellSize * n) / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.fillText('Query →', 0, 0);
      ctx.restore();

      for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
          const v = attn[i][j];
          const x = pad + j * cellSize;
          const y = pad + i * cellSize;
          if (j > i) {
            ctx.fillStyle = isDark ? '#334155' : '#e5e7eb';
            ctx.fillRect(x, y, cellSize - 1, cellSize - 1);
            ctx.fillStyle = isDark ? '#64748b' : '#aaa';
            ctx.font = '10px sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText('-∞', x + cellSize / 2, y + cellSize / 2 + 3);
          } else {
            const intensity = Math.min(v * 1.2, 1);
            const r = isDark ? Math.round(129 + 126 * (1 - intensity)) : Math.round(79 * intensity);
            const g = isDark ? Math.round(140 + 115 * (1 - intensity)) : Math.round(70 * intensity);
            const b = isDark ? 248 : Math.round(229 * intensity);
            ctx.fillStyle = isDark ? `rgb(${r},${g},${b})` : `rgba(79,70,229,${intensity})`;
            ctx.fillRect(x, y, cellSize - 1, cellSize - 1);
            ctx.fillStyle = intensity > 0.5 ? '#fff' : isDark ? '#e2e8f0' : '#333';
            ctx.font = '11px monospace';
            ctx.textAlign = 'center';
            ctx.fillText(v.toFixed(2), x + cellSize / 2, y + cellSize / 2 + 4);
          }
        }
      }
    },
    [isDark, temp],
    300,
    300,
  );

  // KV Cache SVG
  const kvCacheSvg = useMemo(() => {
    const fg = isDark ? '#e2e8f0' : '#1a1a2e';
    const fg2 = isDark ? '#94a3b8' : '#555';
    const green = isDark ? '#34d399' : '#10b981';
    const blue = isDark ? '#60a5fa' : '#3b82f6';
    const orange = isDark ? '#fbbf24' : '#f59e0b';
    return `
      <defs><marker id="arrKV" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0 L10,5 L0,10 Z" fill="${fg2}"/></marker></defs>
      <text x="20" y="20" fill="${fg}" font-size="12" font-weight="bold">推理阶段 KV Cache</text>
      <rect x="20" y="35" width="200" height="40" rx="6" fill="${green}" opacity="0.12" stroke="${green}" stroke-width="1.5" stroke-dasharray="4,2"/>
      <text x="120" y="50" text-anchor="middle" fill="${green}" font-size="10">Cached K: [B, t-1, 2, 64]</text>
      <text x="120" y="65" text-anchor="middle" fill="${green}" font-size="10">Cached V: [B, t-1, 2, 64]</text>
      <rect x="240" y="35" width="100" height="40" rx="6" fill="${blue}" opacity="0.15" stroke="${blue}" stroke-width="1.5"/>
      <text x="290" y="50" text-anchor="middle" fill="${blue}" font-size="10">New Q: [B,1,8,64]</text>
      <text x="290" y="65" text-anchor="middle" fill="${blue}" font-size="10">New K,V: [B,1,2,64]</text>
      <line x1="220" y1="80" x2="260" y2="80" stroke="${fg2}" stroke-width="1" marker-end="url(#arrKV)"/>
      <text x="240" y="95" text-anchor="middle" fill="${fg2}" font-size="9">concat</text>
      <rect x="380" y="35" width="180" height="40" rx="6" fill="${orange}" opacity="0.12" stroke="${orange}" stroke-width="1.5"/>
      <text x="470" y="50" text-anchor="middle" fill="${orange}" font-size="10">K: [B, t, 2, 64] (更新缓存)</text>
      <text x="470" y="65" text-anchor="middle" fill="${orange}" font-size="10">Attn: Q×K^T → [B,8,1,t]</text>
      <line x1="340" y1="55" x2="380" y2="55" stroke="${fg2}" stroke-width="1" marker-end="url(#arrKV)"/>
      <text x="350" y="110" text-anchor="middle" fill="${fg2}" font-size="11">每步只计算 1 个新 token 的 Q/K/V，拼接到缓存 → O(1) 生成而非 O(t)</text>
      <text x="350" y="128" text-anchor="middle" fill="${fg2}" font-size="10">GQA 缓存: 2 个 KV 头 vs MHA 的 8 个 → 节省 4× 显存</text>
    `;
  }, [isDark]);

  return (
    <>
      <h2>3. 自注意力 (Self-Attention)</h2>
      <p className="desc">
        注意力机制让序列中的每个 token 能"看到"其他 token 并动态计算相关度权重。
        MiniMind 使用 GQA（分组查询注意力）：<code>num_attention_heads=8</code> 个 Q 头共享 <code>num_kv_heads=2</code> 个 KV 头，
        每个头的维度 <code>head_dim=64</code>。
        <br/>
        <small style={{ color: 'var(--fg2)' }}>
          关联源码：<code>model/model_minimind.py:150</code> <code>class Attention</code> | <code>:169</code> <code>def forward</code>
        </small>
      </p>

      <Card title="注意力计算逐步动画">
        <p style={{ marginBottom: 10, fontSize: '0.9rem', color: 'var(--fg2)' }}>
          从输入到输出，完整演示注意力机制的 9 个计算步骤。点击「下一步」逐步推进，或点击「自动播放」自动演示。
        </p>
        <div style={{ display: 'flex', gap: 8, marginBottom: 10, flexWrap: 'wrap' }}>
          <button className="btn" onClick={() => setAttnStep(prev => Math.max(prev - 1, 0))}>◀ 上一步</button>
          <button className="btn primary" onClick={() => setAttnStep(prev => Math.min(prev + 1, ATTN_STEPS.length - 1))}>下一步 ▶</button>
          <button className="btn" onClick={playAttn}>{attnPlaying ? '⏸ 暂停' : '▶ 自动播放'}</button>
          <button className="btn" onClick={resetAttn}>重置</button>
        </div>
        <div className="step-indicator">
          {ATTN_STEPS.map((s, i) => (
            <div
              key={i}
              className={`step-dot${i === attnStep ? ' active' : ''}${i < attnStep ? ' done' : ''}`}
              onClick={() => setAttnStep(i)}
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
              height={430}
              viewBox="0 0 310 430"
              onClick={(e) => {
                const t = (e.target as SVGElement).closest('rect[data-astep]');
                if (t) setAttnStep(parseInt(t.getAttribute('data-astep')!));
              }}
              dangerouslySetInnerHTML={{ __html: attnFlowSvg }}
            />
          </div>
          <div>
            {attnStep >= 0 ? (() => {
              const cur = ATTN_STEPS[attnStep];
              return (
                <>
                  <div style={{ fontSize: '1.1rem', fontWeight: 700, color: cur.color, marginBottom: 6 }}>
                    {attnStep}. {cur.name}
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
                  <div className="label">Shape</div>
                  <div className="shape-badge" style={{ marginBottom: 10, borderColor: cur.color, color: cur.color }}>{cur.shape}</div>
                  <div className="label">说明</div>
                  <p style={{ fontSize: '0.88rem', color: 'var(--fg2)', marginBottom: 10 }}>{cur.desc}</p>
                  {attnStep >= 4 && attnStep <= 6 && (
                    <div>
                      <div className="label" style={{ marginTop: 8 }}>{attnStep === 4 ? '原始分数矩阵' : attnStep === 5 ? '因果掩码后' : 'Softmax 后（概率）'}</div>
                      <svg width={180} height={180} viewBox="0 0 180 180" dangerouslySetInnerHTML={{ __html: attnMatrixSvg }} />
                    </div>
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

      <Card title="Q / K / V 计算与 GQA 分组">
        <p style={{ marginBottom: 10, fontSize: '0.9rem', color: 'var(--fg2)' }}>
          注意力计算分为三步：(1) 通过线性投影生成 Query（查询）、Key（键）、Value（值）三组向量；
          (2) Q 和 K 的点积计算注意力分数，衡量 token 间的相关性；(3) 用分数加权求和 V，得到融合了上下文信息的输出。
        </p>
        <p style={{ marginBottom: 10, fontSize: '0.9rem', color: 'var(--fg2)' }}>
          GQA（分组查询注意力）是 MHA 的高效变体：8 个 Q 头共享 2 个 KV 头（每 4 个 Q 头复用 1 个 KV 头），
          KV 投影从 512→128（而非 512→512），推理时 KV Cache 显存减少 4 倍，效果接近完整 MHA。
          下图展示了从输入到分组的完整流程：
        </p>
        <svg width="100%" height={260} viewBox="0 0 700 260" dangerouslySetInnerHTML={{ __html: qkvSvg }} />
        <SourcePanel
          title="对照源码：model/model_minimind.py:150-213 (Attention)"
          code={`class Attention(nn.Module):
    def __init__(self, args):
        self.n_local_heads = args.num_attention_heads      # Q 头数 = 8
        self.n_local_kv_heads = args.num_key_value_heads   # KV 头数 = 2（GQA 核心参数）
        self.n_rep = self.n_local_heads // self.n_local_kv_heads  # 每个 KV 头被复用的次数 = 4
        self.head_dim = args.hidden_size // args.num_attention_heads  # 每头维度 = 512/8 = 64
        # Q 投影：512 → 8*64 = 512（Q 需要 8 个头，所以输出维度不变）
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim)     # 512→512
        # K/V 投影：512 → 2*64 = 128（只有 2 个 KV 头，参数量仅为 Q 的 1/4）
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim)  # 512→128
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim)  # 512→128

    def forward(self, x, position_embeddings, ...):
        # 线性投影后 reshape 为多头格式
        xq = self.q_proj(x).view(bsz, seq_len, 8, 64)   # 8 个 Q 头
        xk = self.k_proj(x).view(bsz, seq_len, 2, 64)   # 2 个 K 头
        xv = self.v_proj(x).view(bsz, seq_len, 2, 64)   # 2 个 V 头
        # 对 Q 和 K 施加 RoPE 旋转位置编码（V 不需要位置信息）
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)
        # GQA 关键步骤：将 2 个 KV 头复制 4 次，扩展到 8 个
        # 这样每个 Q 头都有对应的 KV，但实际参数只有 2 个头的量
        xk = repeat_kv(xk, n_rep=4)  # [B,S,2,64] → [B,S,8,64]
        xv = repeat_kv(xv, n_rep=4)
        # 注意力计算：QK^T / √d_k + 因果掩码 → softmax → 加权 V
        scores = (xq @ xk.T) / sqrt(64) + causal_mask
        output = softmax(scores) @ xv  # [B,S,8,64] → concat → O_proj → [B,S,512]`}
        />
      </Card>

      <Card title="注意力热力图">
        <p style={{ marginBottom: 10, fontSize: '0.9rem', color: 'var(--fg2)' }}>
          热力图展示 4 个 token 之间的注意力权重矩阵。每个格子的颜色深浅代表 Query token（行）对 Key token（列）的关注程度。
          因果掩码将上三角设为 -∞，确保 token 只能看到自身和之前的 token（自回归约束）。
          拖动温度滑块可以观察分布变化：低温使权重集中在少数 token（更"确定"），高温使权重更均匀（更"随机"）。
        </p>
        <div className="viz-grid">
          <div>
            <canvas ref={canvasRef} />
          </div>
          <div>
            <div style={{ marginBottom: 8 }}>
              <div className="label">温度参数 (Temperature)</div>
              <input
                type="range"
                min="0.1"
                max="3"
                step="0.1"
                value={temp}
                onChange={e => setTemp(parseFloat(e.target.value))}
              />
              <span className="value">{temp.toFixed(1)}</span>
            </div>
            <div style={{ marginBottom: 8 }}>
              <div className="label">tokens</div>
              <div style={{ display: 'flex', gap: 4 }}>
                {tokens.map((t, i) => (
                  <span key={i} className="token-box" style={{ background: tokenColor(i), color: '#fff' }}>{t}</span>
                ))}
              </div>
            </div>
            <div className="label">说明</div>
            <p style={{ fontSize: '0.85rem', color: 'var(--fg2)' }}>
              每行表示一个 token 对所有可见 token 的注意力权重（softmax 后）。因果掩码确保 token 只能看到自身和之前的 token。降低温度使分布更尖锐，升高则更平滑。
            </p>
          </div>
        </div>
      </Card>

      <Card title="KV Cache 示意">
        <p style={{ marginBottom: 10, fontSize: '0.9rem', color: 'var(--fg2)' }}>
          推理（生成）阶段每步只产生 1 个新 token。如果每次都重新计算所有 token 的 K/V，复杂度为 O(t²)。
          KV Cache 的思路：将之前所有 token 的 K 和 V 缓存起来，新 token 只需计算自身的 Q/K/V，
          然后将新 K/V 拼接到缓存中，用完整的 K/V 计算注意力，复杂度降为 O(t)。
          GQA 进一步减少缓存量：只需缓存 2 个 KV 头（而非 8 个），节省 4 倍显存。
        </p>
        <svg width="100%" height={140} viewBox="0 0 700 140" dangerouslySetInnerHTML={{ __html: kvCacheSvg }} />
      </Card>
    </>
  );
}
