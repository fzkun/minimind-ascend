import { useState, useRef, useCallback, useMemo, useEffect } from 'react';
import Card from '../components/Card';
import SourcePanel from '../components/SourcePanel';
import { useTheme } from '../context/ThemeContext';
import { useCanvas } from '../hooks/useCanvas';
import { MM, COLORS } from '../constants';
import { softmax, silu, mulberry32 } from '../utils';

interface RoutingResult {
  gates: number[][];
  assignments: { experts: number[]; weights: number[]; allProbs: number[] }[];
}

const N_TOKENS = 8;
const TOKEN_LABELS = ['我', '是', '一', '个', '语', '言', '模', '型'];
const EXPERT_LABELS = ['专家0', '专家1', '专家2', '专家3'];

function computeRouting(): RoutingResult {
  const rng = mulberry32(Date.now() % 10000);
  const gates: number[][] = [];
  const assignments: RoutingResult['assignments'] = [];
  for (let t = 0; t < N_TOKENS; t++) {
    const logits = Array.from({ length: MM.n_routed_experts }, () => (rng() - 0.3) * 4);
    const probs = softmax(logits);
    const sorted = probs.map((p, i) => ({ p, i })).sort((a, b) => b.p - a.p);
    const topIdx = sorted.slice(0, MM.num_experts_per_tok).map(s => s.i);
    const topW = sorted.slice(0, MM.num_experts_per_tok).map(s => s.p);
    const wSum = topW.reduce((a, b) => a + b, 0);
    assignments.push({ experts: topIdx, weights: topW.map(w => w / wSum), allProbs: probs });
    gates.push(probs);
  }
  return { gates, assignments };
}

const SWIGLU_STEPS = [
  { name: '输入 x', shape: '[512]', desc: '一个 token 的 512 维隐藏向量进入前馈网络', analogy: '一份 512 格的信息表进入加工车间', color: '#64748b' },
  { name: 'gate_proj(x)', shape: '[1408]', desc: '门控分支：通过线性层将 512 维扩展到 1408 维，生成门控信号', analogy: '门卫检查：把 512 格扩展成 1408 格的"通行证"', color: '#3b82f6' },
  { name: 'SiLU(gate)', shape: '[1408]', desc: 'SiLU 激活函数：SiLU(x)=x·σ(x)，平滑地决定每个特征的放行程度（0~∞）', analogy: '门卫盖章：每格标注"放行/屏蔽"程度', color: '#f59e0b' },
  { name: 'up_proj(x)', shape: '[1408]', desc: '候选分支：另一个线性层也将 512 维扩展到 1408 维，生成候选特征', analogy: '加工员独立制作一份 1408 格的"候选产品"', color: '#10b981' },
  { name: 'SiLU(gate) ⊙ up', shape: '[1408]', desc: '逐元素相乘：门控值 × 候选值，选择性地保留有用的特征', analogy: '门卫和加工员对账：通行证 × 候选产品 = 最终放行的信息', color: '#8b5cf6' },
  { name: 'down_proj → output', shape: '[512]', desc: '通过线性层将 1408 维压缩回 512 维，输出与输入同维度', analogy: '把 1408 格精选信息压缩回 512 格，送出车间', color: '#818cf8' },
];

export default function FFNMoESection() {
  const { isDark } = useTheme();
  const [routingResult, setRoutingResult] = useState<RoutingResult | null>(null);
  const [moeProgress, setMoeProgress] = useState(0);
  const animIdRef = useRef<number | null>(null);

  useEffect(() => {
    return () => {
      if (animIdRef.current) cancelAnimationFrame(animIdRef.current);
    };
  }, []);

  const handleMoeRun = useCallback(() => {
    if (animIdRef.current) {
      cancelAnimationFrame(animIdRef.current);
      animIdRef.current = null;
    }
    const result = computeRouting();
    setRoutingResult(result);
    let start: number | null = null;
    const duration = 1500;
    const animate = (ts: number) => {
      if (!start) start = ts;
      const progress = Math.min(1, (ts - start) / duration);
      setMoeProgress(progress);
      if (progress < 1) animIdRef.current = requestAnimationFrame(animate);
      else animIdRef.current = null;
    };
    animIdRef.current = requestAnimationFrame(animate);
  }, []);

  const handleMoeReset = useCallback(() => {
    if (animIdRef.current) {
      cancelAnimationFrame(animIdRef.current);
      animIdRef.current = null;
    }
    setRoutingResult(null);
    setMoeProgress(0);
  }, []);

  // SwiGLU step animation state
  const [ffnStep, setFfnStep] = useState(-1);
  const [ffnPlaying, setFfnPlaying] = useState(false);
  const ffnTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    return () => {
      if (ffnTimerRef.current) clearInterval(ffnTimerRef.current);
    };
  }, []);

  const stopFfn = useCallback(() => {
    if (ffnTimerRef.current) { clearInterval(ffnTimerRef.current); ffnTimerRef.current = null; }
    setFfnPlaying(false);
  }, []);

  const playFfn = useCallback(() => {
    if (ffnTimerRef.current) { stopFfn(); return; }
    setFfnPlaying(true);
    ffnTimerRef.current = setInterval(() => {
      setFfnStep(prev => {
        if (prev < SWIGLU_STEPS.length - 1) return prev + 1;
        stopFfn();
        return prev;
      });
    }, 1200);
  }, [stopFfn]);

  const resetFfn = useCallback(() => { stopFfn(); setFfnStep(-1); }, [stopFfn]);

  const ffnFlowSvg = useMemo(() => {
    const fg = isDark ? '#e2e8f0' : '#1a1a2e';
    const fg2 = isDark ? '#94a3b8' : '#555';

    const boxes = SWIGLU_STEPS.map((s, i) => ({
      y: 10 + i * 52,
      h: 38,
      label: `${i}. ${s.name}`,
      color: s.color,
    }));

    let html = `<defs><marker id="arrFfn" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0 L10,5 L0,10 Z" fill="${fg2}"/></marker></defs>`;

    boxes.forEach((b, i) => {
      const isActive = i === ffnStep;
      const isDone = i < ffnStep;
      const opacity = isActive ? 0.35 : isDone ? 0.2 : 0.06;
      const strokeW = isActive ? 2.5 : 1.5;
      const dash = !isActive && !isDone ? 'stroke-dasharray="4,2"' : '';

      html += `<rect x="20" y="${b.y}" width="260" height="${b.h}" rx="6" fill="${b.color}" opacity="${opacity}" stroke="${b.color}" stroke-width="${strokeW}" ${dash} style="cursor:pointer" data-fstep="${i}"/>`;
      html += `<text x="150" y="${b.y + b.h / 2 + 5}" text-anchor="middle" fill="${isActive ? fg : fg2}" font-size="${isActive ? 12 : 10.5}" ${isActive ? 'font-weight="bold"' : ''} style="pointer-events:none">${b.label}</text>`;

      if (isActive) {
        html += `<circle cx="10" cy="${b.y + b.h / 2}" r="5" fill="${b.color}"><animate attributeName="r" values="4;7;4" dur="1s" repeatCount="indefinite"/></circle>`;
      }
      if (isDone) {
        html += `<text x="288" y="${b.y + b.h / 2 + 4}" fill="${b.color}" font-size="12">✓</text>`;
      }
      if (i < boxes.length - 1) {
        // Special: step 0 forks to step 1 and step 3, step 2 and step 3 merge into step 4
        if (i === 0) {
          html += `<line x1="120" y1="${b.y + b.h}" x2="120" y2="${boxes[1].y}" stroke="${fg2}" stroke-width="1" marker-end="url(#arrFfn)" opacity="0.4"/>`;
          html += `<line x1="180" y1="${b.y + b.h}" x2="180" y2="${boxes[1].y}" stroke="${fg2}" stroke-width="1" opacity="0.15" stroke-dasharray="3,2"/>`;
        } else if (i === 2) {
          html += `<line x1="150" y1="${b.y + b.h}" x2="150" y2="${boxes[i + 1].y}" stroke="${fg2}" stroke-width="1" opacity="0.15" stroke-dasharray="3,2"/>`;
        } else {
          html += `<line x1="150" y1="${b.y + b.h}" x2="150" y2="${boxes[i + 1].y}" stroke="${fg2}" stroke-width="1" marker-end="url(#arrFfn)" opacity="0.4"/>`;
        }
      }
      // Fork line from step 0 to step 3
      if (i === 0) {
        html += `<path d="M 180 ${b.y + b.h} L 270 ${b.y + b.h + 10} L 270 ${boxes[3].y - 5} L 180 ${boxes[3].y}" fill="none" stroke="${fg2}" stroke-width="1" marker-end="url(#arrFfn)" opacity="0.4" stroke-dasharray="4,2"/>`;
        html += `<text x="278" y="${(b.y + b.h + boxes[3].y) / 2 + 5}" fill="${fg2}" font-size="8" opacity="0.6">并行</text>`;
      }
      // Merge indicator at step 4
      if (i === 3) {
        html += `<path d="M 120 ${boxes[2].y + boxes[2].h} L 100 ${boxes[2].y + boxes[2].h + 8} L 100 ${boxes[4].y - 5} L 120 ${boxes[4].y}" fill="none" stroke="${fg2}" stroke-width="1" marker-end="url(#arrFfn)" opacity="0.4"/>`;
      }
    });

    return html;
  }, [isDark, ffnStep]);

  // Step detail visualization for FFN steps
  const ffnStepDetailSvg = useMemo(() => {
    if (ffnStep < 0) return '';
    const fg = isDark ? '#e2e8f0' : '#1a1a2e';
    const fg2 = isDark ? '#94a3b8' : '#555';
    const accent = isDark ? '#818cf8' : '#4f46e5';

    let html = '';
    const dims = [512, 1408, 1408, 1408, 1408, 512];
    const dim = dims[ffnStep];
    const maxDim = 1408;
    const barW = Math.max(40, (dim / maxDim) * 220);

    html += `<text x="5" y="14" fill="${fg}" font-size="9" font-weight="bold">维度变化</text>`;
    html += `<rect x="5" y="22" width="${barW}" height="18" rx="3" fill="${SWIGLU_STEPS[ffnStep].color}" opacity="0.4"/>`;
    html += `<text x="${10 + barW}" y="35" fill="${fg}" font-size="9">${dim} 维</text>`;

    if (ffnStep === 2) {
      // SiLU visualization
      html += `<text x="5" y="60" fill="${fg}" font-size="9" font-weight="bold">SiLU 激活示例（8 个采样值）</text>`;
      const sampleInputs = [-2, -1, -0.5, 0, 0.5, 1, 2, 3];
      const siluFn = (x: number) => x / (1 + Math.exp(-x));
      sampleInputs.forEach((x, i) => {
        const y = siluFn(x);
        const barH = Math.max(1, Math.abs(y) * 18);
        const barX = 10 + i * 30;
        const baseY = 100;
        html += `<rect x="${barX}" y="${y >= 0 ? baseY - barH : baseY}" width="22" height="${barH}" rx="2" fill="${y >= 0 ? '#10b981' : '#ef4444'}" opacity="0.6"/>`;
        html += `<text x="${barX + 11}" y="118" text-anchor="middle" fill="${fg2}" font-size="7">${x}</text>`;
        html += `<text x="${barX + 11}" y="${y >= 0 ? baseY - barH - 3 : baseY + barH + 9}" text-anchor="middle" fill="${fg}" font-size="7">${y.toFixed(1)}</text>`;
      });
      html += `<line x1="5" y1="100" x2="250" y2="100" stroke="${fg2}" stroke-width="0.5" stroke-dasharray="2,2"/>`;
    }

    if (ffnStep === 4) {
      // Element-wise multiply visualization
      html += `<text x="5" y="60" fill="${fg}" font-size="9" font-weight="bold">逐元素相乘（门控选择）</text>`;
      const gateVals = [0.8, 0.1, 0.9, 0.0, 0.6, 0.3];
      const upVals = [1.2, 0.5, -0.8, 1.5, 0.9, -0.3];
      gateVals.forEach((g, i) => {
        const result = g * upVals[i];
        const barX = 5 + i * 42;
        html += `<text x="${barX + 16}" y="76" text-anchor="middle" fill="#f59e0b" font-size="7">${g.toFixed(1)}</text>`;
        html += `<text x="${barX + 16}" y="86" text-anchor="middle" fill="${accent}" font-size="8">×</text>`;
        html += `<text x="${barX + 16}" y="96" text-anchor="middle" fill="#10b981" font-size="7">${upVals[i].toFixed(1)}</text>`;
        html += `<line x1="${barX + 2}" y1="99" x2="${barX + 30}" y2="99" stroke="${fg2}" stroke-width="0.5"/>`;
        html += `<text x="${barX + 16}" y="110" text-anchor="middle" fill="${fg}" font-size="7" font-weight="bold">${result.toFixed(2)}</text>`;
      });
    }

    return html;
  }, [isDark, ffnStep]);

  // SwiGLU SVG
  const swigluSvg = useMemo(() => {
    const fg = isDark ? '#e2e8f0' : '#1a1a2e';
    const fg2 = isDark ? '#94a3b8' : '#555';
    const accent = isDark ? '#818cf8' : '#4f46e5';
    const green = isDark ? '#34d399' : '#10b981';
    const orange = isDark ? '#fbbf24' : '#f59e0b';
    const blue = isDark ? '#60a5fa' : '#3b82f6';
    return `
      <defs><marker id="arrSG" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0 L10,5 L0,10 Z" fill="${fg2}"/></marker></defs>
      <rect x="10" y="75" width="80" height="40" rx="6" fill="none" stroke="${fg2}" stroke-width="1.5"/>
      <text x="50" y="99" text-anchor="middle" fill="${fg}" font-size="11">x [512]</text>
      <line x1="90" y1="95" x2="130" y2="55" stroke="${fg2}" stroke-width="1" marker-end="url(#arrSG)"/>
      <line x1="90" y1="95" x2="130" y2="135" stroke="${fg2}" stroke-width="1" marker-end="url(#arrSG)"/>
      <rect x="135" y="35" width="110" height="35" rx="6" fill="${blue}" opacity="0.15" stroke="${blue}" stroke-width="1.5"/>
      <text x="190" y="50" text-anchor="middle" fill="${blue}" font-size="10" font-weight="bold">gate_proj</text>
      <text x="190" y="63" text-anchor="middle" fill="${fg2}" font-size="9">512→1408</text>
      <rect x="135" y="120" width="110" height="35" rx="6" fill="${green}" opacity="0.15" stroke="${green}" stroke-width="1.5"/>
      <text x="190" y="135" text-anchor="middle" fill="${green}" font-size="10" font-weight="bold">up_proj</text>
      <text x="190" y="148" text-anchor="middle" fill="${fg2}" font-size="9">512→1408</text>
      <line x1="245" y1="52" x2="285" y2="52" stroke="${fg2}" stroke-width="1" marker-end="url(#arrSG)"/>
      <rect x="290" y="35" width="70" height="35" rx="6" fill="${orange}" opacity="0.15" stroke="${orange}" stroke-width="1.5"/>
      <text x="325" y="57" text-anchor="middle" fill="${orange}" font-size="11" font-weight="bold">SiLU</text>
      <line x1="360" y1="52" x2="410" y2="90" stroke="${fg2}" stroke-width="1" marker-end="url(#arrSG)"/>
      <line x1="245" y1="137" x2="410" y2="96" stroke="${fg2}" stroke-width="1" marker-end="url(#arrSG)"/>
      <circle cx="420" cy="93" r="14" fill="none" stroke="${accent}" stroke-width="2"/>
      <text x="420" y="98" text-anchor="middle" fill="${accent}" font-size="16" font-weight="bold">×</text>
      <text x="420" y="120" text-anchor="middle" fill="${fg2}" font-size="9">[1408]</text>
      <line x1="434" y1="93" x2="480" y2="93" stroke="${fg2}" stroke-width="1" marker-end="url(#arrSG)"/>
      <rect x="485" y="75" width="110" height="35" rx="6" fill="${accent}" opacity="0.15" stroke="${accent}" stroke-width="1.5"/>
      <text x="540" y="90" text-anchor="middle" fill="${accent}" font-size="10" font-weight="bold">down_proj</text>
      <text x="540" y="103" text-anchor="middle" fill="${fg2}" font-size="9">1408→512</text>
      <line x1="595" y1="93" x2="640" y2="93" stroke="${fg2}" stroke-width="1" marker-end="url(#arrSG)"/>
      <rect x="645" y="75" width="50" height="35" rx="6" fill="none" stroke="${fg2}" stroke-width="1.5"/>
      <text x="670" y="97" text-anchor="middle" fill="${fg}" font-size="11">out</text>
      <text x="350" y="185" text-anchor="middle" fill="${fg}" font-size="12">FFN(x) = down_proj( SiLU(gate_proj(x)) ⊙ up_proj(x) )</text>
    `;
  }, [isDark]);

  // SiLU canvas
  const siluCanvasRef = useCanvas(
    (ctx, w, h) => {
      ctx.fillStyle = isDark ? '#1e293b' : '#f8f8f8';
      ctx.fillRect(0, 0, w, h);
      const padL = 30, padR = 10, padT = 10, padB = 25;
      const plotW = w - padL - padR, plotH = h - padT - padB;
      const xRange = [-5, 5], yRange = [-1.5, 5];
      const toX = (v: number) => padL + ((v - xRange[0]) / (xRange[1] - xRange[0])) * plotW;
      const toY = (v: number) => padT + plotH - ((v - yRange[0]) / (yRange[1] - yRange[0])) * plotH;

      const fg2c = isDark ? '#475569' : '#ccc';
      ctx.strokeStyle = fg2c;
      ctx.lineWidth = 0.5;
      ctx.beginPath(); ctx.moveTo(toX(0), padT); ctx.lineTo(toX(0), padT + plotH); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(padL, toY(0)); ctx.lineTo(padL + plotW, toY(0)); ctx.stroke();

      const accent = isDark ? '#818cf8' : '#4f46e5';
      ctx.strokeStyle = accent;
      ctx.lineWidth = 2;
      ctx.beginPath();
      for (let px = 0; px <= plotW; px++) {
        const x = xRange[0] + (px / plotW) * (xRange[1] - xRange[0]);
        const y = silu(x);
        const sy = toY(y);
        if (px === 0) ctx.moveTo(padL + px, sy);
        else ctx.lineTo(padL + px, sy);
      }
      ctx.stroke();

      const red = isDark ? '#f87171' : '#ef4444';
      ctx.strokeStyle = red;
      ctx.lineWidth = 1.5;
      ctx.setLineDash([4, 3]);
      ctx.beginPath();
      for (let px = 0; px <= plotW; px++) {
        const x = xRange[0] + (px / plotW) * (xRange[1] - xRange[0]);
        const y = Math.max(0, x);
        const sy = toY(y);
        if (px === 0) ctx.moveTo(padL + px, sy);
        else ctx.lineTo(padL + px, sy);
      }
      ctx.stroke();
      ctx.setLineDash([]);

      ctx.fillStyle = accent;
      ctx.font = '10px sans-serif';
      ctx.textAlign = 'left';
      ctx.fillText('SiLU', padL + plotW - 30, toY(silu(4.5)) - 3);
      ctx.fillStyle = red;
      ctx.fillText('ReLU', padL + plotW - 32, toY(4.5) - 3);
    },
    [isDark],
    220,
    140,
  );

  // MoE canvas
  const moeCanvasRef = useCanvas(
    (ctx, w, h) => {
      const fg = isDark ? '#e2e8f0' : '#1a1a2e';
      const fg2 = isDark ? '#94a3b8' : '#888';

      ctx.fillStyle = isDark ? '#1e293b' : '#f8f8f8';
      ctx.fillRect(0, 0, w, h);

      const tokenY = 40, expertY = 220;
      const tokenStartX = 80, tokenGap = 70;
      const expertStartX = 120, expertGap = 120;
      const sharedX = 580;
      const progress = moeProgress;

      // Expert boxes
      for (let e = 0; e < MM.n_routed_experts; e++) {
        const ex = expertStartX + e * expertGap;
        ctx.fillStyle = isDark ? '#334155' : '#e8e8e8';
        ctx.strokeStyle = COLORS[e];
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.roundRect(ex - 35, expertY - 20, 70, 40, 8);
        ctx.fill();
        ctx.stroke();
        ctx.fillStyle = fg;
        ctx.font = '11px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(EXPERT_LABELS[e], ex, expertY + 5);
      }

      // Shared expert
      ctx.fillStyle = isDark ? '#334155' : '#e8e8e8';
      ctx.strokeStyle = isDark ? '#a78bfa' : '#7c3aed';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.roundRect(sharedX - 35, expertY - 20, 70, 40, 8);
      ctx.fill();
      ctx.stroke();
      ctx.fillStyle = fg;
      ctx.font = '11px sans-serif';
      ctx.fillText('共享专家', sharedX, expertY + 5);

      // Tokens and routing
      for (let t = 0; t < N_TOKENS; t++) {
        const tx = tokenStartX + t * tokenGap;

        if (routingResult && progress > 0) {
          const a = routingResult.assignments[t];
          for (let k = 0; k < MM.num_experts_per_tok; k++) {
            const eIdx = a.experts[k];
            const ex = expertStartX + eIdx * expertGap;
            const lineProgress = Math.min(1, progress * 2 - k * 0.3);
            if (lineProgress > 0) {
              ctx.globalAlpha = lineProgress * a.weights[k];
              ctx.strokeStyle = COLORS[eIdx];
              ctx.lineWidth = Math.max(1, a.weights[k] * 4);
              ctx.beginPath();
              ctx.moveTo(tx, tokenY + 18);
              ctx.lineTo(ex, expertY - 20);
              ctx.stroke();
            }
          }
          if (progress > 0.5) {
            ctx.globalAlpha = Math.min(1, (progress - 0.5) * 2) * 0.5;
            ctx.strokeStyle = isDark ? '#a78bfa' : '#7c3aed';
            ctx.lineWidth = 1.5;
            ctx.beginPath();
            ctx.moveTo(tx, tokenY + 18);
            ctx.lineTo(sharedX, expertY - 20);
            ctx.stroke();
          }
          ctx.globalAlpha = 1;
        }

        ctx.fillStyle = isDark ? '#475569' : '#ddd';
        ctx.strokeStyle = COLORS[t % COLORS.length];
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(tx, tokenY, 18, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
        ctx.fillStyle = fg;
        ctx.font = '13px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(TOKEN_LABELS[t], tx, tokenY + 5);

        if (routingResult && progress >= 1) {
          const a = routingResult.assignments[t];
          ctx.font = '9px monospace';
          ctx.fillStyle = fg2;
          ctx.fillText(`E${a.experts[0]}:${a.weights[0].toFixed(2)} E${a.experts[1]}:${a.weights[1].toFixed(2)}`, tx, tokenY + 33);
        }
      }

      ctx.fillStyle = fg;
      ctx.font = '12px sans-serif';
      ctx.textAlign = 'left';
      ctx.fillText('Token 序列', 10, 15);
      ctx.fillText('路由专家 (top-2)', 10, expertY - 35);
      ctx.fillText('+ 共享', sharedX - 30, expertY - 35);

      if (routingResult && progress >= 1) {
        ctx.font = '10px monospace';
        ctx.textAlign = 'center';
        for (let e = 0; e < MM.n_routed_experts; e++) {
          const ex = expertStartX + e * expertGap;
          const count = routingResult.assignments.filter(a => a.experts.includes(e)).length;
          ctx.fillStyle = fg2;
          ctx.fillText(`${count} tokens`, ex, expertY + 35);
        }
      }

      ctx.fillStyle = fg2;
      ctx.font = '10px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('Gate: softmax(W_gate · x) → top-2 选择', w / 2, h - 15);
    },
    [isDark, routingResult, moeProgress],
    680,
    320,
  );

  // Load balance data
  const loadBalanceData = useMemo(() => {
    if (!routingResult || moeProgress < 1) return null;
    const counts = Array(MM.n_routed_experts).fill(0);
    const avgProbs = Array(MM.n_routed_experts).fill(0);
    routingResult.assignments.forEach(a => {
      a.experts.forEach(e => counts[e]++);
      a.allProbs.forEach((p, e) => { avgProbs[e] += p; });
    });
    avgProbs.forEach((_, i) => { avgProbs[i] /= N_TOKENS; });
    const fi = counts.map(c => c / (N_TOKENS * MM.num_experts_per_tok));
    const auxLoss = MM.aux_loss_alpha * MM.n_routed_experts * fi.reduce((s, f, i) => s + f * avgProbs[i], 0);
    const maxCount = Math.max(...counts, 1);
    return { counts, maxCount, auxLoss };
  }, [routingResult, moeProgress]);

  return (
    <>
      <h2>5. 前馈网络 &amp; MoE</h2>
      <p className="desc">
        每个 Transformer Block 中，Attention 之后是前馈网络 (FFN)，对每个 token 独立做非线性变换。
        MiniMind 使用 SwiGLU 结构：<code>output = silu(x @ W_gate) * (x @ W_up) @ W_down</code>。
        可通过 <code>use_moe=True</code> 切换为 MoE（混合专家）架构——多个 Expert FFN 由 Router 动态选择 top-k 激活。
        <br/>
        <small style={{ color: 'var(--fg2)' }}>
          关联源码：<code>model/model_minimind.py:216</code> <code>class FeedForward</code> | <code>:232</code> <code>class MoEGate</code> | <code>:288</code> <code>class MOEFeedForward</code>
        </small>
      </p>

      <Card title="SwiGLU 逐步计算动画">
        <p style={{ marginBottom: 10, fontSize: '0.9rem', color: 'var(--fg2)' }}>
          一步步演示 SwiGLU 前馈网络如何处理一个 token 的隐藏向量：门控分支 × 候选分支 → 压缩输出。
        </p>
        <div style={{ display: 'flex', gap: 8, marginBottom: 10, flexWrap: 'wrap' }}>
          <button className="btn" onClick={() => setFfnStep(prev => Math.max(prev - 1, 0))}>◀ 上一步</button>
          <button className="btn primary" onClick={() => setFfnStep(prev => Math.min(prev + 1, SWIGLU_STEPS.length - 1))}>下一步 ▶</button>
          <button className="btn" onClick={playFfn}>{ffnPlaying ? '⏸ 暂停' : '▶ 自动播放'}</button>
          <button className="btn" onClick={resetFfn}>重置</button>
        </div>
        <div className="step-indicator">
          {SWIGLU_STEPS.map((s, i) => (
            <div
              key={i}
              className={`step-dot${i === ffnStep ? ' active' : ''}${i < ffnStep ? ' done' : ''}`}
              onClick={() => setFfnStep(i)}
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
              height={330}
              viewBox="0 0 310 330"
              onClick={(e) => {
                const t = (e.target as SVGElement).closest('rect[data-fstep]');
                if (t) setFfnStep(parseInt(t.getAttribute('data-fstep')!));
              }}
              dangerouslySetInnerHTML={{ __html: ffnFlowSvg }}
            />
          </div>
          <div>
            {ffnStep >= 0 ? (() => {
              const cur = SWIGLU_STEPS[ffnStep];
              return (
                <>
                  <div style={{ fontSize: '1.1rem', fontWeight: 700, color: cur.color, marginBottom: 6 }}>
                    {ffnStep}. {cur.name}
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
                  {ffnStepDetailSvg && (
                    <svg width="100%" height={ffnStep === 2 || ffnStep === 4 ? 130 : 50} viewBox={`0 0 260 ${ffnStep === 2 || ffnStep === 4 ? 130 : 50}`} dangerouslySetInnerHTML={{ __html: ffnStepDetailSvg }} />
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

      <Card title="SwiGLU 数据流">
        <p style={{ marginBottom: 10, fontSize: '0.9rem', color: 'var(--fg2)' }}>
          SwiGLU 是 FFN 的核心激活机制，通过门控选择性地传递信息。数据流为：输入 x（512维）同时送入两个分支——gate_proj 产生门控信号经 SiLU 激活，up_proj 产生候选特征，两者逐元素相乘后由 down_proj 降维回 512。
          中间维度 1408 ≈ round_up(512 × 8/3, 64)，兼顾表达力和效率。下图展示了完整的数据流向：
        </p>
        <svg width="100%" height={200} viewBox="0 0 700 200" dangerouslySetInnerHTML={{ __html: swigluSvg }} />
        <div className="silu-demo" style={{ marginTop: 12 }}>
          <div>
            <div className="label">SiLU(x) = x · σ(x) 函数图像</div>
            <canvas ref={siluCanvasRef} />
          </div>
          <div style={{ fontSize: '0.85rem', color: 'var(--fg2)', maxWidth: 300 }}>
            <p><strong>为什么用 SwiGLU？</strong></p>
            <p>相比 ReLU，SiLU 平滑且非单调，门控机制让网络可以选择性地传递信息，实验证明效果更好。</p>
          </div>
        </div>
        <SourcePanel
          title="对照源码：model/model_minimind.py:216-229"
          code={`class FeedForward(nn.Module):
    def __init__(self, config):
        # 中间维度 = round_up(hidden_size * 8/3, 64) = 1408
        # 8/3 是 SwiGLU 的推荐扩展比（相比 ReLU 的 4x，参数量相近但效果更好）
        self.gate_proj = nn.Linear(512, 1408, bias=False)  # 门控分支：决定"放行"哪些特征
        self.down_proj = nn.Linear(1408, 512, bias=False)  # 降维：1408 → 512 回到隐藏维度
        self.up_proj   = nn.Linear(512, 1408, bias=False)  # 候选分支：产生待筛选的特征
        self.act_fn = silu  # SiLU(x) = x × σ(x)，平滑非单调激活函数

    def forward(self, x):
        # gate_proj(x) → SiLU 激活作为"门"
        # up_proj(x) 作为候选特征
        # 两者逐元素相乘 (⊙) = 门控选择性地放行候选特征
        # 最后 down_proj 降维回 512
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))`}
        />
      </Card>

      <Card title="MoE 路由动画">
        <p style={{ marginBottom: 10, fontSize: '0.9rem', color: 'var(--fg2)' }}>
          MoE（混合专家）用多个独立的 FFN（专家）替代单个 FFN。门控网络 W_gate 对每个 token 计算各专家的 softmax 概率，
          选取 top-2 专家进行计算，并用归一化权重加权求和。这样模型参数更多但每个 token 只激活部分参数，兼顾容量与效率。
        </p>
        <p style={{ marginBottom: 10, fontSize: '0.9rem', color: 'var(--fg2)' }}>
          点击"运行路由"按钮，动画展示 8 个 token 被随机路由到 4 个专家的过程。连线粗细代表门控权重大小，
          每个 token 同时送给 2 个专家处理，最后还会加上共享专家（紫色）的输出。每次点击会重新随机采样门控分数。
        </p>
        <div style={{ display: 'flex', gap: 8, marginBottom: 10 }}>
          <button className="btn primary" onClick={handleMoeRun}>▶ 运行路由</button>
          <button className="btn" onClick={handleMoeReset}>重置</button>
        </div>
        <canvas ref={moeCanvasRef} />
        <SourcePanel
          title="对照源码：model/model_minimind.py:232-349"
          code={`class MoEGate(nn.Module):
    """门控网络：决定每个 token 路由到哪些专家"""
    def forward(self, hidden_states):
        # 门控打分：hidden_states @ W_gate → 每个专家的 logit
        logits = F.linear(hidden_states, self.weight)  # [B*S, n_experts]
        # softmax 得到概率分布（所有专家概率之和 = 1）
        scores = logits.softmax(dim=-1)
        # 选取概率最高的 top-2 专家
        topk_weight, topk_idx = torch.topk(scores, k=2)  # top-2
        # 归一化 top-2 权重使其和为 1（丢弃的专家概率重新分配）
        topk_weight = topk_weight / topk_weight.sum(dim=-1, keepdim=True)
        # aux_loss = α × N × Σ(fᵢ × Pᵢ)，鼓励负载均衡
        return topk_idx, topk_weight, aux_loss

class MOEFeedForward(nn.Module):
    """MoE FFN 层：4 个路由专家 + 1 个共享专家"""
    def forward(self, x):
        # 第一步：门控路由，确定每个 token 去哪两个专家
        topk_idx, topk_weight, aux_loss = self.gate(x)
        # 第二步：将 token 分发到对应专家独立计算
        for i, expert in enumerate(self.experts):
            # 只有被路由到专家 i 的 token 才会经过 expert_i 计算
            y[topk_idx == i] = expert(x[topk_idx == i])
        # 第三步：用归一化权重加权求和 top-2 专家的输出
        y = (y * topk_weight).sum(dim=1)
        # 第四步：所有 token 都额外经过共享专家，保证基础能力
        y = y + self.shared_experts(identity)`}
        />
      </Card>

      <Card title="负载均衡">
        <p style={{ marginBottom: 10, fontSize: '0.9rem', color: 'var(--fg2)' }}>
          MoE 的常见问题是"专家塌缩"——门控网络可能学会把所有 token 都发给少数专家，导致其他专家从不被训练。
          辅助损失 aux_loss 通过惩罚不均衡的路由分布来解决这个问题。先运行上方的路由动画后，
          左侧柱状图会显示各专家实际收到的 token 数，右侧显示计算出的 aux_loss 值。
          当所有专家均匀分配时，aux_loss 达到最小值。
        </p>
        <div className="viz-grid">
          <div>
            <div className="label">各专家收到的 token 数</div>
            <div style={{ height: 140, display: 'flex', alignItems: 'flex-end', gap: 8, paddingBottom: 20, position: 'relative' }}>
              {Array.from({ length: MM.n_routed_experts }, (_, e) => {
                const count = loadBalanceData?.counts[e] ?? 0;
                const maxCount = loadBalanceData?.maxCount ?? 1;
                return (
                  <div key={e} style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 4 }}>
                    <div style={{
                      width: '100%',
                      height: loadBalanceData ? (count / maxCount * 100) + 'px' : '4px',
                      background: COLORS[e],
                      borderRadius: '4px 4px 0 0',
                      transition: 'height 0.5s',
                      minHeight: 4,
                    }} />
                    <div style={{ fontSize: '0.75rem', color: 'var(--fg2)' }}>E{e}: {count}</div>
                  </div>
                );
              })}
            </div>
          </div>
          <div>
            <div className="label">aux_loss = α × Σ(fᵢ × Pᵢ)</div>
            <div style={{ fontSize: '1.5rem', color: 'var(--accent)', fontFamily: 'monospace', padding: '20px 0' }}>
              {loadBalanceData ? loadBalanceData.auxLoss.toFixed(6) : '0.0000'}
            </div>
            <p style={{ fontSize: '0.85rem', color: 'var(--fg2)' }}>
              α=0.01, fᵢ=专家i实际频率, Pᵢ=专家i平均门控概率。<br />当所有专家均衡时，aux_loss 最小。
            </p>
          </div>
        </div>
      </Card>
    </>
  );
}
