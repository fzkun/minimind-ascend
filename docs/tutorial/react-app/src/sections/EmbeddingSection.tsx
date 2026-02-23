import { useState, useMemo } from 'react';
import Card from '../components/Card';
import SourcePanel from '../components/SourcePanel';
import { useTheme } from '../context/ThemeContext';
import { MM } from '../constants';
import { mulberry32 } from '../utils';

interface DemoToken {
  text: string;
  id: number;
}

const DEMO_TOKENS: DemoToken[] = [
  { text: '你', id: 340 }, { text: '好', id: 590 }, { text: '学', id: 670 },
  { text: '习', id: 680 }, { text: 'AI', id: 1100 }, { text: '。', id: 16 },
];

function generateEmbData() {
  const rng = mulberry32(42);
  const data: Record<number, number[]> = {};
  DEMO_TOKENS.forEach(t => {
    data[t.id] = Array.from({ length: MM.hidden_size }, () => (rng() - 0.5) * 2);
  });
  return data;
}

const EMB_DATA = generateEmbData();

export default function EmbeddingSection() {
  const { isDark } = useTheme();
  const [selectedIdx, setSelectedIdx] = useState(0);
  const token = DEMO_TOKENS[selectedIdx];

  const matrixRows = useMemo(() => {
    const tid = token.id;
    const displayIds: number[] = [];
    for (let r = Math.max(0, tid - 2); r <= Math.min(MM.vocab_size - 1, tid + 2); r++) {
      displayIds.push(r);
    }
    return displayIds.map(id => {
      const isTarget = id === tid;
      const vals = EMB_DATA[id] || (() => {
        const r2 = mulberry32(id * 512 + 7);
        return Array.from({ length: MM.hidden_size }, () => (r2() - 0.5) * 2);
      })();
      return { id, isTarget, vals };
    });
  }, [token.id]);

  const barData = useMemo(() => {
    const vec = EMB_DATA[token.id];
    const show = 64;
    const maxVal = Math.max(...vec.slice(0, show).map(Math.abs));
    return vec.slice(0, show).map((v, i) => ({
      value: v,
      height: (Math.abs(v) / maxVal) * 80,
      dim: i,
    }));
  }, [token.id]);

  const weightTyingSvg = useMemo(() => {
    const fg = isDark ? '#e2e8f0' : '#1a1a2e';
    const accent = isDark ? '#818cf8' : '#4f46e5';
    const green = isDark ? '#34d399' : '#10b981';
    return `
      <rect x="20" y="30" width="150" height="50" rx="8" fill="none" stroke="${accent}" stroke-width="2"/>
      <text x="95" y="60" text-anchor="middle" fill="${fg}" font-size="14">Embedding</text>
      <text x="95" y="100" text-anchor="middle" fill="${fg}" font-size="11">[6400 × 512]</text>
      <rect x="430" y="30" width="150" height="50" rx="8" fill="none" stroke="${accent}" stroke-width="2"/>
      <text x="505" y="60" text-anchor="middle" fill="${fg}" font-size="14">LM Head</text>
      <text x="505" y="100" text-anchor="middle" fill="${fg}" font-size="11">[512 × 6400]</text>
      <line x1="170" y1="55" x2="430" y2="55" stroke="${green}" stroke-width="2" stroke-dasharray="8,4"/>
      <text x="300" y="48" text-anchor="middle" fill="${green}" font-size="12" font-weight="bold">权重共享 (Weight Tying)</text>
      <text x="300" y="140" text-anchor="middle" fill="${fg}" font-size="11" opacity="0.7">embed_tokens.weight = lm_head.weight → 节省 512×6400 = 3.3M 参数</text>
    `;
  }, [isDark]);

  return (
    <>
      <h2>2. Token Embedding</h2>
      <p className="desc">
        把 token ID 映射成一个 <code>hidden_size=512</code> 维的向量，相当于 <code>nn.Embedding(6400, 512)</code> 的查表操作：
        输入 token ID，输出一行 512 维浮点数。MiniMind 中 Embedding 层和 LM Head 共享同一个权重矩阵（<code>tie_word_embeddings=True</code>）。
        <br/>
        <small style={{ color: 'var(--fg2)' }}>
          关联源码：<code>model/model_minimind.py:381</code> <code>self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)</code>
        </small>
      </p>

      <Card title="矩阵查找动画">
        <p style={{ marginBottom: 10, fontSize: '0.9rem', color: 'var(--fg2)' }}>
          Embedding 本质上是一次查表操作：以 token ID 为行索引，从形状为 [6400, 512] 的权重矩阵中取出对应行，
          得到一个 512 维的稠密向量。这个向量就是该 token 的语义表示，模型后续所有计算都基于它。
          点击下方示例 token，左侧矩阵高亮对应行，右侧柱状图展示前 64 维的数值分布（蓝色=正值，红色=负值）：
        </p>
        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginBottom: 12 }}>
          {DEMO_TOKENS.map((t, i) => (
            <button
              key={t.id}
              className={`btn${i === selectedIdx ? ' primary' : ''}`}
              onClick={() => setSelectedIdx(i)}
            >
              &quot;{t.text}&quot; ({t.id})
            </button>
          ))}
        </div>
        <div className="viz-grid">
          <div>
            <div className="label">Embedding 矩阵 (6400 × 512) 局部视图</div>
            <div style={{ overflow: 'auto', maxHeight: 260 }}>
              <table className="matrix-table">
                <thead>
                  <tr>
                    <th>ID</th>
                    {Array.from({ length: 8 }, (_, j) => <th key={j}>d{j}</th>)}
                    <th>...</th>
                    {[508, 509, 510, 511].map(j => <th key={j}>d{j}</th>)}
                  </tr>
                </thead>
                <tbody>
                  {matrixRows.map(row => (
                    <tr key={row.id}>
                      <td className={row.isTarget ? 'highlight' : ''}>{row.id}</td>
                      {Array.from({ length: 8 }, (_, j) => (
                        <td key={j} className={row.isTarget ? 'highlight' : ''}>
                          {row.vals[j].toFixed(2)}
                        </td>
                      ))}
                      <td className={row.isTarget ? 'highlight' : ''}>...</td>
                      {[508, 509, 510, 511].map(j => (
                        <td key={j} className={row.isTarget ? 'highlight' : ''}>
                          {row.vals[j].toFixed(2)}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
          <div>
            <div className="label">输出向量 (512 维) 前 64 维可视化</div>
            <div style={{ height: 180, display: 'flex', alignItems: 'flex-end', gap: 1 }}>
              {barData.map((b, i) => (
                <div
                  key={i}
                  style={{
                    flex: 1, minWidth: 3,
                    height: b.height,
                    background: b.value >= 0 ? 'var(--accent)' : 'var(--red)',
                    borderRadius: '1px 1px 0 0',
                    transition: 'height 0.4s ease',
                  }}
                  title={`d${b.dim}: ${b.value.toFixed(4)}`}
                />
              ))}
            </div>
          </div>
        </div>
        <SourcePanel
          title="对照源码：model/model_minimind.py:381, 434-435"
          code={`# MiniMindModel.__init__
# 创建 Embedding 查找表：输入 token_id (0~6399) → 输出 512 维向量
self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)  # [6400, 512]

# MiniMindForCausalLM.__init__ — 权重共享 (Weight Tying)
# LM Head 将隐藏状态投影回词表空间 [512] → [6400]
self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
# 关键：让 embed_tokens 和 lm_head 共享同一个 Parameter
# 这样 embed_tokens.weight 和 lm_head.weight 指向同一块 GPU 显存
# 效果：节省 3.3M 参数 + 输入/输出语义空间对齐
self.model.embed_tokens.weight = self.lm_head.weight  # 共享！`}
        />
      </Card>

      <Card title="权重共享 (Weight Tying)">
        <p style={{ marginBottom: 10, fontSize: '0.9rem', color: 'var(--fg2)' }}>
          输入端的 Embedding 层（token ID → 向量）和输出端的 LM Head（向量 → logits）共享同一个权重矩阵 [6400, 512]。
          这意味着语义相近的 token 在输入空间和输出空间中表现一致，同时节省了 512 × 6400 = 3.3M 参数（约占小模型 12%）。
          下图中虚线表示两个矩阵指向同一块内存，训练时梯度会同时更新它们。
        </p>
        <svg width="100%" height={160} viewBox="0 0 600 160" dangerouslySetInnerHTML={{ __html: weightTyingSvg }} />
      </Card>
    </>
  );
}
