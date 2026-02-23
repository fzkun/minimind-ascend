import{u as Q,r as g,j as t}from"./index-DB782DBc.js";import{C as u,S}from"./SourcePanel-BwVAQ71f.js";import{u as M}from"./useCanvas-BahyMyhc.js";import{M as y}from"./constants-BXPCUJh7.js";import{t as A,s as B,m as T}from"./utils-ZSIfeTeC.js";function N(){const{isDark:r}=Q(),[k,j]=g.useState(1),p=["我","喜","欢","你"],v=g.useMemo(()=>{const e=r?"#e2e8f0":"#1a1a2e",i=r?"#94a3b8":"#555",x=r?"#818cf8":"#4f46e5",o=r?"#34d399":"#10b981",l=r?"#fbbf24":"#f59e0b",d=r?"#60a5fa":"#3b82f6";return`
      <rect x="10" y="20" width="120" height="40" rx="6" fill="none" stroke="${i}" stroke-width="1.5"/>
      <text x="70" y="45" text-anchor="middle" fill="${e}" font-size="12">Input [B,S,512]</text>
      <line x1="130" y1="40" x2="180" y2="40" stroke="${i}" stroke-width="1" marker-end="url(#arrowGray)"/>
      <defs><marker id="arrowGray" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0 L10,5 L0,10 Z" fill="${i}"/></marker></defs>
      <rect x="185" y="5" width="100" height="30" rx="5" fill="${d}" opacity="0.15" stroke="${d}" stroke-width="1.5"/>
      <text x="235" y="24" text-anchor="middle" fill="${d}" font-size="11" font-weight="bold">Q Proj</text>
      <text x="235" y="52" text-anchor="middle" fill="${i}" font-size="10">512→512</text>
      <rect x="185" y="62" width="100" height="30" rx="5" fill="${o}" opacity="0.15" stroke="${o}" stroke-width="1.5"/>
      <text x="235" y="81" text-anchor="middle" fill="${o}" font-size="11" font-weight="bold">K Proj</text>
      <text x="235" y="108" text-anchor="middle" fill="${i}" font-size="10">512→128</text>
      <rect x="185" y="118" width="100" height="30" rx="5" fill="${l}" opacity="0.15" stroke="${l}" stroke-width="1.5"/>
      <text x="235" y="137" text-anchor="middle" fill="${l}" font-size="11" font-weight="bold">V Proj</text>
      <text x="235" y="163" text-anchor="middle" fill="${i}" font-size="10">512→128</text>
      <line x1="285" y1="20" x2="340" y2="20" stroke="${d}" stroke-width="1"/>
      <line x1="285" y1="77" x2="340" y2="77" stroke="${o}" stroke-width="1"/>
      <line x1="285" y1="133" x2="340" y2="133" stroke="${l}" stroke-width="1"/>
      <text x="380" y="12" text-anchor="middle" fill="${i}" font-size="9">8 个 Q 头</text>
      ${[0,1,2,3,4,5,6,7].map(h=>`<rect x="${345+h*22}" y="16" width="18" height="14" rx="2" fill="${d}" opacity="${.4+h*.07}"/>`).join("")}
      <text x="380" y="69" text-anchor="middle" fill="${i}" font-size="9">2 个 KV 头</text>
      <rect x="345" y="73" width="40" height="14" rx="2" fill="${o}" opacity="0.6"/>
      <rect x="390" y="73" width="40" height="14" rx="2" fill="${o}" opacity="0.8"/>
      <rect x="345" y="129" width="40" height="14" rx="2" fill="${l}" opacity="0.6"/>
      <rect x="390" y="129" width="40" height="14" rx="2" fill="${l}" opacity="0.8"/>
      <text x="530" y="12" text-anchor="middle" fill="${e}" font-size="11" font-weight="bold">GQA 分组</text>
      <rect x="470" y="20" width="120" height="50" rx="6" fill="${o}" opacity="0.08" stroke="${o}" stroke-width="1" stroke-dasharray="4,2"/>
      <text x="530" y="36" text-anchor="middle" fill="${i}" font-size="9">KV Head 0 → Q Head 0,1,2,3</text>
      ${[0,1,2,3].map(h=>`<rect x="${485+h*26}" y="42" width="22" height="12" rx="2" fill="${d}" opacity="0.7"/>`).join("")}
      <text x="530" y="64" text-anchor="middle" fill="${o}" font-size="9">repeat_kv ×4</text>
      <rect x="470" y="78" width="120" height="50" rx="6" fill="${l}" opacity="0.08" stroke="${l}" stroke-width="1" stroke-dasharray="4,2"/>
      <text x="530" y="94" text-anchor="middle" fill="${i}" font-size="9">KV Head 1 → Q Head 4,5,6,7</text>
      ${[0,1,2,3].map(h=>`<rect x="${485+h*26}" y="100" width="22" height="12" rx="2" fill="${d}" opacity="0.7"/>`).join("")}
      <text x="530" y="122" text-anchor="middle" fill="${l}" font-size="9">repeat_kv ×4</text>
      <rect x="470" y="145" width="120" height="30" rx="6" fill="none" stroke="${x}" stroke-width="1.5"/>
      <text x="530" y="164" text-anchor="middle" fill="${x}" font-size="11">scores @ V → O Proj</text>
      <text x="300" y="200" text-anchor="middle" fill="${e}" font-size="12">Attention(Q,K,V) = softmax(QK<tspan baseline-shift="super" font-size="9">T</tspan>/√d<tspan baseline-shift="sub" font-size="9">k</tspan> + mask) × V</text>
      <text x="300" y="220" text-anchor="middle" fill="${i}" font-size="10">d_k = head_dim = ${y.head_dim}，√d_k = ${Math.sqrt(y.head_dim).toFixed(1)}</text>
      <text x="300" y="245" text-anchor="middle" fill="${i}" font-size="10">GQA: ${y.num_heads} Q heads 共享 ${y.num_kv_heads} KV heads → 减少 KV 缓存 ${y.num_heads/y.num_kv_heads}×</text>
    `},[r]),w=M((e,i,x)=>{const o=p.length,l=T(123),d=[];for(let s=0;s<o;s++){d[s]=[];for(let a=0;a<o;a++)d[s][a]=(l()-.3)*3}const h=[];for(let s=0;s<o;s++){const a=[];for(let c=0;c<o;c++)a.push(c<=s?d[s][c]:-1e9);h.push(B(a,k))}const f=50,n=(i-f-10)/o;e.fillStyle=r?"#1e293b":"#f8f8f8",e.fillRect(0,0,i,x),e.font="13px sans-serif",e.fillStyle=r?"#e2e8f0":"#1a1a2e",e.textAlign="center";for(let s=0;s<o;s++)e.fillText(p[s],f+n*s+n/2,f-10),e.fillText(p[s],f-20,f+n*s+n/2+5);e.font="10px sans-serif",e.fillStyle=r?"#94a3b8":"#888",e.fillText("Key →",f+n*o/2,15),e.save(),e.translate(10,f+n*o/2),e.rotate(-Math.PI/2),e.fillText("Query →",0,0),e.restore();for(let s=0;s<o;s++)for(let a=0;a<o;a++){const c=h[s][a],_=f+a*n,$=f+s*n;if(a>s)e.fillStyle=r?"#334155":"#e5e7eb",e.fillRect(_,$,n-1,n-1),e.fillStyle=r?"#64748b":"#aaa",e.font="10px sans-serif",e.textAlign="center",e.fillText("-∞",_+n/2,$+n/2+3);else{const m=Math.min(c*1.2,1),z=Math.round(r?129+126*(1-m):79*m),V=Math.round(r?140+115*(1-m):70*m),b=r?248:Math.round(229*m);e.fillStyle=r?`rgb(${z},${V},${b})`:`rgba(79,70,229,${m})`,e.fillRect(_,$,n-1,n-1),e.fillStyle=m>.5?"#fff":r?"#e2e8f0":"#333",e.font="11px monospace",e.textAlign="center",e.fillText(c.toFixed(2),_+n/2,$+n/2+4)}}},[r,k],300,300),K=g.useMemo(()=>{const e=r?"#e2e8f0":"#1a1a2e",i=r?"#94a3b8":"#555",x=r?"#34d399":"#10b981",o=r?"#60a5fa":"#3b82f6",l=r?"#fbbf24":"#f59e0b";return`
      <defs><marker id="arrKV" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0 L10,5 L0,10 Z" fill="${i}"/></marker></defs>
      <text x="20" y="20" fill="${e}" font-size="12" font-weight="bold">推理阶段 KV Cache</text>
      <rect x="20" y="35" width="200" height="40" rx="6" fill="${x}" opacity="0.12" stroke="${x}" stroke-width="1.5" stroke-dasharray="4,2"/>
      <text x="120" y="50" text-anchor="middle" fill="${x}" font-size="10">Cached K: [B, t-1, 2, 64]</text>
      <text x="120" y="65" text-anchor="middle" fill="${x}" font-size="10">Cached V: [B, t-1, 2, 64]</text>
      <rect x="240" y="35" width="100" height="40" rx="6" fill="${o}" opacity="0.15" stroke="${o}" stroke-width="1.5"/>
      <text x="290" y="50" text-anchor="middle" fill="${o}" font-size="10">New Q: [B,1,8,64]</text>
      <text x="290" y="65" text-anchor="middle" fill="${o}" font-size="10">New K,V: [B,1,2,64]</text>
      <line x1="220" y1="80" x2="260" y2="80" stroke="${i}" stroke-width="1" marker-end="url(#arrKV)"/>
      <text x="240" y="95" text-anchor="middle" fill="${i}" font-size="9">concat</text>
      <rect x="380" y="35" width="180" height="40" rx="6" fill="${l}" opacity="0.12" stroke="${l}" stroke-width="1.5"/>
      <text x="470" y="50" text-anchor="middle" fill="${l}" font-size="10">K: [B, t, 2, 64] (更新缓存)</text>
      <text x="470" y="65" text-anchor="middle" fill="${l}" font-size="10">Attn: Q×K^T → [B,8,1,t]</text>
      <line x1="340" y1="55" x2="380" y2="55" stroke="${i}" stroke-width="1" marker-end="url(#arrKV)"/>
      <text x="350" y="110" text-anchor="middle" fill="${i}" font-size="11">每步只计算 1 个新 token 的 Q/K/V，拼接到缓存 → O(1) 生成而非 O(t)</text>
      <text x="350" y="128" text-anchor="middle" fill="${i}" font-size="10">GQA 缓存: 2 个 KV 头 vs MHA 的 8 个 → 节省 4× 显存</text>
    `},[r]);return t.jsxs(t.Fragment,{children:[t.jsx("h2",{children:"3. 自注意力 (Self-Attention)"}),t.jsxs("p",{className:"desc",children:['注意力机制让序列中的每个 token 能"看到"其他 token 并动态计算相关度权重。 MiniMind 使用 GQA（分组查询注意力）：',t.jsx("code",{children:"num_attention_heads=8"})," 个 Q 头共享 ",t.jsx("code",{children:"num_kv_heads=2"})," 个 KV 头， 每个头的维度 ",t.jsx("code",{children:"head_dim=64"}),"。",t.jsx("br",{}),t.jsxs("small",{style:{color:"var(--fg2)"},children:["关联源码：",t.jsx("code",{children:"model/model_minimind.py:150"})," ",t.jsx("code",{children:"class Attention"})," | ",t.jsx("code",{children:":169"})," ",t.jsx("code",{children:"def forward"})]})]}),t.jsxs(u,{title:"Q / K / V 计算与 GQA 分组",children:[t.jsx("p",{style:{marginBottom:10,fontSize:"0.9rem",color:"var(--fg2)"},children:"注意力计算分为三步：(1) 通过线性投影生成 Query（查询）、Key（键）、Value（值）三组向量； (2) Q 和 K 的点积计算注意力分数，衡量 token 间的相关性；(3) 用分数加权求和 V，得到融合了上下文信息的输出。"}),t.jsx("p",{style:{marginBottom:10,fontSize:"0.9rem",color:"var(--fg2)"},children:"GQA（分组查询注意力）是 MHA 的高效变体：8 个 Q 头共享 2 个 KV 头（每 4 个 Q 头复用 1 个 KV 头）， KV 投影从 512→128（而非 512→512），推理时 KV Cache 显存减少 4 倍，效果接近完整 MHA。 下图展示了从输入到分组的完整流程："}),t.jsx("svg",{width:"100%",height:260,viewBox:"0 0 700 260",dangerouslySetInnerHTML:{__html:v}}),t.jsx(S,{title:"对照源码：model/model_minimind.py:150-213 (Attention)",code:`class Attention(nn.Module):
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
        output = softmax(scores) @ xv  # [B,S,8,64] → concat → O_proj → [B,S,512]`})]}),t.jsxs(u,{title:"注意力热力图",children:[t.jsx("p",{style:{marginBottom:10,fontSize:"0.9rem",color:"var(--fg2)"},children:'热力图展示 4 个 token 之间的注意力权重矩阵。每个格子的颜色深浅代表 Query token（行）对 Key token（列）的关注程度。 因果掩码将上三角设为 -∞，确保 token 只能看到自身和之前的 token（自回归约束）。 拖动温度滑块可以观察分布变化：低温使权重集中在少数 token（更"确定"），高温使权重更均匀（更"随机"）。'}),t.jsxs("div",{className:"viz-grid",children:[t.jsx("div",{children:t.jsx("canvas",{ref:w})}),t.jsxs("div",{children:[t.jsxs("div",{style:{marginBottom:8},children:[t.jsx("div",{className:"label",children:"温度参数 (Temperature)"}),t.jsx("input",{type:"range",min:"0.1",max:"3",step:"0.1",value:k,onChange:e=>j(parseFloat(e.target.value))}),t.jsx("span",{className:"value",children:k.toFixed(1)})]}),t.jsxs("div",{style:{marginBottom:8},children:[t.jsx("div",{className:"label",children:"tokens"}),t.jsx("div",{style:{display:"flex",gap:4},children:p.map((e,i)=>t.jsx("span",{className:"token-box",style:{background:A(i),color:"#fff"},children:e},i))})]}),t.jsx("div",{className:"label",children:"说明"}),t.jsx("p",{style:{fontSize:"0.85rem",color:"var(--fg2)"},children:"每行表示一个 token 对所有可见 token 的注意力权重（softmax 后）。因果掩码确保 token 只能看到自身和之前的 token。降低温度使分布更尖锐，升高则更平滑。"})]})]})]}),t.jsxs(u,{title:"KV Cache 示意",children:[t.jsx("p",{style:{marginBottom:10,fontSize:"0.9rem",color:"var(--fg2)"},children:"推理（生成）阶段每步只产生 1 个新 token。如果每次都重新计算所有 token 的 K/V，复杂度为 O(t²)。 KV Cache 的思路：将之前所有 token 的 K 和 V 缓存起来，新 token 只需计算自身的 Q/K/V， 然后将新 K/V 拼接到缓存中，用完整的 K/V 计算注意力，复杂度降为 O(t)。 GQA 进一步减少缓存量：只需缓存 2 个 KV 头（而非 8 个），节省 4 倍显存。"}),t.jsx("svg",{width:"100%",height:140,viewBox:"0 0 700 140",dangerouslySetInnerHTML:{__html:K}})]})]})}export{N as default};
