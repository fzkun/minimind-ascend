import{u as B,r as S,j as t}from"./index-DB782DBc.js";import{C as G,S as W}from"./SourcePanel-BwVAQ71f.js";import{u as U}from"./useCanvas-BahyMyhc.js";import{M as _,C as L}from"./constants-BXPCUJh7.js";import{a as C,s as I,m as O}from"./utils-ZSIfeTeC.js";const A=8,X=["我","是","一","个","语","言","模","型"],H=["专家0","专家1","专家2","专家3"];function Y(){const i=O(Date.now()%1e4),m=[],$=[];for(let w=0;w<A;w++){const E=Array.from({length:_.n_routed_experts},()=>(i()-.3)*4),f=I(E),R=f.map((c,h)=>({p:c,i:h})).sort((c,h)=>h.p-c.p),P=R.slice(0,_.num_experts_per_tok).map(c=>c.i),z=R.slice(0,_.num_experts_per_tok).map(c=>c.p),N=z.reduce((c,h)=>c+h,0);$.push({experts:P,weights:z.map(c=>c/N),allProbs:f}),m.push(f)}return{gates:m,assignments:$}}function Q(){const{isDark:i}=B(),[m,$]=S.useState(null),[w,E]=S.useState(0),f=S.useRef(null);S.useEffect(()=>()=>{f.current&&cancelAnimationFrame(f.current)},[]);const R=S.useCallback(()=>{f.current&&(cancelAnimationFrame(f.current),f.current=null);const e=Y();$(e);let o=null;const d=1500,s=p=>{o||(o=p);const n=Math.min(1,(p-o)/d);E(n),n<1?f.current=requestAnimationFrame(s):f.current=null};f.current=requestAnimationFrame(s)},[]),P=S.useCallback(()=>{f.current&&(cancelAnimationFrame(f.current),f.current=null),$(null),E(0)},[]),z=S.useMemo(()=>{const e=i?"#e2e8f0":"#1a1a2e",o=i?"#94a3b8":"#555",d=i?"#818cf8":"#4f46e5",s=i?"#34d399":"#10b981",p=i?"#fbbf24":"#f59e0b",n=i?"#60a5fa":"#3b82f6";return`
      <defs><marker id="arrSG" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0 L10,5 L0,10 Z" fill="${o}"/></marker></defs>
      <rect x="10" y="75" width="80" height="40" rx="6" fill="none" stroke="${o}" stroke-width="1.5"/>
      <text x="50" y="99" text-anchor="middle" fill="${e}" font-size="11">x [512]</text>
      <line x1="90" y1="95" x2="130" y2="55" stroke="${o}" stroke-width="1" marker-end="url(#arrSG)"/>
      <line x1="90" y1="95" x2="130" y2="135" stroke="${o}" stroke-width="1" marker-end="url(#arrSG)"/>
      <rect x="135" y="35" width="110" height="35" rx="6" fill="${n}" opacity="0.15" stroke="${n}" stroke-width="1.5"/>
      <text x="190" y="50" text-anchor="middle" fill="${n}" font-size="10" font-weight="bold">gate_proj</text>
      <text x="190" y="63" text-anchor="middle" fill="${o}" font-size="9">512→1408</text>
      <rect x="135" y="120" width="110" height="35" rx="6" fill="${s}" opacity="0.15" stroke="${s}" stroke-width="1.5"/>
      <text x="190" y="135" text-anchor="middle" fill="${s}" font-size="10" font-weight="bold">up_proj</text>
      <text x="190" y="148" text-anchor="middle" fill="${o}" font-size="9">512→1408</text>
      <line x1="245" y1="52" x2="285" y2="52" stroke="${o}" stroke-width="1" marker-end="url(#arrSG)"/>
      <rect x="290" y="35" width="70" height="35" rx="6" fill="${p}" opacity="0.15" stroke="${p}" stroke-width="1.5"/>
      <text x="325" y="57" text-anchor="middle" fill="${p}" font-size="11" font-weight="bold">SiLU</text>
      <line x1="360" y1="52" x2="410" y2="90" stroke="${o}" stroke-width="1" marker-end="url(#arrSG)"/>
      <line x1="245" y1="137" x2="410" y2="96" stroke="${o}" stroke-width="1" marker-end="url(#arrSG)"/>
      <circle cx="420" cy="93" r="14" fill="none" stroke="${d}" stroke-width="2"/>
      <text x="420" y="98" text-anchor="middle" fill="${d}" font-size="16" font-weight="bold">×</text>
      <text x="420" y="120" text-anchor="middle" fill="${o}" font-size="9">[1408]</text>
      <line x1="434" y1="93" x2="480" y2="93" stroke="${o}" stroke-width="1" marker-end="url(#arrSG)"/>
      <rect x="485" y="75" width="110" height="35" rx="6" fill="${d}" opacity="0.15" stroke="${d}" stroke-width="1.5"/>
      <text x="540" y="90" text-anchor="middle" fill="${d}" font-size="10" font-weight="bold">down_proj</text>
      <text x="540" y="103" text-anchor="middle" fill="${o}" font-size="9">1408→512</text>
      <line x1="595" y1="93" x2="640" y2="93" stroke="${o}" stroke-width="1" marker-end="url(#arrSG)"/>
      <rect x="645" y="75" width="50" height="35" rx="6" fill="none" stroke="${o}" stroke-width="1.5"/>
      <text x="670" y="97" text-anchor="middle" fill="${e}" font-size="11">out</text>
      <text x="350" y="185" text-anchor="middle" fill="${e}" font-size="12">FFN(x) = down_proj( SiLU(gate_proj(x)) ⊙ up_proj(x) )</text>
    `},[i]),N=U((e,o,d)=>{e.fillStyle=i?"#1e293b":"#f8f8f8",e.fillRect(0,0,o,d);const s=30,p=10,n=10,a=25,x=o-s-p,T=d-n-a,u=[-5,5],b=[-1.5,5],v=r=>s+(r-u[0])/(u[1]-u[0])*x,g=r=>n+T-(r-b[0])/(b[1]-b[0])*T,l=i?"#475569":"#ccc";e.strokeStyle=l,e.lineWidth=.5,e.beginPath(),e.moveTo(v(0),n),e.lineTo(v(0),n+T),e.stroke(),e.beginPath(),e.moveTo(s,g(0)),e.lineTo(s+x,g(0)),e.stroke();const k=i?"#818cf8":"#4f46e5";e.strokeStyle=k,e.lineWidth=2,e.beginPath();for(let r=0;r<=x;r++){const F=u[0]+r/x*(u[1]-u[0]),M=C(F),j=g(M);r===0?e.moveTo(s+r,j):e.lineTo(s+r,j)}e.stroke();const y=i?"#f87171":"#ef4444";e.strokeStyle=y,e.lineWidth=1.5,e.setLineDash([4,3]),e.beginPath();for(let r=0;r<=x;r++){const F=u[0]+r/x*(u[1]-u[0]),M=Math.max(0,F),j=g(M);r===0?e.moveTo(s+r,j):e.lineTo(s+r,j)}e.stroke(),e.setLineDash([]),e.fillStyle=k,e.font="10px sans-serif",e.textAlign="left",e.fillText("SiLU",s+x-30,g(C(4.5))-3),e.fillStyle=y,e.fillText("ReLU",s+x-32,g(4.5)-3)},[i],220,140),c=U((e,o,d)=>{const s=i?"#e2e8f0":"#1a1a2e",p=i?"#94a3b8":"#888";e.fillStyle=i?"#1e293b":"#f8f8f8",e.fillRect(0,0,o,d);const n=40,a=220,x=80,T=70,u=120,b=120,v=580,g=w;for(let l=0;l<_.n_routed_experts;l++){const k=u+l*b;e.fillStyle=i?"#334155":"#e8e8e8",e.strokeStyle=L[l],e.lineWidth=2,e.beginPath(),e.roundRect(k-35,a-20,70,40,8),e.fill(),e.stroke(),e.fillStyle=s,e.font="11px sans-serif",e.textAlign="center",e.fillText(H[l],k,a+5)}e.fillStyle=i?"#334155":"#e8e8e8",e.strokeStyle=i?"#a78bfa":"#7c3aed",e.lineWidth=2,e.beginPath(),e.roundRect(v-35,a-20,70,40,8),e.fill(),e.stroke(),e.fillStyle=s,e.font="11px sans-serif",e.fillText("共享专家",v,a+5);for(let l=0;l<A;l++){const k=x+l*T;if(m&&g>0){const y=m.assignments[l];for(let r=0;r<_.num_experts_per_tok;r++){const F=y.experts[r],M=u+F*b,j=Math.min(1,g*2-r*.3);j>0&&(e.globalAlpha=j*y.weights[r],e.strokeStyle=L[F],e.lineWidth=Math.max(1,y.weights[r]*4),e.beginPath(),e.moveTo(k,n+18),e.lineTo(M,a-20),e.stroke())}g>.5&&(e.globalAlpha=Math.min(1,(g-.5)*2)*.5,e.strokeStyle=i?"#a78bfa":"#7c3aed",e.lineWidth=1.5,e.beginPath(),e.moveTo(k,n+18),e.lineTo(v,a-20),e.stroke()),e.globalAlpha=1}if(e.fillStyle=i?"#475569":"#ddd",e.strokeStyle=L[l%L.length],e.lineWidth=2,e.beginPath(),e.arc(k,n,18,0,Math.PI*2),e.fill(),e.stroke(),e.fillStyle=s,e.font="13px sans-serif",e.textAlign="center",e.fillText(X[l],k,n+5),m&&g>=1){const y=m.assignments[l];e.font="9px monospace",e.fillStyle=p,e.fillText(`E${y.experts[0]}:${y.weights[0].toFixed(2)} E${y.experts[1]}:${y.weights[1].toFixed(2)}`,k,n+33)}}if(e.fillStyle=s,e.font="12px sans-serif",e.textAlign="left",e.fillText("Token 序列",10,15),e.fillText("路由专家 (top-2)",10,a-35),e.fillText("+ 共享",v-30,a-35),m&&g>=1){e.font="10px monospace",e.textAlign="center";for(let l=0;l<_.n_routed_experts;l++){const k=u+l*b,y=m.assignments.filter(r=>r.experts.includes(l)).length;e.fillStyle=p,e.fillText(`${y} tokens`,k,a+35)}}e.fillStyle=p,e.font="10px sans-serif",e.textAlign="center",e.fillText("Gate: softmax(W_gate · x) → top-2 选择",o/2,d-15)},[i,m,w],680,320),h=S.useMemo(()=>{if(!m||w<1)return null;const e=Array(_.n_routed_experts).fill(0),o=Array(_.n_routed_experts).fill(0);m.assignments.forEach(n=>{n.experts.forEach(a=>e[a]++),n.allProbs.forEach((a,x)=>{o[x]+=a})}),o.forEach((n,a)=>{o[a]/=A});const d=e.map(n=>n/(A*_.num_experts_per_tok)),s=_.aux_loss_alpha*_.n_routed_experts*d.reduce((n,a,x)=>n+a*o[x],0),p=Math.max(...e,1);return{counts:e,maxCount:p,auxLoss:s}},[m,w]);return t.jsxs(t.Fragment,{children:[t.jsx("h2",{children:"5. 前馈网络 & MoE"}),t.jsxs("p",{className:"desc",children:["每个 Transformer Block 中，Attention 之后是前馈网络 (FFN)，对每个 token 独立做非线性变换。 MiniMind 使用 SwiGLU 结构：",t.jsx("code",{children:"output = silu(x @ W_gate) * (x @ W_up) @ W_down"}),"。 可通过 ",t.jsx("code",{children:"use_moe=True"})," 切换为 MoE（混合专家）架构——多个 Expert FFN 由 Router 动态选择 top-k 激活。",t.jsx("br",{}),t.jsxs("small",{style:{color:"var(--fg2)"},children:["关联源码：",t.jsx("code",{children:"model/model_minimind.py:216"})," ",t.jsx("code",{children:"class FeedForward"})," | ",t.jsx("code",{children:":232"})," ",t.jsx("code",{children:"class MoEGate"})," | ",t.jsx("code",{children:":288"})," ",t.jsx("code",{children:"class MOEFeedForward"})]})]}),t.jsxs(G,{title:"SwiGLU 数据流",children:[t.jsx("p",{style:{marginBottom:10,fontSize:"0.9rem",color:"var(--fg2)"},children:"SwiGLU 是 FFN 的核心激活机制，通过门控选择性地传递信息。数据流为：输入 x（512维）同时送入两个分支——gate_proj 产生门控信号经 SiLU 激活，up_proj 产生候选特征，两者逐元素相乘后由 down_proj 降维回 512。 中间维度 1408 ≈ round_up(512 × 8/3, 64)，兼顾表达力和效率。下图展示了完整的数据流向："}),t.jsx("svg",{width:"100%",height:200,viewBox:"0 0 700 200",dangerouslySetInnerHTML:{__html:z}}),t.jsxs("div",{className:"silu-demo",style:{marginTop:12},children:[t.jsxs("div",{children:[t.jsx("div",{className:"label",children:"SiLU(x) = x · σ(x) 函数图像"}),t.jsx("canvas",{ref:N})]}),t.jsxs("div",{style:{fontSize:"0.85rem",color:"var(--fg2)",maxWidth:300},children:[t.jsx("p",{children:t.jsx("strong",{children:"为什么用 SwiGLU？"})}),t.jsx("p",{children:"相比 ReLU，SiLU 平滑且非单调，门控机制让网络可以选择性地传递信息，实验证明效果更好。"})]})]}),t.jsx(W,{title:"对照源码：model/model_minimind.py:216-229",code:`class FeedForward(nn.Module):
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
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))`})]}),t.jsxs(G,{title:"MoE 路由动画",children:[t.jsx("p",{style:{marginBottom:10,fontSize:"0.9rem",color:"var(--fg2)"},children:"MoE（混合专家）用多个独立的 FFN（专家）替代单个 FFN。门控网络 W_gate 对每个 token 计算各专家的 softmax 概率， 选取 top-2 专家进行计算，并用归一化权重加权求和。这样模型参数更多但每个 token 只激活部分参数，兼顾容量与效率。"}),t.jsx("p",{style:{marginBottom:10,fontSize:"0.9rem",color:"var(--fg2)"},children:'点击"运行路由"按钮，动画展示 8 个 token 被随机路由到 4 个专家的过程。连线粗细代表门控权重大小， 每个 token 同时送给 2 个专家处理，最后还会加上共享专家（紫色）的输出。每次点击会重新随机采样门控分数。'}),t.jsxs("div",{style:{display:"flex",gap:8,marginBottom:10},children:[t.jsx("button",{className:"btn primary",onClick:R,children:"▶ 运行路由"}),t.jsx("button",{className:"btn",onClick:P,children:"重置"})]}),t.jsx("canvas",{ref:c}),t.jsx(W,{title:"对照源码：model/model_minimind.py:232-349",code:`class MoEGate(nn.Module):
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
        y = y + self.shared_experts(identity)`})]}),t.jsxs(G,{title:"负载均衡",children:[t.jsx("p",{style:{marginBottom:10,fontSize:"0.9rem",color:"var(--fg2)"},children:'MoE 的常见问题是"专家塌缩"——门控网络可能学会把所有 token 都发给少数专家，导致其他专家从不被训练。 辅助损失 aux_loss 通过惩罚不均衡的路由分布来解决这个问题。先运行上方的路由动画后， 左侧柱状图会显示各专家实际收到的 token 数，右侧显示计算出的 aux_loss 值。 当所有专家均匀分配时，aux_loss 达到最小值。'}),t.jsxs("div",{className:"viz-grid",children:[t.jsxs("div",{children:[t.jsx("div",{className:"label",children:"各专家收到的 token 数"}),t.jsx("div",{style:{height:140,display:"flex",alignItems:"flex-end",gap:8,paddingBottom:20,position:"relative"},children:Array.from({length:_.n_routed_experts},(e,o)=>{const d=(h==null?void 0:h.counts[o])??0,s=(h==null?void 0:h.maxCount)??1;return t.jsxs("div",{style:{flex:1,display:"flex",flexDirection:"column",alignItems:"center",gap:4},children:[t.jsx("div",{style:{width:"100%",height:h?d/s*100+"px":"4px",background:L[o],borderRadius:"4px 4px 0 0",transition:"height 0.5s",minHeight:4}}),t.jsxs("div",{style:{fontSize:"0.75rem",color:"var(--fg2)"},children:["E",o,": ",d]})]},o)})})]}),t.jsxs("div",{children:[t.jsx("div",{className:"label",children:"aux_loss = α × Σ(fᵢ × Pᵢ)"}),t.jsx("div",{style:{fontSize:"1.5rem",color:"var(--accent)",fontFamily:"monospace",padding:"20px 0"},children:h?h.auxLoss.toFixed(6):"0.0000"}),t.jsxs("p",{style:{fontSize:"0.85rem",color:"var(--fg2)"},children:["α=0.01, fᵢ=专家i实际频率, Pᵢ=专家i平均门控概率。",t.jsx("br",{}),"当所有专家均衡时，aux_loss 最小。"]})]})]})]})]})}export{Q as default};
