import { useState, useRef, useEffect, useCallback, useMemo } from 'react';
import Card from '../components/Card';
import SourcePanel from '../components/SourcePanel';
import { useTheme } from '../context/ThemeContext';

type ViewMode = 'project' | 'model' | 'pipeline';

/* ------------------------------------------------------------------ */
/*  CONSTANTS                                                         */
/* ------------------------------------------------------------------ */

const PROJECT_TREE = [
  { path: 'model/', desc: '模型定义（MiniMindConfig + MiniMindForCausalLM）', color: '#3b82f6', indent: 0 },
  { path: '  model_minimind.py', desc: '完整模型架构：Config、RMSNorm、Attention、FFN、MoE、Block、Model、CausalLM', color: '#60a5fa', indent: 1 },
  { path: '  tokenizer.json', desc: 'BPE 分词器词表（vocab_size=6400）', color: '#60a5fa', indent: 1 },
  { path: 'trainer/', desc: '9 个训练脚本，每个对应一个训练阶段', color: '#10b981', indent: 0 },
  { path: '  trainer_utils.py', desc: '公共基础：init_model()、init_distributed_mode()、get_lr()、lm_checkpoint()', color: '#34d399', indent: 1 },
  { path: '  train_pretrain.py', desc: '预训练：在原始文本上学习 next token prediction', color: '#34d399', indent: 1 },
  { path: '  train_full_sft.py', desc: '全参数 SFT：在对话数据上微调，只对 assistant 回复计算损失', color: '#34d399', indent: 1 },
  { path: '  train_lora.py', desc: 'LoRA 微调：冻结主干，只训练低秩适配器（rank=16）', color: '#34d399', indent: 1 },
  { path: '  train_dpo.py', desc: 'DPO 对齐：通过 chosen/rejected 对比优化偏好', color: '#34d399', indent: 1 },
  { path: '  train_grpo.py', desc: 'GRPO：组相对策略优化，在线 RL 采样', color: '#34d399', indent: 1 },
  { path: 'dataset/', desc: '数据集加载器', color: '#f59e0b', indent: 0 },
  { path: '  lm_dataset.py', desc: '4 个 Dataset 类：PretrainDataset、SFTDataset、DPODataset、RLAIFDataset', color: '#fbbf24', indent: 1 },
  { path: 'scripts/', desc: '工具脚本', color: '#8b5cf6', indent: 0 },
  { path: '  convert_to_hf.py', desc: 'Dense 模型转 HuggingFace Llama 格式', color: '#a78bfa', indent: 1 },
  { path: '  convert_model.py', desc: 'MoE 模型转 HuggingFace 自定义格式', color: '#a78bfa', indent: 1 },
  { path: '  run_all_npu.sh', desc: '昇腾 NPU 一键训练 + 部署编排脚本', color: '#a78bfa', indent: 1 },
  { path: '  run_train_npu.sh', desc: 'NPU 8 卡分布式训练启动器', color: '#a78bfa', indent: 1 },
  { path: 'out/', desc: '训练产物目录', color: '#ef4444', indent: 0 },
  { path: '  {stage}_{size}.pth', desc: '各阶段权重文件，如 pretrain_512.pth、full_sft_512.pth', color: '#f87171', indent: 1 },
];

const MODEL_LAYERS = [
  { name: 'embed_tokens', cls: 'nn.Embedding', shape: '(6400, 512)', desc: 'Token ID → 向量，与 lm_head 共享权重', color: '#3b82f6' },
  { name: 'dropout', cls: 'nn.Dropout', shape: '—', desc: '训练时随机丢弃，推理时关闭', color: '#64748b' },
  { name: 'layers[0..7]', cls: 'MiniMindBlock ×8', shape: '—', desc: '8 个 Transformer Block，每个包含 Attention + FFN', color: '#8b5cf6', isBlock: true },
  { name: '  input_layernorm', cls: 'RMSNorm', shape: '(512,)', desc: 'Attention 前的归一化：x / RMS(x) × γ', color: '#10b981' },
  { name: '  self_attn', cls: 'Attention', shape: '—', desc: 'GQA 注意力：Q[8头] K[2头] V[2头] + RoPE + KV Cache', color: '#f59e0b' },
  { name: '    q_proj', cls: 'Linear', shape: '(512, 512)', desc: '8 个 Q 头，每头 64 维', color: '#fbbf24' },
  { name: '    k_proj', cls: 'Linear', shape: '(512, 128)', desc: '2 个 KV 头，通过 repeat_kv 扩展到 8 头', color: '#fbbf24' },
  { name: '    v_proj', cls: 'Linear', shape: '(512, 128)', desc: '2 个 KV 头，与 K 共享头数配置', color: '#fbbf24' },
  { name: '    o_proj', cls: 'Linear', shape: '(512, 512)', desc: '多头拼接后投影回 hidden_size', color: '#fbbf24' },
  { name: '  post_attention_layernorm', cls: 'RMSNorm', shape: '(512,)', desc: 'FFN 前的归一化', color: '#10b981' },
  { name: '  mlp (Dense)', cls: 'FeedForward', shape: '—', desc: 'SwiGLU：down(SiLU(gate(x)) ⊙ up(x))', color: '#ef4444' },
  { name: '    gate_proj', cls: 'Linear', shape: '(512, 1408)', desc: 'SiLU 激活的门控分支', color: '#f87171' },
  { name: '    up_proj', cls: 'Linear', shape: '(512, 1408)', desc: '与 gate 逐元素相乘的分支', color: '#f87171' },
  { name: '    down_proj', cls: 'Linear', shape: '(1408, 512)', desc: '投影回 hidden_size', color: '#f87171' },
  { name: 'norm', cls: 'RMSNorm', shape: '(512,)', desc: '最终归一化，在 lm_head 前', color: '#10b981' },
  { name: 'lm_head', cls: 'Linear', shape: '(512, 6400)', desc: '投影到词表空间，与 embed_tokens 共享权重（tie_word_embeddings）', color: '#06b6d4' },
];

const PIPELINE_STAGES = [
  { name: 'Pretrain', script: 'train_pretrain.py', data: 'PretrainDataset', weight_in: '—', weight_out: 'pretrain_512.pth', lr: '5e-4', desc: '在大规模文本上学习语言模式，损失函数 CrossEntropy(next token prediction)，所有 token 都参与训练', color: '#3b82f6' },
  { name: 'Full SFT', script: 'train_full_sft.py', data: 'SFTDataset', weight_in: 'pretrain_512.pth', weight_out: 'full_sft_512.pth', lr: '1e-6', desc: '在对话数据上微调，通过 labels=-100 掩码只对 assistant 回复计算损失，学习指令跟随能力', color: '#10b981' },
  { name: 'LoRA', script: 'train_lora.py', data: 'SFTDataset', weight_in: 'full_sft_512.pth', weight_out: 'lora_512.pth', lr: '5e-4', desc: '冻结主干参数，只训练插入到 Q/V 投影的低秩矩阵（rank=16），可训练参数仅 ~0.5M', color: '#f59e0b' },
  { name: 'DPO', script: 'train_dpo.py', data: 'DPODataset', weight_in: 'full_sft_512.pth', weight_out: 'dpo_512.pth', lr: '4e-8', desc: '直接偏好优化：给定 chosen/rejected 回复对，调整概率使好回复似然更高，β=0.1 控制惩罚力度', color: '#ef4444' },
  { name: 'GRPO', script: 'train_grpo.py', data: 'RLAIFDataset', weight_in: 'full_sft_512.pth', weight_out: 'grpo_512.pth', lr: '2e-6', desc: '组相对策略优化：采样多个回复，用组内相对奖励替代 critic，无需训练奖励模型', color: '#8b5cf6' },
];

// ---- Animated generation demo data ----
const GEN_INPUT = '你好';
const GEN_OUTPUT_CHARS = ['你', '好', '，', '我', '是', 'Mini', 'Mind', '！'];
const GEN_PROBS = [
  [{ t: '你', p: 0.82 }, { t: '大', p: 0.08 }, { t: '我', p: 0.05 }, { t: '世', p: 0.03 }, { t: '请', p: 0.02 }],
  [{ t: '好', p: 0.91 }, { t: '们', p: 0.04 }, { t: '的', p: 0.03 }, { t: '是', p: 0.01 }, { t: '呀', p: 0.01 }],
  [{ t: '，', p: 0.75 }, { t: '！', p: 0.12 }, { t: '。', p: 0.08 }, { t: '~', p: 0.03 }, { t: '呀', p: 0.02 }],
  [{ t: '我', p: 0.88 }, { t: '很', p: 0.06 }, { t: '请', p: 0.03 }, { t: '有', p: 0.02 }, { t: '你', p: 0.01 }],
  [{ t: '是', p: 0.93 }, { t: '叫', p: 0.04 }, { t: '的', p: 0.02 }, { t: '来', p: 0.005 }, { t: '要', p: 0.005 }],
  [{ t: 'Mini', p: 0.85 }, { t: '一个', p: 0.07 }, { t: '你的', p: 0.04 }, { t: 'AI', p: 0.03 }, { t: '小', p: 0.01 }],
  [{ t: 'Mind', p: 0.97 }, { t: 'Bot', p: 0.015 }, { t: 'LLM', p: 0.01 }, { t: 'AI', p: 0.003 }, { t: '!', p: 0.002 }],
  [{ t: '！', p: 0.78 }, { t: '。', p: 0.12 }, { t: '~', p: 0.06 }, { t: '呢', p: 0.02 }, { t: '哦', p: 0.02 }],
];

// ---- Animated data journey steps ----
interface JourneyStep {
  title: string;
  analogy: string;
  detail: string;
  shape: string;
  color: string;
  code: string;
}
const JOURNEY_STEPS: JourneyStep[] = [
  {
    title: '输入文本',
    analogy: '就像你在聊天框里打了一句话',
    detail: '用户输入 "你好" 两个汉字，这是模型的原始输入。但计算机不认识汉字，所以需要先把文字变成数字。',
    shape: '"你好"',
    color: '#64748b',
    code: 'input_text = "你好"',
  },
  {
    title: '分词 → Token IDs',
    analogy: '像查字典一样，把每个字找到对应的编号',
    detail: '分词器（BPE Tokenizer）把文字拆成 token，并从词表（vocab_size=6400）中查找对应的整数 ID。"你" → 868，"好" → 1059。',
    shape: '[868, 1059]',
    color: '#8b5cf6',
    code: 'token_ids = tokenizer.encode("你好")  # → [868, 1059]',
  },
  {
    title: 'Embedding 查表',
    analogy: '像给每个编号发一张"名片"——一组 512 个数字来描述它的"个性"',
    detail: 'Embedding 层是一个 [6400, 512] 的大表格。用 token ID 作为行号，查出对应的 512 维向量。每个 token 从此有了丰富的数值表示。',
    shape: '[1, 2, 512]',
    color: '#3b82f6',
    code: 'hidden = embed_tokens(token_ids)  # [1,2] → [1,2,512]',
  },
  {
    title: 'RoPE 位置编码',
    analogy: '给每个 token 盖一个"座位号"章，让模型知道谁在前谁在后',
    detail: '相同的字在句子不同位置含义可能不同。RoPE（旋转位置编码）通过对向量施加基于位置的旋转变换，将位置信息融入每个 token 的表示中。',
    shape: 'cos/sin: [2, 64]',
    color: '#f59e0b',
    code: 'cos, sin = precompute_freqs_cis(dim=64, end=seq_len)',
  },
  {
    title: 'Attention（注意力）',
    analogy: '"你"和"好"互相看一眼，决定对方跟自己有多相关',
    detail: '每个 token 生成 Query（问题）、Key（标签）、Value（内容），通过 Q×K 的点积计算注意力分数，再用分数加权求和 V。这样每个 token 都能从上下文中获取信息。MiniMind 用 GQA：8 个 Q 头共享 2 个 KV 头。',
    shape: '[1, 2, 512]',
    color: '#8b5cf6',
    code: 'attn_output = softmax(Q @ K.T / √64) @ V  # GQA: 8Q, 2KV',
  },
  {
    title: '残差连接 ①',
    analogy: '把注意力的结果和原始输入加在一起——"别忘了我原来是谁"',
    detail: '残差连接是深度网络的关键技巧：将注意力的输出与输入直接相加（x + attention(x)），防止深层网络中信息丢失，同时让梯度更容易传播。',
    shape: '[1, 2, 512]',
    color: '#3b82f6',
    code: 'hidden = hidden + self_attn(layernorm(hidden))',
  },
  {
    title: 'FFN（前馈网络）',
    analogy: '每个 token 独立"思考"——先展开想法，再压缩总结',
    detail: 'FFN 是一个两层的全连接网络，中间维度扩大到 1408（512 的 2.75 倍），用 SwiGLU 激活函数进行非线性变换。这一步让每个 token 独立做"深度加工"，与 Attention 的"交流"形成互补。',
    shape: '[1, 2, 512]',
    color: '#ef4444',
    code: 'ffn_out = down_proj(SiLU(gate_proj(x)) * up_proj(x))',
  },
  {
    title: '残差连接 ② → 重复 ×8',
    analogy: '以上步骤是一个"思考回合"，模型要思考 8 回合才给出答案',
    detail: 'FFN 输出同样加上残差连接，然后把整个 Attention + FFN 的过程重复 8 次（8 个 TransformerBlock）。每一层理解不同层次的信息：浅层理解语法，深层理解语义。',
    shape: '[1, 2, 512]',
    color: '#10b981',
    code: 'for block in layers:  # 8 layers\n  hidden = block(hidden)  # Attn + FFN + 2×residual',
  },
  {
    title: 'LM Head 投影',
    analogy: '最后一步：从 512 维"想法"转换成对 6400 个候选词的打分',
    detail: 'LM Head 是一个 Linear(512 → 6400) 的线性层，将最后一个 token 的隐藏状态投影到词表空间。输出 6400 个 logits（原始分数），每个对应一个候选 token。注意：LM Head 的权重和 Embedding 层共享（tie_word_embeddings）。',
    shape: '[1, 2, 6400]',
    color: '#06b6d4',
    code: 'logits = lm_head(norm(hidden))  # 与 embed_tokens 共享权重',
  },
  {
    title: 'Softmax → 采样',
    analogy: '把打分变成概率，然后"掷骰子"选出下一个字',
    detail: 'Softmax 将 logits 转换为概率分布（所有值变为 0~1 之间且总和为 1）。然后根据 top-p / top-k 等策略从概率分布中随机采样一个 token 作为输出。这就是为什么 LLM 每次回答可能略有不同——因为有随机采样。',
    shape: '→ token_id',
    color: '#10b981',
    code: 'probs = softmax(logits[:, -1, :] / temperature)\nnext_token = sample(probs, top_k=50)',
  },
];

// ---- Training loop steps ----
const TRAIN_STEPS = [
  { label: '输入', desc: '取一条训练数据：问题 + 正确回答', icon: '📥', color: '#3b82f6' },
  { label: '前向传播', desc: '模型根据当前参数预测每个位置的下一个 token', icon: '➡️', color: '#8b5cf6' },
  { label: '计算损失', desc: '对比预测和正确答案，计算差距（CrossEntropy Loss）', icon: '📊', color: '#ef4444' },
  { label: '反向传播', desc: '计算每个参数对损失的贡献（梯度），找到"改进方向"', icon: '⬅️', color: '#f59e0b' },
  { label: '更新参数', desc: '沿梯度方向微调所有参数（optimizer.step()），让模型下次预测更准', icon: '🔄', color: '#10b981' },
];

/* ------------------------------------------------------------------ */
/*  COMPONENT                                                         */
/* ------------------------------------------------------------------ */

export default function ArchitectureSection() {
  const { isDark } = useTheme();
  const [view, setView] = useState<ViewMode>('model');
  const [selectedLayer, setSelectedLayer] = useState<number | null>(null);
  const [selectedStage, setSelectedStage] = useState<number | null>(null);

  // ---- Generation demo state ----
  const [genStep, setGenStep] = useState(-1);
  const [genPlaying, setGenPlaying] = useState(false);
  const genTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // ---- Journey animation state ----
  const [journeyStep, setJourneyStep] = useState(-1);
  const [journeyPlaying, setJourneyPlaying] = useState(false);
  const journeyTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // ---- Training animation state ----
  const [trainStep, setTrainStep] = useState(-1);
  const [trainEpoch, setTrainEpoch] = useState(0);
  const [trainPlaying, setTrainPlaying] = useState(false);
  const trainTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const trainPredictions = useMemo(() => [
    { epoch: 0, pred: '上海', correct: '北京', loss: 2.8, confidence: 0.15 },
    { epoch: 1, pred: '南京', correct: '北京', loss: 1.9, confidence: 0.35 },
    { epoch: 2, pred: '北京', correct: '北京', loss: 0.4, confidence: 0.85 },
  ], []);

  const fg = isDark ? '#e2e8f0' : '#1a1a2e';
  const fg2 = isDark ? '#94a3b8' : '#555';
  const border = isDark ? '#334155' : '#e2e8f0';

  // Cleanup timers
  useEffect(() => {
    return () => {
      if (genTimerRef.current) clearInterval(genTimerRef.current);
      if (journeyTimerRef.current) clearInterval(journeyTimerRef.current);
      if (trainTimerRef.current) clearInterval(trainTimerRef.current);
    };
  }, []);

  /* ---------- Generation demo controls ---------- */
  const stopGen = useCallback(() => {
    if (genTimerRef.current) { clearInterval(genTimerRef.current); genTimerRef.current = null; }
    setGenPlaying(false);
  }, []);

  const playGen = useCallback(() => {
    if (genTimerRef.current) { stopGen(); return; }
    setGenStep(-1);
    setGenPlaying(true);
    let step = -1;
    genTimerRef.current = setInterval(() => {
      step++;
      if (step >= GEN_OUTPUT_CHARS.length) { stopGen(); return; }
      setGenStep(step);
    }, 600);
  }, [stopGen]);

  const resetGen = useCallback(() => { stopGen(); setGenStep(-1); }, [stopGen]);

  /* ---------- Journey animation controls ---------- */
  const stopJourney = useCallback(() => {
    if (journeyTimerRef.current) { clearInterval(journeyTimerRef.current); journeyTimerRef.current = null; }
    setJourneyPlaying(false);
  }, []);

  const playJourney = useCallback(() => {
    if (journeyTimerRef.current) { stopJourney(); return; }
    setJourneyPlaying(true);
    journeyTimerRef.current = setInterval(() => {
      setJourneyStep(prev => {
        if (prev >= JOURNEY_STEPS.length - 1) { stopJourney(); return prev; }
        return prev + 1;
      });
    }, 2000);
  }, [stopJourney]);

  /* ---------- Training animation controls ---------- */
  const stopTrain = useCallback(() => {
    if (trainTimerRef.current) { clearInterval(trainTimerRef.current); trainTimerRef.current = null; }
    setTrainPlaying(false);
  }, []);

  const playTrain = useCallback(() => {
    if (trainTimerRef.current) { stopTrain(); return; }
    setTrainStep(-1);
    setTrainEpoch(0);
    setTrainPlaying(true);
    let s = -1;
    let e = 0;
    trainTimerRef.current = setInterval(() => {
      s++;
      if (s >= TRAIN_STEPS.length) {
        s = 0;
        e++;
        if (e >= trainPredictions.length) { stopTrain(); return; }
        setTrainEpoch(e);
      }
      setTrainStep(s);
    }, 800);
  }, [stopTrain, trainPredictions.length]);

  /* ---------- Model architecture SVG ---------- */
  const modelSvg = useMemo(() => {
    const bg1 = isDark ? '#1e3a5f' : '#dbeafe';
    const bg2 = isDark ? '#312e81' : '#ede9fe';
    const bg3 = isDark ? '#1a3a2a' : '#d1fae5';
    const bg4 = isDark ? '#4a1d1d' : '#fee2e2';
    const bg5 = isDark ? '#1e293b' : '#f1f5f9';

    let svg = `<defs>
      <marker id="arrArch" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto">
        <path d="M0,0 L10,5 L0,10 Z" fill="${fg2}"/>
      </marker>
    </defs>`;

    svg += `<text x="360" y="18" text-anchor="middle" fill="${fg}" font-size="13" font-weight="bold">MiniMindForCausalLM 模型结构</text>`;
    svg += `<rect x="280" y="30" width="160" height="28" rx="6" fill="${bg5}" stroke="${fg2}" stroke-width="1.5"/>`;
    svg += `<text x="360" y="49" text-anchor="middle" fill="${fg}" font-size="11">input_ids [B, S]</text>`;
    svg += `<line x1="360" y1="58" x2="360" y2="72" stroke="${fg2}" stroke-width="1" marker-end="url(#arrArch)"/>`;
    svg += `<rect x="250" y="75" width="220" height="32" rx="6" fill="${bg1}" stroke="#3b82f6" stroke-width="2"/>`;
    svg += `<text x="360" y="95" text-anchor="middle" fill="${fg}" font-size="11" font-weight="bold">embed_tokens  Embedding(6400, 512)</text>`;
    svg += `<line x1="360" y1="107" x2="360" y2="121" stroke="${fg2}" stroke-width="1" marker-end="url(#arrArch)"/>`;
    svg += `<rect x="300" y="124" width="120" height="24" rx="6" fill="${bg5}" stroke="${fg2}" stroke-width="1"/>`;
    svg += `<text x="360" y="140" text-anchor="middle" fill="${fg2}" font-size="10">Dropout</text>`;
    svg += `<line x1="360" y1="148" x2="360" y2="168" stroke="${fg2}" stroke-width="1" marker-end="url(#arrArch)"/>`;
    svg += `<rect x="100" y="170" width="520" height="230" rx="10" fill="none" stroke="${fg2}" stroke-width="1.5" stroke-dasharray="6,3"/>`;
    svg += `<text x="110" y="188" fill="${fg2}" font-size="10">MiniMindBlock × 8</text>`;
    svg += `<rect x="280" y="195" width="160" height="26" rx="5" fill="${bg3}" stroke="#10b981" stroke-width="1.5"/>`;
    svg += `<text x="360" y="212" text-anchor="middle" fill="${fg}" font-size="10">input_layernorm (RMSNorm)</text>`;
    svg += `<line x1="360" y1="221" x2="360" y2="233" stroke="${fg2}" stroke-width="1" marker-end="url(#arrArch)"/>`;
    svg += `<rect x="200" y="236" width="320" height="50" rx="6" fill="${bg2}" stroke="#8b5cf6" stroke-width="2"/>`;
    svg += `<text x="360" y="255" text-anchor="middle" fill="${fg}" font-size="11" font-weight="bold">Attention (GQA)</text>`;
    svg += `<text x="360" y="275" text-anchor="middle" fill="${fg2}" font-size="9">Q[8头] × K[2头] → scores → softmax → × V[2头] → concat → O_proj</text>`;
    svg += `<rect x="555" y="236" width="55" height="22" rx="4" fill="${isDark ? '#4a3a1d' : '#fef3c7'}" stroke="#f59e0b" stroke-width="1"/>`;
    svg += `<text x="582" y="251" text-anchor="middle" fill="${fg}" font-size="9">+ RoPE</text>`;
    svg += `<path d="M 195 195 L 165 195 L 165 300 L 195 300" fill="none" stroke="#3b82f6" stroke-width="1.5" stroke-dasharray="4,2"/>`;
    svg += `<text x="140" y="250" fill="#3b82f6" font-size="8" text-anchor="middle">+残差</text>`;
    svg += `<line x1="360" y1="286" x2="360" y2="298" stroke="${fg2}" stroke-width="1" marker-end="url(#arrArch)"/>`;
    svg += `<circle cx="360" cy="304" r="8" fill="${bg1}" stroke="#3b82f6" stroke-width="1.5"/>`;
    svg += `<text x="360" y="308" text-anchor="middle" fill="#3b82f6" font-size="11" font-weight="bold">+</text>`;
    svg += `<line x1="360" y1="312" x2="360" y2="322" stroke="${fg2}" stroke-width="1" marker-end="url(#arrArch)"/>`;
    svg += `<rect x="280" y="325" width="160" height="26" rx="5" fill="${bg3}" stroke="#10b981" stroke-width="1.5"/>`;
    svg += `<text x="360" y="342" text-anchor="middle" fill="${fg}" font-size="10">post_attn_layernorm (RMSNorm)</text>`;
    svg += `<line x1="360" y1="351" x2="360" y2="363" stroke="${fg2}" stroke-width="1" marker-end="url(#arrArch)"/>`;
    svg += `<rect x="230" y="366" width="260" height="26" rx="6" fill="${bg4}" stroke="#ef4444" stroke-width="2"/>`;
    svg += `<text x="360" y="383" text-anchor="middle" fill="${fg}" font-size="10" font-weight="bold">FFN: down(SiLU(gate(x)) ⊙ up(x))</text>`;
    svg += `<path d="M 525 325 L 555 325 L 555 392 L 525 392" fill="none" stroke="#ef4444" stroke-width="1.5" stroke-dasharray="4,2"/>`;
    svg += `<text x="580" y="360" fill="#ef4444" font-size="8" text-anchor="middle">+残差</text>`;
    svg += `<line x1="360" y1="392" x2="360" y2="410" stroke="${fg2}" stroke-width="1" marker-end="url(#arrArch)"/>`;
    svg += `<rect x="280" y="415" width="160" height="26" rx="5" fill="${bg3}" stroke="#10b981" stroke-width="1.5"/>`;
    svg += `<text x="360" y="432" text-anchor="middle" fill="${fg}" font-size="10">norm (RMSNorm)</text>`;
    svg += `<line x1="360" y1="441" x2="360" y2="455" stroke="${fg2}" stroke-width="1" marker-end="url(#arrArch)"/>`;
    svg += `<rect x="250" y="458" width="220" height="32" rx="6" fill="${isDark ? '#164e63' : '#cffafe'}" stroke="#06b6d4" stroke-width="2"/>`;
    svg += `<text x="360" y="478" text-anchor="middle" fill="${fg}" font-size="11" font-weight="bold">lm_head  Linear(512, 6400)</text>`;
    svg += `<path d="M 475 90 Q 510 90 510 250 Q 510 470 475 474" fill="none" stroke="#06b6d4" stroke-width="1" stroke-dasharray="3,2"/>`;
    svg += `<text x="520" y="280" fill="#06b6d4" font-size="8" transform="rotate(90, 520, 280)">tie_word_embeddings</text>`;
    svg += `<line x1="360" y1="490" x2="360" y2="504" stroke="${fg2}" stroke-width="1" marker-end="url(#arrArch)"/>`;
    svg += `<rect x="280" y="507" width="160" height="28" rx="6" fill="${bg5}" stroke="${fg2}" stroke-width="1.5"/>`;
    svg += `<text x="360" y="526" text-anchor="middle" fill="${fg}" font-size="11">logits [B, S, 6400]</text>`;
    return svg;
  }, [isDark, fg, fg2]);

  /* ---------- Pipeline SVG ---------- */
  const pipelineSvg = useMemo(() => {
    let svg = `<defs><marker id="arrPL" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0 L10,5 L0,10 Z" fill="${fg2}"/></marker></defs>`;
    const xs = [20, 155, 290, 425, 560];
    PIPELINE_STAGES.forEach((s, i) => {
      const x = xs[i];
      const isSelected = selectedStage === i;
      const opacity = isSelected ? 0.4 : 0.15;
      svg += `<rect x="${x}" y="20" width="120" height="45" rx="8" fill="${s.color}" opacity="${opacity}" stroke="${s.color}" stroke-width="2" style="cursor:pointer" data-stage="${i}"/>`;
      svg += `<text x="${x + 60}" y="40" text-anchor="middle" fill="${fg}" font-size="11" font-weight="bold" style="pointer-events:none">${s.name}</text>`;
      svg += `<text x="${x + 60}" y="55" text-anchor="middle" fill="${fg2}" font-size="8" style="pointer-events:none">lr=${s.lr}</text>`;
      if (i < PIPELINE_STAGES.length - 1) {
        svg += `<line x1="${x + 120}" y1="42" x2="${xs[i + 1]}" y2="42" stroke="${fg2}" stroke-width="1.5" marker-end="url(#arrPL)"/>`;
      }
    });
    svg += `<text x="350" y="82" text-anchor="middle" fill="${fg2}" font-size="9">每阶段产出 out/{stage}_512.pth，下一阶段通过 --from_weight 加载</text>`;
    return svg;
  }, [isDark, fg, fg2, selectedStage]);

  /* ---------- Journey SVG (animated) ---------- */
  const journeySvg = useMemo(() => {
    const bg = isDark ? '#0f172a' : '#f8fafc';
    let svg = `<defs>
      <marker id="arrJ" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto">
        <path d="M0,0 L10,5 L0,10 Z" fill="${fg2}"/>
      </marker>
    </defs>`;

    const boxes = JOURNEY_STEPS.map((s, i) => ({
      y: 10 + i * 42,
      h: 32,
      label: s.title,
      color: s.color,
    }));

    boxes.forEach((b, i) => {
      const isActive = i === journeyStep;
      const isDone = i < journeyStep;
      const opacity = isActive ? 0.35 : isDone ? 0.2 : 0.06;
      const strokeW = isActive ? 2.5 : 1.5;
      const dash = !isActive && !isDone ? 'stroke-dasharray="4,2"' : '';

      svg += `<rect x="20" y="${b.y}" width="260" height="${b.h}" rx="6" fill="${b.color}" opacity="${opacity}" stroke="${b.color}" stroke-width="${strokeW}" ${dash} style="cursor:pointer" data-jstep="${i}"/>`;
      svg += `<text x="150" y="${b.y + b.h / 2 + 5}" text-anchor="middle" fill="${isActive ? fg : fg2}" font-size="${isActive ? 12 : 10.5}" ${isActive ? 'font-weight="bold"' : ''} style="pointer-events:none">${i}. ${b.label}</text>`;

      if (isActive) {
        svg += `<circle cx="10" cy="${b.y + b.h / 2}" r="5" fill="${b.color}"><animate attributeName="r" values="4;7;4" dur="1s" repeatCount="indefinite"/></circle>`;
      }
      if (isDone) {
        svg += `<text x="288" y="${b.y + b.h / 2 + 4}" fill="${b.color}" font-size="12">✓</text>`;
      }
      if (i < boxes.length - 1) {
        svg += `<line x1="150" y1="${b.y + b.h}" x2="150" y2="${boxes[i + 1].y}" stroke="${fg2}" stroke-width="1" marker-end="url(#arrJ)" opacity="0.4"/>`;
      }
    });

    // Block bracket for steps 4-7
    if (journeyStep >= 4 && journeyStep <= 7) {
      svg += `<rect x="5" y="${boxes[4].y - 3}" width="296" height="${boxes[7].y + boxes[7].h - boxes[4].y + 6}" rx="8" fill="none" stroke="${isDark ? '#818cf8' : '#6366f1'}" stroke-width="1.5" stroke-dasharray="6,3" opacity="0.5"/>`;
      svg += `<text x="303" y="${(boxes[4].y + boxes[7].y + boxes[7].h) / 2}" fill="${isDark ? '#818cf8' : '#6366f1'}" font-size="9" opacity="0.7">×8 层</text>`;
    }

    // Background for svg
    svg = `<rect width="320" height="${boxes[boxes.length - 1].y + boxes[boxes.length - 1].h + 10}" fill="${bg}" rx="8" opacity="0.5"/>` + svg;

    return svg;
  }, [isDark, fg, fg2, journeyStep]);

  const viewToggle = (
    <div style={{ display: 'flex', gap: 0, marginBottom: 16, borderRadius: 'var(--radius)', overflow: 'hidden', border: '2px solid var(--accent)', width: 'fit-content' }}>
      {([
        ['model', '模型架构'],
        ['pipeline', '训练 Pipeline'],
        ['project', '项目结构'],
      ] as [ViewMode, string][]).map(([key, label]) => (
        <button
          key={key}
          className="btn"
          style={{
            background: view === key ? 'var(--accent)' : 'transparent',
            color: view === key ? '#fff' : 'var(--accent)',
            border: 'none',
            borderRadius: 0,
            borderLeft: key !== 'model' ? '2px solid var(--accent)' : 'none',
            fontWeight: 'bold',
            padding: '6px 16px',
          }}
          onClick={() => setView(key)}
        >
          {label}
        </button>
      ))}
    </div>
  );

  const curJourney = journeyStep >= 0 ? JOURNEY_STEPS[journeyStep] : null;
  const curTrain = trainPredictions[trainEpoch] || trainPredictions[0];

  return (
    <>
      <h2>0. LLM 架构总览</h2>
      <p className="desc">
        MiniMind 是一个完整的 LLM 训练框架，核心模型 <code>MiniMindForCausalLM</code> 采用 Decoder-only Transformer 架构，
        支持 Dense（标准 FFN）和 MoE（混合专家）两种模式。训练分 5 个阶段串联执行，每阶段产出独立权重文件。
        <br/>
        <small style={{ color: 'var(--fg2)' }}>
          关联源码：<code>model/model_minimind.py:427</code> <code>MiniMindForCausalLM</code> |
          <code>:376</code> <code>MiniMindModel</code> |
          <code>:352</code> <code>MiniMindBlock</code> |
          <code>:8</code> <code>MiniMindConfig</code>
        </small>
      </p>

      {/* ============================================================ */}
      {/*  INTRO: "LLM 是什么" — 动画演示                               */}
      {/* ============================================================ */}
      <Card title="LLM 是什么？— 看它一个字一个字生成回答">
        <p style={{ marginBottom: 12, fontSize: '0.9rem', color: 'var(--fg2)' }}>
          大语言模型（LLM）的核心能力就一件事：<strong style={{ color: 'var(--accent)' }}>根据已有的文字，预测下一个最可能的字</strong>。
          它并不是一次性输出整句话，而是像打字一样一个 token 一个 token 地"蹦"出来。
          每一步，模型都会给词表中所有 6400 个候选 token 打分，然后选出最可能的那个拼接到末尾，再预测下一个……如此循环。
          点击「开始生成」观看这个过程：
        </p>
        <div style={{ display: 'flex', gap: 8, marginBottom: 12 }}>
          <button className="btn primary" onClick={playGen}>{genPlaying ? '暂停' : '开始生成'}</button>
          <button className="btn" onClick={resetGen}>重置</button>
        </div>

        {/* Chat-like display */}
        <div style={{ padding: 16, background: 'var(--bg)', borderRadius: 'var(--radius)', border: `1px solid ${border}`, marginBottom: 12 }}>
          <div style={{ display: 'flex', gap: 8, marginBottom: 12 }}>
            <div style={{ width: 32, height: 32, borderRadius: '50%', background: '#3b82f6', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#fff', fontSize: '0.8rem', flexShrink: 0 }}>你</div>
            <div style={{ padding: '6px 12px', borderRadius: '12px', background: isDark ? '#1e3a5f' : '#dbeafe', fontSize: '0.9rem' }}>{GEN_INPUT}</div>
          </div>
          <div style={{ display: 'flex', gap: 8 }}>
            <div style={{ width: 32, height: 32, borderRadius: '50%', background: '#10b981', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#fff', fontSize: '0.7rem', flexShrink: 0 }}>AI</div>
            <div style={{ padding: '6px 12px', borderRadius: '12px', background: isDark ? '#1a3a2a' : '#d1fae5', fontSize: '0.9rem', minHeight: 32, display: 'flex', alignItems: 'center', gap: 2, flexWrap: 'wrap' }}>
              {genStep < 0 ? (
                <span style={{ color: 'var(--fg3)' }}>等待生成...</span>
              ) : (
                GEN_OUTPUT_CHARS.slice(0, genStep + 1).map((ch, i) => (
                  <span
                    key={i}
                    style={{
                      display: 'inline-block',
                      fontWeight: i === genStep ? 'bold' : 'normal',
                      color: i === genStep ? 'var(--accent)' : fg,
                      transition: 'all 0.3s',
                      borderBottom: i === genStep ? '2px solid var(--accent)' : '2px solid transparent',
                    }}
                  >
                    {ch}
                  </span>
                ))
              )}
              {genPlaying && genStep < GEN_OUTPUT_CHARS.length - 1 && (
                <span style={{ color: 'var(--accent)', animation: 'pulse 0.6s ease infinite' }}>▌</span>
              )}
            </div>
          </div>
        </div>

        {/* Probability distribution for current step */}
        {genStep >= 0 && genStep < GEN_PROBS.length && (
          <div style={{ padding: 12, background: 'var(--bg)', borderRadius: 'var(--radius)', border: `1px solid ${border}` }}>
            <div className="label" style={{ marginBottom: 6 }}>
              第 {genStep + 1} 步预测 — 模型给 6400 个候选 token 打分，这是 top-5：
            </div>
            {GEN_PROBS[genStep].map((item, i) => (
              <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 3 }}>
                <span className="token-box" style={{
                  background: i === 0 ? 'var(--accent)' : 'var(--bg3)',
                  color: i === 0 ? '#fff' : 'var(--fg2)',
                  minWidth: 40,
                  textAlign: 'center',
                  fontWeight: i === 0 ? 'bold' : 'normal',
                  border: i === 0 ? '2px solid var(--accent)' : '1px solid var(--border)',
                }}>
                  {item.t}
                </span>
                <div style={{ flex: 1, height: 16, background: 'var(--bg3)', borderRadius: 4, overflow: 'hidden' }}>
                  <div style={{
                    height: '100%',
                    width: `${item.p * 100}%`,
                    background: i === 0 ? 'var(--accent)' : (isDark ? '#475569' : '#cbd5e1'),
                    borderRadius: 4,
                    transition: 'width 0.5s',
                  }} />
                </div>
                <span style={{ fontFamily: 'monospace', fontSize: '0.78rem', color: i === 0 ? 'var(--accent)' : 'var(--fg2)', minWidth: 42 }}>
                  {(item.p * 100).toFixed(1)}%
                </span>
              </div>
            ))}
            <div style={{ marginTop: 6, fontSize: '0.78rem', color: 'var(--fg3)' }}>
              选中概率最高的 <strong style={{ color: 'var(--accent)' }}>「{GEN_PROBS[genStep][0].t}」</strong> → 拼接到末尾 → 继续预测下一个
            </div>
          </div>
        )}

        <SourcePanel
          title="对照源码：eval_llm.py:80-85 (模型推理生成)"
          code={`# 调用 model.generate() 逐 token 生成回答
generated_ids = model.generate(
    inputs=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    max_new_tokens=args.max_new_tokens,
    do_sample=True,             # 启用随机采样（非贪心）
    streamer=streamer,          # TextStreamer 实现打字机效果
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
    top_p=args.top_p,           # nucleus 采样阈值 (0.85)
    temperature=args.temperature # 温度控制随机性 (0.85)
)
# generate() 内部循环：
# 1. logits = lm_head(model(input_ids))   → 6400 维打分
# 2. probs = softmax(logits / temperature) → 概率分布
# 3. next_token = sample(probs, top_p)     → 采样一个 token
# 4. input_ids = concat(input_ids, next_token) → 拼接
# 5. 重复直到遇到 eos_token_id 或达到 max_new_tokens`}
        />
      </Card>

      {viewToggle}

      {/* ============================================================ */}
      {/*  MODEL VIEW                                                   */}
      {/* ============================================================ */}
      {view === 'model' && (
        <>
          {/* ---- Animated Data Journey ---- */}
          <Card title="数据在模型中的旅程 — 逐步动画">
            <p style={{ marginBottom: 10, fontSize: '0.9rem', color: 'var(--fg2)' }}>
              当你输入 "你好" 时，数据从头到尾穿过整个模型。下面用 10 个步骤展示这段旅程。
              每一步都有<strong>一句大白话类比</strong>帮你理解，再配合代码和 shape 变化。
              点击「自动播放」或手动点击左侧步骤。
            </p>
            <div style={{ display: 'flex', gap: 8, marginBottom: 10, flexWrap: 'wrap' }}>
              <button className="btn" onClick={() => setJourneyStep(prev => Math.max(prev - 1, 0))}>◀ 上一步</button>
              <button className="btn primary" onClick={() => setJourneyStep(prev => Math.min(prev + 1, JOURNEY_STEPS.length - 1))}>下一步 ▶</button>
              <button className="btn" onClick={playJourney}>{journeyPlaying ? '⏸ 暂停' : '▶ 自动播放'}</button>
              <button className="btn" onClick={() => { stopJourney(); setJourneyStep(-1); }}>重置</button>
            </div>

            {/* Step dots */}
            <div className="step-indicator">
              {JOURNEY_STEPS.map((s, i) => (
                <div
                  key={i}
                  className={`step-dot${i === journeyStep ? ' active' : ''}${i < journeyStep ? ' done' : ''}`}
                  onClick={() => setJourneyStep(i)}
                  title={s.title}
                >
                  {i}
                </div>
              ))}
            </div>

            <div className="viz-grid">
              {/* Left: animated flow */}
              <div>
                <svg
                  width="100%"
                  height={440}
                  viewBox="0 0 320 440"
                  onClick={(e) => {
                    const t = e.target as SVGElement;
                    const r = t.closest('rect[data-jstep]');
                    if (r) setJourneyStep(parseInt(r.getAttribute('data-jstep')!));
                  }}
                  dangerouslySetInnerHTML={{ __html: journeySvg }}
                />
              </div>

              {/* Right: step details */}
              <div>
                {curJourney ? (
                  <>
                    <div style={{ fontSize: '1.1rem', fontWeight: 700, color: curJourney.color, marginBottom: 6 }}>
                      {journeyStep}. {curJourney.title}
                    </div>

                    {/* Analogy callout */}
                    <div style={{
                      padding: '10px 14px',
                      background: isDark ? '#1e293b' : '#fffbeb',
                      border: `2px solid ${curJourney.color}`,
                      borderRadius: 'var(--radius)',
                      marginBottom: 10,
                      fontSize: '0.92rem',
                    }}>
                      <span style={{ marginRight: 6 }}>💡</span>
                      <strong>大白话：</strong>{curJourney.analogy}
                    </div>

                    <div className="label">具体过程</div>
                    <p style={{ fontSize: '0.88rem', color: 'var(--fg2)', marginBottom: 10 }}>{curJourney.detail}</p>

                    <div className="label">数据形状</div>
                    <div className="shape-badge" style={{ marginBottom: 10, borderColor: curJourney.color, color: curJourney.color }}>{curJourney.shape}</div>

                    <div className="label">对应代码</div>
                    <pre style={{ background: 'var(--bg)', padding: 10, borderRadius: 'var(--radius)', border: `1px solid ${border}`, fontSize: '0.82rem', margin: 0, overflowX: 'auto' }}>
                      <code>{curJourney.code}</code>
                    </pre>

                    {/* Visual mini-demo for specific steps */}
                    {journeyStep === 1 && (
                      <div style={{ marginTop: 12, display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
                        {['你', '好'].map((ch, i) => (
                          <div key={i} style={{ textAlign: 'center' }}>
                            <span className="token-box" style={{ background: '#8b5cf6', color: '#fff' }}>{ch}</span>
                            <div style={{ fontSize: '0.75rem', color: '#8b5cf6', fontFamily: 'monospace' }}>→ {i === 0 ? 868 : 1059}</div>
                          </div>
                        ))}
                      </div>
                    )}
                    {journeyStep === 2 && (
                      <div style={{ marginTop: 12, display: 'flex', alignItems: 'center', gap: 6 }}>
                        <span style={{ fontSize: '0.82rem', color: '#3b82f6' }}>ID 868 →</span>
                        <div style={{ display: 'flex', gap: 1 }}>
                          {Array.from({ length: 20 }).map((_, i) => (
                            <div key={i} style={{ width: 6, height: 16, background: '#3b82f6', opacity: 0.3 + Math.random() * 0.7, borderRadius: 1 }} />
                          ))}
                        </div>
                        <span style={{ fontSize: '0.75rem', color: 'var(--fg3)' }}>...×512 维</span>
                      </div>
                    )}
                    {journeyStep === 4 && (
                      <div style={{ marginTop: 12 }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
                          <div style={{ textAlign: 'center' }}>
                            <span className="token-box" style={{ background: '#8b5cf6', color: '#fff' }}>你</span>
                          </div>
                          <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                            <div style={{ fontSize: '0.75rem', color: '#8b5cf6' }}>← 注意力 0.65 →</div>
                            <div style={{ height: 3, background: '#8b5cf6', borderRadius: 2, width: `${0.65 * 100}px` }} />
                          </div>
                          <div style={{ textAlign: 'center' }}>
                            <span className="token-box" style={{ background: '#8b5cf6', color: '#fff' }}>好</span>
                          </div>
                        </div>
                        <div style={{ fontSize: '0.78rem', color: 'var(--fg3)', marginTop: 6 }}>
                          "你"和"好"互相看对方，计算相关度分数
                        </div>
                      </div>
                    )}
                    {journeyStep === 9 && (
                      <div style={{ marginTop: 12 }}>
                        {[{ t: '，', p: 0.75 }, { t: '！', p: 0.12 }, { t: '。', p: 0.08 }].map((item, i) => (
                          <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 2 }}>
                            <span className="token-box" style={{ background: i === 0 ? '#10b981' : 'var(--bg3)', color: i === 0 ? '#fff' : 'var(--fg2)', minWidth: 28, textAlign: 'center' }}>{item.t}</span>
                            <div style={{ width: `${item.p * 120}px`, height: 12, background: i === 0 ? '#10b981' : (isDark ? '#475569' : '#cbd5e1'), borderRadius: 3 }} />
                            <span style={{ fontSize: '0.75rem', fontFamily: 'monospace', color: 'var(--fg2)' }}>{(item.p * 100).toFixed(0)}%</span>
                          </div>
                        ))}
                      </div>
                    )}
                  </>
                ) : (
                  <div style={{ color: 'var(--fg3)', fontSize: '0.9rem', padding: 20 }}>
                    点击「下一步」或左侧流程图开始旅程
                  </div>
                )}
              </div>
            </div>

            <SourcePanel
              title="对照源码：model/model_minimind.py:392-468 (完整前向传播)"
              code={`class MiniMindModel(nn.Module):
    def forward(self, input_ids, ...):
        # Step 2-3: Token ID → Embedding 向量 → Dropout
        hidden_states = self.dropout(self.embed_tokens(input_ids))
        # Step 3: 预计算 RoPE 位置编码 (cos/sin)
        position_embeddings = (freqs_cos[start:end], freqs_sin[start:end])
        # Step 4-7: 8 个 TransformerBlock (每个: Attn+残差 → FFN+残差)
        for layer in self.layers:
            hidden_states, _ = layer(hidden_states, position_embeddings, ...)
        # Step 8: 最终 RMSNorm
        return self.norm(hidden_states)  # [B, S, 512]

class MiniMindForCausalLM(PreTrainedModel):
    def forward(self, input_ids, labels=None, ...):
        hidden, _, aux_loss = self.model(input_ids, ...)
        # Step 9: LM Head 投影到词表
        logits = self.lm_head(hidden)  # [B, S, 6400]
        # Step 10: 训练时计算 loss，推理时 softmax → 采样
        if labels is not None:
            loss = cross_entropy(logits[:-1], labels[1:])`}
            />
          </Card>

          {/* ---- Static full structure diagram ---- */}
          <Card title="MiniMindForCausalLM 完整结构图">
            <p style={{ marginBottom: 10, fontSize: '0.9rem', color: 'var(--fg2)' }}>
              模型由 <code>MiniMindModel</code>（主干）和 <code>lm_head</code>（输出头）组成。
              <code>lm_head</code> 与 <code>embed_tokens</code> 共享权重（<code>tie_word_embeddings</code>），减少参数量。
            </p>
            <svg width="100%" height={550} viewBox="0 0 720 550" dangerouslySetInnerHTML={{ __html: modelSvg }} />
            <SourcePanel
              title="对照源码：model/model_minimind.py:427-435 (MiniMindForCausalLM.__init__)"
              code={`class MiniMindForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = MiniMindConfig

    def __init__(self, config: MiniMindConfig = None):
        self.config = config or MiniMindConfig()
        super().__init__(self.config)
        # 模型主干：Embedding + N × TransformerBlock + RMSNorm
        self.model = MiniMindModel(self.config)
        # 输出投影头：hidden_size → vocab_size
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        # 关键：Embedding 和 LM Head 共享权重（tie_word_embeddings）
        self.model.embed_tokens.weight = self.lm_head.weight`}
            />
          </Card>

          <Card title="逐层参数详解">
            <p style={{ marginBottom: 10, fontSize: '0.9rem', color: 'var(--fg2)' }}>
              点击任意一层查看详细说明。以 <code>hidden_size=512</code>（26M 模型）为例展示各层的 shape。
            </p>
            <div style={{ background: 'var(--bg)', borderRadius: 'var(--radius)', border: `1px solid ${border}`, overflow: 'hidden' }}>
              {MODEL_LAYERS.map((layer, i) => (
                <div
                  key={i}
                  style={{
                    display: 'grid',
                    gridTemplateColumns: '220px 160px 120px 1fr',
                    gap: 0,
                    borderBottom: i < MODEL_LAYERS.length - 1 ? `1px solid ${border}` : 'none',
                    fontSize: '0.82rem',
                    cursor: 'pointer',
                    background: selectedLayer === i
                      ? (isDark ? '#1e293b' : '#f1f5f9')
                      : layer.isBlock ? (isDark ? '#0f172a' : '#f8fafc') : 'transparent',
                  }}
                  onClick={() => setSelectedLayer(selectedLayer === i ? null : i)}
                >
                  <div style={{ padding: '5px 10px', fontFamily: 'monospace', color: layer.color, fontWeight: layer.isBlock ? 'bold' : 'normal' }}>{layer.name}</div>
                  <div style={{ padding: '5px 10px', color: 'var(--fg2)', borderLeft: `1px solid ${border}` }}>{layer.cls}</div>
                  <div style={{ padding: '5px 10px', fontFamily: 'monospace', color: 'var(--fg2)', borderLeft: `1px solid ${border}`, fontSize: '0.78rem' }}>{layer.shape}</div>
                  <div style={{ padding: '5px 10px', color: 'var(--fg2)', borderLeft: `1px solid ${border}` }}>{layer.desc}</div>
                </div>
              ))}
            </div>
            <SourcePanel
              title="对照源码：model/model_minimind.py:376-390 (MiniMindModel.__init__)"
              code={`class MiniMindModel(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.num_hidden_layers = config.num_hidden_layers
        # Token ID → 向量 (6400 × 512)，与 lm_head 共享
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        # N 个 TransformerBlock（每个含 Attention + FFN）
        self.layers = nn.ModuleList([
            MiniMindBlock(l, config) for l in range(self.num_hidden_layers)
        ])
        # 最终归一化
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # 预计算 RoPE 位置编码 (cos/sin)
        freqs_cos, freqs_sin = precompute_freqs_cis(
            dim=config.hidden_size // config.num_attention_heads,
            end=config.max_position_embeddings,
            rope_base=config.rope_theta
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)`}
            />
          </Card>

          <Card title="Dense vs MoE：FFN 层的差异">
            <p style={{ marginBottom: 10, fontSize: '0.9rem', color: 'var(--fg2)' }}>
              Dense 和 MoE 模型的唯一区别在于 <code>MiniMindBlock.mlp</code>：
              Dense 使用单个 <code>FeedForward</code>，MoE 使用 <code>MOEFeedForward</code>（Gate + 多个 Expert + 共享 Expert）。
            </p>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
              <div>
                <div className="label" style={{ color: '#3b82f6' }}>Dense FFN</div>
                <pre style={{ background: 'var(--bg)', padding: 10, borderRadius: 'var(--radius)', border: '1px solid var(--border)', fontSize: '0.8rem', margin: 0 }}>
                  <code>{`# model_minimind.py:228
FeedForward.forward(x):
  gate = SiLU(gate_proj(x))  # [B,S,1408]
  up   = up_proj(x)          # [B,S,1408]
  out  = down_proj(gate ⊙ up) # [B,S,512]
  return dropout(out)`}</code>
                </pre>
              </div>
              <div>
                <div className="label" style={{ color: '#8b5cf6' }}>MoE FFN</div>
                <pre style={{ background: 'var(--bg)', padding: 10, borderRadius: 'var(--radius)', border: '1px solid var(--border)', fontSize: '0.8rem', margin: 0 }}>
                  <code>{`# model_minimind.py:303
MOEFeedForward.forward(x):
  idx, weight, aux = gate(x) # top-2 专家
  for expert_i in experts:   # 4 个路由专家
    y[idx==i] = expert_i(x[idx==i])
  y = Σ(y × weight)          # 加权求和
  y += shared_expert(x)      # +1 共享专家
  return y`}</code>
                </pre>
              </div>
            </div>
            <SourcePanel
              title="对照源码：model/model_minimind.py:352-373 (MiniMindBlock)"
              code={`class MiniMindBlock(nn.Module):
    def __init__(self, layer_id, config):
        self.self_attn = Attention(config)
        self.input_layernorm = RMSNorm(config.hidden_size)
        self.post_attention_layernorm = RMSNorm(config.hidden_size)
        # Dense 用 FeedForward，MoE 用 MOEFeedForward
        self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)

    def forward(self, hidden_states, position_embeddings, ...):
        residual = hidden_states
        hidden_states, present = self.self_attn(
            self.input_layernorm(hidden_states), position_embeddings, ...
        )
        hidden_states += residual
        hidden_states = hidden_states + self.mlp(
            self.post_attention_layernorm(hidden_states)
        )
        return hidden_states, present`}
            />
          </Card>

          <Card title="关键设计：参数共享与 GQA">
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
              <div style={{ padding: 12, background: 'var(--bg)', border: '2px solid #06b6d4', borderRadius: 'var(--radius)' }}>
                <div className="label" style={{ color: '#06b6d4' }}>Tie Word Embeddings</div>
                <p style={{ fontSize: '0.85rem', color: 'var(--fg2)', margin: 0 }}>
                  <code>lm_head.weight = embed_tokens.weight</code><br/>
                  输入 Embedding 和输出投影共享同一个 [6400, 512] 的权重矩阵。
                  减少 ~3.3M 参数（6400×512），对小模型影响显著。
                </p>
              </div>
              <div style={{ padding: 12, background: 'var(--bg)', border: '2px solid #f59e0b', borderRadius: 'var(--radius)' }}>
                <div className="label" style={{ color: '#f59e0b' }}>Grouped-Query Attention (GQA)</div>
                <p style={{ fontSize: '0.85rem', color: 'var(--fg2)', margin: 0 }}>
                  Q 头数 = <code>num_attention_heads = 8</code><br/>
                  KV 头数 = <code>num_key_value_heads = 2</code><br/>
                  每 4 个 Q 头共享 1 组 KV，KV Cache 大小减少 75%。
                </p>
              </div>
            </div>
            <SourcePanel
              title="对照源码：model/model_minimind.py:150-162 & 435 (Attention + 权重共享)"
              code={`class Attention(nn.Module):
    def __init__(self, args: MiniMindConfig):
        # GQA: Q 头数 (8) 和 KV 头数 (2) 可以不同
        self.n_local_heads = args.num_attention_heads      # Q 头数 = 8
        self.n_local_kv_heads = args.num_key_value_heads   # KV 头数 = 2
        self.n_rep = self.n_local_heads // self.n_local_kv_heads  # 每组 4 个 Q 共享 1 个 KV
        self.head_dim = args.hidden_size // args.num_attention_heads  # 64

        # Q: [512, 8×64=512]  K: [512, 2×64=128]  V: [512, 2×64=128]
        self.q_proj = nn.Linear(args.hidden_size, args.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(args.num_attention_heads * self.head_dim, args.hidden_size, bias=False)

# ---- 权重共享 (MiniMindForCausalLM.__init__) ----
# embed_tokens 和 lm_head 共享同一个 [6400, 512] 权重矩阵
self.model.embed_tokens.weight = self.lm_head.weight  # tie_word_embeddings`}
            />
          </Card>
        </>
      )}

      {/* ============================================================ */}
      {/*  PIPELINE VIEW                                                */}
      {/* ============================================================ */}
      {view === 'pipeline' && (
        <>
          {/* ---- Training Loop Animation ---- */}
          <Card title="训练就是不断纠错 — 动画演示">
            <p style={{ marginBottom: 10, fontSize: '0.9rem', color: 'var(--fg2)' }}>
              训练 LLM 就像教一个学生做题：给它看问题和正确答案，让它尝试回答，答错了就告诉它错在哪，
              它调整自己的"记忆"（参数），下次争取答对。这个过程反复数千万次，模型就越来越聪明。
              点击「开始训练」观看这个循环：
            </p>
            <div style={{ display: 'flex', gap: 8, marginBottom: 12 }}>
              <button className="btn primary" onClick={playTrain}>{trainPlaying ? '暂停' : '开始训练'}</button>
              <button className="btn" onClick={() => { stopTrain(); setTrainStep(-1); setTrainEpoch(0); }}>重置</button>
            </div>

            {/* Training loop steps */}
            <div style={{ display: 'flex', gap: 4, marginBottom: 16, flexWrap: 'wrap' }}>
              {TRAIN_STEPS.map((s, i) => (
                <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                  <div style={{
                    padding: '8px 14px',
                    borderRadius: 'var(--radius)',
                    border: `2px solid ${s.color}`,
                    background: trainStep === i ? s.color : 'transparent',
                    color: trainStep === i ? '#fff' : s.color,
                    fontSize: '0.85rem',
                    fontWeight: trainStep === i ? 'bold' : 'normal',
                    transition: 'all 0.3s',
                    textAlign: 'center',
                    minWidth: 80,
                  }}>
                    <div>{s.icon} {s.label}</div>
                  </div>
                  {i < TRAIN_STEPS.length - 1 && (
                    <span style={{ color: 'var(--fg3)', fontSize: '1.2rem' }}>→</span>
                  )}
                </div>
              ))}
              <span style={{ color: 'var(--fg3)', fontSize: '1.2rem', display: 'flex', alignItems: 'center' }}>↩</span>
            </div>

            {/* Current step description */}
            {trainStep >= 0 && (
              <div style={{ padding: 10, background: 'var(--bg)', border: `2px solid ${TRAIN_STEPS[trainStep].color}`, borderRadius: 'var(--radius)', marginBottom: 12, fontSize: '0.88rem', color: 'var(--fg2)' }}>
                <strong style={{ color: TRAIN_STEPS[trainStep].color }}>{TRAIN_STEPS[trainStep].label}</strong>：{TRAIN_STEPS[trainStep].desc}
              </div>
            )}

            {/* Live training dashboard */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 12, marginBottom: 12 }}>
              <div style={{ padding: 12, background: 'var(--bg)', borderRadius: 'var(--radius)', border: `1px solid ${border}`, textAlign: 'center' }}>
                <div className="label">训练轮次</div>
                <div style={{ fontSize: '1.4rem', fontWeight: 'bold', color: 'var(--accent)' }}>{trainEpoch + 1} / 3</div>
              </div>
              <div style={{ padding: 12, background: 'var(--bg)', borderRadius: 'var(--radius)', border: `1px solid ${border}`, textAlign: 'center' }}>
                <div className="label">模型预测</div>
                <div style={{ fontSize: '1.2rem', fontWeight: 'bold', color: curTrain.pred === curTrain.correct ? '#10b981' : '#ef4444' }}>
                  {curTrain.pred} {curTrain.pred === curTrain.correct ? '✓' : '✗'}
                </div>
              </div>
              <div style={{ padding: 12, background: 'var(--bg)', borderRadius: 'var(--radius)', border: `1px solid ${border}`, textAlign: 'center' }}>
                <div className="label">损失值 (Loss)</div>
                <div style={{ fontSize: '1.2rem', fontWeight: 'bold', color: curTrain.loss < 1 ? '#10b981' : curTrain.loss < 2 ? '#f59e0b' : '#ef4444' }}>
                  {curTrain.loss.toFixed(1)}
                </div>
              </div>
            </div>

            {/* Training progress illustration */}
            <div style={{ padding: 12, background: 'var(--bg)', borderRadius: 'var(--radius)', border: `1px solid ${border}` }}>
              <div className="label" style={{ marginBottom: 8 }}>训练进度 — 模型从"乱猜"到"答对"</div>
              <div style={{ display: 'flex', gap: 12, alignItems: 'flex-end' }}>
                {trainPredictions.map((p, i) => (
                  <div key={i} style={{
                    flex: 1,
                    padding: 10,
                    borderRadius: 'var(--radius)',
                    border: `2px solid ${i <= trainEpoch ? (p.pred === p.correct ? '#10b981' : '#f59e0b') : border}`,
                    background: i === trainEpoch ? (isDark ? '#1e293b' : '#f1f5f9') : 'transparent',
                    opacity: i <= trainEpoch ? 1 : 0.4,
                    transition: 'all 0.5s',
                  }}>
                    <div style={{ fontSize: '0.78rem', color: 'var(--fg3)', marginBottom: 4 }}>轮次 {i + 1}</div>
                    <div style={{ fontSize: '0.85rem', marginBottom: 4 }}>
                      问：中国的首都是？
                    </div>
                    <div style={{ fontSize: '0.92rem', fontWeight: 'bold', color: p.pred === p.correct ? '#10b981' : '#ef4444' }}>
                      答：{p.pred} {p.pred === p.correct ? '✓' : '✗'}
                    </div>
                    <div style={{ marginTop: 6 }}>
                      <div style={{ fontSize: '0.72rem', color: 'var(--fg3)' }}>置信度</div>
                      <div style={{ height: 6, background: 'var(--bg3)', borderRadius: 3, overflow: 'hidden' }}>
                        <div style={{
                          height: '100%',
                          width: `${p.confidence * 100}%`,
                          background: p.pred === p.correct ? '#10b981' : '#f59e0b',
                          borderRadius: 3,
                          transition: 'width 0.8s',
                        }} />
                      </div>
                      <div style={{ fontSize: '0.72rem', color: 'var(--fg2)', fontFamily: 'monospace' }}>{(p.confidence * 100).toFixed(0)}%</div>
                    </div>
                  </div>
                ))}
              </div>
              <div style={{ marginTop: 8, fontSize: '0.82rem', color: 'var(--fg3)' }}>
                正确答案：<strong style={{ color: '#10b981' }}>北京</strong> — 每轮训练后，模型的预测越来越接近正确答案，损失值（loss）不断下降
              </div>
            </div>

            <SourcePanel
              title="对照源码：trainer/train_full_sft.py:23-68 (训练循环)"
              code={`def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    for step, (input_ids, labels) in enumerate(loader):
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        # 动态调整学习率（余弦退火）
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)

        # Step 1-2: 前向传播 — 模型预测 + 计算损失
        with autocast_ctx:  # 混合精度 (float16/bfloat16)
            res = model(input_ids, labels=labels)  # → CrossEntropy Loss
            loss = res.loss + res.aux_loss         # MoE 加辅助损失
            loss = loss / args.accumulation_steps  # 梯度累积

        # Step 3: 反向传播 — 计算每个参数的梯度
        scaler.scale(loss).backward()

        # Step 4-5: 更新参数（每 accumulation_steps 步执行一次）
        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)  # 梯度裁剪
            scaler.step(optimizer)   # 用梯度更新参数
            scaler.update()
            optimizer.zero_grad(set_to_none=True)  # 清空梯度

        # 定期保存权重
        if step % args.save_interval == 0:
            torch.save(model.state_dict(), "out/full_sft_512.pth")`}
            />
          </Card>

          <Card title="训练阶段流程图">
            <p style={{ marginBottom: 10, fontSize: '0.9rem', color: 'var(--fg2)' }}>
              MiniMind 的完整训练分 5 个阶段，点击各阶段查看详情。
              每个阶段产出 <code>out/{'{stage}'}_{'{hidden_size}'}.pth</code> 权重文件，
              下一阶段通过 <code>--from_weight</code> 加载继续训练。
            </p>
            <svg
              width="100%"
              height={95}
              viewBox="0 0 700 95"
              onClick={(e) => {
                const target = e.target as SVGElement;
                const rect = target.closest('rect[data-stage]');
                if (rect) setSelectedStage(parseInt(rect.getAttribute('data-stage')!));
              }}
              dangerouslySetInnerHTML={{ __html: pipelineSvg }}
            />
            {selectedStage !== null && (
              <div style={{ marginTop: 10, padding: 12, background: 'var(--bg)', border: `2px solid ${PIPELINE_STAGES[selectedStage].color}`, borderRadius: 'var(--radius)' }}>
                <strong style={{ color: PIPELINE_STAGES[selectedStage].color }}>{PIPELINE_STAGES[selectedStage].name}</strong>
                <span style={{ fontSize: '0.85rem', color: 'var(--fg2)' }}> — {PIPELINE_STAGES[selectedStage].desc}</span>
              </div>
            )}
            <SourcePanel
              title="对照源码：训练脚本启动命令（trainer/*.py）"
              code={`# 1. 预训练：在大规模文本上学习语言模式
python trainer/train_pretrain.py --epochs 1 --batch_size 32 --learning_rate 5e-4

# 2. 全参数 SFT：在对话数据上微调
python trainer/train_full_sft.py --epochs 2 --batch_size 16 --learning_rate 1e-6 \\
    --from_weight pretrain  # 加载预训练权重

# 3. LoRA 微调：冻结主干，只训练低秩适配器
python trainer/train_lora.py --epochs 50 --batch_size 32 --learning_rate 5e-4 \\
    --from_weight full_sft  # 加载 SFT 权重

# 4. DPO 偏好对齐：通过 chosen/rejected 对比优化
python trainer/train_dpo.py --epochs 1 --batch_size 4 --learning_rate 4e-8 \\
    --from_weight full_sft

# 5. GRPO 在线强化学习：组相对策略优化
python trainer/train_grpo.py --epochs 1 --batch_size 2 --learning_rate 2e-6 \\
    --from_weight full_sft`}
            />
          </Card>

          <Card title="各阶段配置对比">
            <div style={{ background: 'var(--bg)', borderRadius: 'var(--radius)', border: `1px solid ${border}`, overflow: 'hidden' }}>
              <div style={{ display: 'grid', gridTemplateColumns: '80px 150px 120px 130px 140px 80px', gap: 0, borderBottom: `1px solid ${border}`, fontSize: '0.8rem', fontWeight: 'bold', color: fg }}>
                {['阶段', '脚本', '数据集', '加载权重', '输出权重', '学习率'].map((h, i) => (
                  <div key={i} style={{ padding: '6px 8px', background: isDark ? '#1e293b' : '#f1f5f9', borderLeft: i > 0 ? `1px solid ${border}` : 'none' }}>{h}</div>
                ))}
              </div>
              {PIPELINE_STAGES.map((s, i) => (
                <div key={i} style={{ display: 'grid', gridTemplateColumns: '80px 150px 120px 130px 140px 80px', gap: 0, borderBottom: i < PIPELINE_STAGES.length - 1 ? `1px solid ${border}` : 'none', fontSize: '0.8rem' }}>
                  <div style={{ padding: '5px 8px', fontWeight: 'bold', color: s.color }}>{s.name}</div>
                  <div style={{ padding: '5px 8px', borderLeft: `1px solid ${border}`, fontFamily: 'monospace', color: 'var(--fg2)' }}>{s.script}</div>
                  <div style={{ padding: '5px 8px', borderLeft: `1px solid ${border}`, fontFamily: 'monospace', color: 'var(--fg2)' }}>{s.data}</div>
                  <div style={{ padding: '5px 8px', borderLeft: `1px solid ${border}`, fontFamily: 'monospace', color: 'var(--fg2)' }}>{s.weight_in}</div>
                  <div style={{ padding: '5px 8px', borderLeft: `1px solid ${border}`, fontFamily: 'monospace', color: 'var(--fg2)' }}>{s.weight_out}</div>
                  <div style={{ padding: '5px 8px', borderLeft: `1px solid ${border}`, fontFamily: 'monospace', color: 'var(--fg2)' }}>{s.lr}</div>
                </div>
              ))}
            </div>
            <SourcePanel
              title="对照源码：各训练脚本的 argparse 参数（共同模式）"
              code={`# 所有 9 个训练脚本共享的核心参数模式：
parser.add_argument("--save_dir", default="../out")          # 权重保存目录
parser.add_argument("--save_weight", default="full_sft")     # 保存权重前缀名
parser.add_argument("--from_weight", default="pretrain")     # 加载的上一阶段权重
parser.add_argument("--epochs", type=int, default=2)         # 训练轮数
parser.add_argument("--batch_size", type=int, default=16)    # 批大小
parser.add_argument("--learning_rate", type=float, default=1e-6)  # 初始学习率
parser.add_argument("--hidden_size", type=int, default=512)  # 隐藏层维度
parser.add_argument("--num_hidden_layers", type=int, default=8)   # Transformer 层数
parser.add_argument("--use_moe", type=int, default=0)        # 是否使用 MoE
parser.add_argument("--grad_clip", type=float, default=1.0)  # 梯度裁剪阈值
parser.add_argument("--accumulation_steps", type=int, default=1)  # 梯度累积步数

# 权重文件命名规则：out/{save_weight}_{hidden_size}[_moe].pth
# 例如：out/pretrain_512.pth → out/full_sft_512.pth → out/dpo_512.pth`}
            />
          </Card>

          <Card title="权重传递与数据流">
            <p style={{ marginBottom: 10, fontSize: '0.9rem', color: 'var(--fg2)' }}>
              所有训练脚本遵循相同结构：解析参数 → <code>init_distributed_mode()</code> → <code>init_model()</code> → 训练循环 → 保存权重。
            </p>
            <pre style={{ background: 'var(--bg)', padding: 12, borderRadius: 'var(--radius)', border: `1px solid ${border}`, fontSize: '0.82rem', overflowX: 'auto' }}>
              <code>{`# 每个训练脚本的执行流程（以 train_full_sft.py 为例）：

1. args = parse_args()                              # --from_weight pretrain --hidden_size 512
2. init_distributed_mode(args)                       # DDP: NCCL(CUDA) / HCCL(NPU)
3. model, tokenizer = init_model(args)               # 加载 MiniMindConfig + 权重 + 分词器
4. dataset = SFTDataset(args.data_path, tokenizer)   # 构建数据集
5. for epoch in range(args.epochs):
     for step, batch in enumerate(dataloader):
       with autocast():                              # 混合精度 (float16/bfloat16)
         loss = model(input_ids, labels=labels).loss  # 前向传播
       scaler.scale(loss).backward()                 # 反向传播
       scaler.step(optimizer)                        # 更新参数
       lr = get_lr(step, total_steps, args.lr)       # 余弦退火学习率
6. torch.save(model.state_dict(), "out/full_sft_512.pth")  # 保存权重`}</code>
            </pre>
            <SourcePanel
              title="对照源码：trainer/trainer_utils.py:139-165 (init_model)"
              code={`def init_model(args):
    """初始化模型和分词器——所有训练脚本的入口"""
    tokenizer = AutoTokenizer.from_pretrained('./model/')
    config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=args.use_moe,
    )
    model = MiniMindForCausalLM(config)
    moe_path = '_moe' if args.use_moe else ''
    ckpt = f'out/{args.from_weight}_{args.hidden_size}{moe_path}.pth'
    state_dict = torch.load(ckpt, map_location=args.device)
    model.load_state_dict(state_dict, strict=False)
    return model.to(args.device), tokenizer`}
            />
          </Card>
        </>
      )}

      {/* ============================================================ */}
      {/*  PROJECT VIEW                                                 */}
      {/* ============================================================ */}
      {view === 'project' && (
        <>
          <Card title="项目目录结构">
            <p style={{ marginBottom: 10, fontSize: '0.9rem', color: 'var(--fg2)' }}>
              MiniMind 的核心代码分布在 4 个目录中：<code>model/</code>（模型定义）、<code>trainer/</code>（训练脚本）、
              <code>dataset/</code>（数据加载）、<code>scripts/</code>（工具脚本）。
            </p>
            <div style={{ background: 'var(--bg)', borderRadius: 'var(--radius)', border: `1px solid ${border}`, overflow: 'hidden' }}>
              {PROJECT_TREE.map((item, i) => (
                <div
                  key={i}
                  style={{
                    display: 'flex',
                    gap: 12,
                    padding: '5px 10px',
                    paddingLeft: 10 + item.indent * 16,
                    borderBottom: i < PROJECT_TREE.length - 1 ? `1px solid ${border}` : 'none',
                    fontSize: '0.82rem',
                    background: item.indent === 0 ? (isDark ? '#0f172a' : '#f8fafc') : 'transparent',
                  }}
                >
                  <span style={{ fontFamily: 'monospace', color: item.color, minWidth: 200, flexShrink: 0, fontWeight: item.indent === 0 ? 'bold' : 'normal' }}>
                    {item.path}
                  </span>
                  <span style={{ color: 'var(--fg2)' }}>{item.desc}</span>
                </div>
              ))}
            </div>
            <SourcePanel
              title="对照源码：model/model_minimind.py 核心类定义位置"
              code={`# model/model_minimind.py 中的类定义及行号：
#
# Line 8:   class MiniMindConfig(PretrainedConfig)   — 模型超参数配置
# Line 96:  class RMSNorm(nn.Module)                  — 归一化层
# Line 107: precompute_freqs_cis()                    — RoPE 预计算
# Line 150: class Attention(nn.Module)                — GQA 注意力
# Line 216: class FeedForward(nn.Module)              — SwiGLU FFN
# Line 232: class MoEGate(nn.Module)                  — MoE 路由门控
# Line 288: class MOEFeedForward(nn.Module)           — MoE FFN 层
# Line 352: class MiniMindBlock(nn.Module)            — Transformer Block
# Line 376: class MiniMindModel(nn.Module)            — 模型主干
# Line 427: class MiniMindForCausalLM(PreTrainedModel) — 完整 LLM

# trainer/trainer_utils.py 中的公共函数：
# init_model()               — 加载模型和分词器
# init_distributed_mode()    — DDP 初始化 (NCCL/HCCL)
# get_lr()                   — 余弦退火学习率调度
# lm_checkpoint()            — 断点续训保存/加载`}
            />
          </Card>

          <Card title="模块依赖关系">
            <p style={{ marginBottom: 10, fontSize: '0.9rem', color: 'var(--fg2)' }}>
              <code>trainer_utils.py</code> 是被所有 9 个训练脚本 import 的公共模块。
            </p>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
              <div>
                <div className="label">trainer/*.py 的 import 结构</div>
                <pre style={{ background: 'var(--bg)', padding: 10, borderRadius: 'var(--radius)', border: `1px solid ${border}`, fontSize: '0.8rem', margin: 0 }}>
                  <code>{`# 所有训练脚本共同的 import
from trainer.trainer_utils import (
    init_model,              # → model_minimind.py
    init_distributed_mode,   # DDP 初始化
    get_lr,                  # 余弦退火 LR
    lm_checkpoint,           # 断点续训
)
from dataset.lm_dataset import (
    PretrainDataset,   # 预训练数据
    SFTDataset,        # SFT 对话数据
    DPODataset,        # DPO 偏好对
    RLAIFDataset,      # RL 在线采样
)`}</code>
                </pre>
              </div>
              <div>
                <div className="label">scripts/*.py 的 import 结构</div>
                <pre style={{ background: 'var(--bg)', padding: 10, borderRadius: 'var(--radius)', border: `1px solid ${border}`, fontSize: '0.8rem', margin: 0 }}>
                  <code>{`# 转换脚本的 import
from model.model_minimind import (
    MiniMindConfig,          # 模型配置
    MiniMindForCausalLM,     # 完整模型
)

# convert_to_hf.py (Dense → Llama)
# 手动构建 config.json + safetensors

# convert_model.py (MoE → MiniMind-HF)
# 使用 save_pretrained() 保留自定义架构
# + register_for_auto_class() 注册`}</code>
                </pre>
              </div>
            </div>
            <SourcePanel
              title="对照源码：dataset/lm_dataset.py (4 个数据集类)"
              code={`# dataset/lm_dataset.py 提供 4 个 Dataset 类，分别服务不同训练阶段：

class PretrainDataset(Dataset):
    """预训练数据集：读取 JSONL 原始文本，拼接为固定长度序列"""
    # labels = input_ids（所有 token 都参与 loss 计算）

class SFTDataset(Dataset):
    """SFT 对话数据集：使用 chat_template 格式化多轮对话"""
    # labels 中 user 部分设为 -100（不计算 loss）
    # 只对 assistant 回复部分计算 loss

class DPODataset(Dataset):
    """DPO 偏好数据集：每条数据包含 chosen 和 rejected 两个回复"""
    # 用于 Direct Preference Optimization 训练

class RLAIFDataset(Dataset):
    """RLAIF 在线采样数据集：只包含 prompt，回复由模型在线生成"""
    # 用于 PPO / GRPO / SPO 等强化学习训练`}
            />
          </Card>

          <Card title="模型尺寸对比">
            <div style={{ background: 'var(--bg)', borderRadius: 'var(--radius)', border: `1px solid ${border}`, overflow: 'hidden' }}>
              {[
                { config: 'Small', params: '26M', hidden: '512', layers: '8', heads: '8/2', ffn: '1408', moe: '—' },
                { config: 'Base', params: '104M', hidden: '768', layers: '16', heads: '8/2', ffn: '2048', moe: '—' },
                { config: 'MoE', params: '145M', hidden: '768', layers: '16', heads: '8/2', ffn: '2048', moe: '4路由+1共享' },
              ].map((row, i) => (
                <div key={i} style={{ display: 'grid', gridTemplateColumns: '70px 70px 70px 60px 70px 70px 110px', gap: 0, borderBottom: i < 2 ? `1px solid ${border}` : 'none', fontSize: '0.82rem' }}>
                  {[row.config, row.params, row.hidden, row.layers, row.heads, row.ffn, row.moe].map((val, j) => (
                    <div key={j} style={{
                      padding: '5px 8px',
                      borderLeft: j > 0 ? `1px solid ${border}` : 'none',
                      fontWeight: j === 0 ? 'bold' : 'normal',
                      color: j === 0 ? fg : 'var(--fg2)',
                      fontFamily: j > 0 ? 'monospace' : 'inherit',
                    }}>
                      {val}
                    </div>
                  ))}
                </div>
              ))}
            </div>
            <div style={{ marginTop: 6, display: 'flex', gap: 16, flexWrap: 'wrap', fontSize: '0.78rem', color: 'var(--fg2)' }}>
              <span>heads 格式: Q头数/KV头数</span>
              <span>ffn = intermediate_size</span>
              <span>MoE: num_experts_per_tok=2</span>
            </div>
            <SourcePanel
              title="对照源码：model/model_minimind.py:8-78 (MiniMindConfig)"
              code={`class MiniMindConfig(PretrainedConfig):
    model_type = "minimind"

    def __init__(self,
        hidden_size: int = 512,          # Small=512, Base/MoE=768
        intermediate_size: int = None,   # 自动计算: int(hidden_size * 8/3) 对齐到 64
        num_attention_heads: int = 8,    # Q 头数
        num_key_value_heads: int = 2,    # KV 头数 (GQA)
        num_hidden_layers: int = 8,      # Small/MoE=8, Base=16
        vocab_size: int = 6400,          # BPE 词表大小
        rms_norm_eps: float = 1e-05,
        rope_theta: int = 1000000.0,     # RoPE 基频
        flash_attn: bool = True,         # 是否使用 Flash Attention
        # MoE 配置 (use_moe=False 时以下无效)
        use_moe: bool = False,
        num_experts_per_tok: int = 2,    # 每个 token 选 2 个专家
        n_routed_experts: int = 4,       # 路由专家总数
        n_shared_experts: int = 1,       # 共享专家数量
        aux_loss_alpha: float = 0.01,    # 辅助负载均衡损失系数
        ...
    ):
    # intermediate_size 自动计算示例：
    # hidden_size=512 → int(512*8/3)=1365 → 对齐64 → 1408
    # hidden_size=768 → int(768*8/3)=2048 → 对齐64 → 2048`}
            />
          </Card>
        </>
      )}
    </>
  );
}
