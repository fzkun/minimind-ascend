import { useState, useMemo } from 'react';
import Card from '../components/Card';
import SourcePanel from '../components/SourcePanel';
import { useTheme } from '../context/ThemeContext';

type ModelType = 'dense' | 'moe';

interface Step {
  label: string;
  desc: string;
  color: string;
}

const CONVERT_STEPS: Step[] = [
  { label: '加载 .pth', desc: '从 out/ 目录加载 MiniMind 原始 PyTorch 权重', color: '#3b82f6' },
  { label: '构建 config.json', desc: '生成 LlamaForCausalLM 兼容的配置文件', color: '#8b5cf6' },
  { label: '导出 safetensors', desc: '以 float16 精度保存为 model.safetensors', color: '#10b981' },
  { label: '复制 tokenizer', desc: '复制 tokenizer.json 等分词器文件到输出目录', color: '#f59e0b' },
];

const HF_FILES = [
  { name: 'config.json', desc: 'LlamaForCausalLM 模型配置（架构、维度、注意力头数等）' },
  { name: 'model.safetensors', desc: '模型权重（float16，与 embed_tokens 共享 lm_head）' },
  { name: 'tokenizer.json', desc: 'HuggingFace 格式分词器（BPE 词表 + 合并规则）' },
  { name: 'special_tokens_map.json', desc: '特殊标记映射（bos、eos、pad 等）' },
  { name: 'tokenizer_config.json', desc: '分词器配置（chat_template 等）' },
];

export default function DeploySection() {
  const { isDark } = useTheme();
  const [selectedStep, setSelectedStep] = useState<number | null>(null);
  const [modelType, setModelType] = useState<ModelType>('dense');

  const isDense = modelType === 'dense';

  const pipelineSvg = useMemo(() => {
    const fg = isDark ? '#e2e8f0' : '#1a1a2e';
    const fg2 = isDark ? '#94a3b8' : '#555';
    let html = `<defs><marker id="arrDeploy" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0 L10,5 L0,10 Z" fill="${fg2}"/></marker></defs>`;

    // 训练产物
    const pthFile = isDense ? 'full_sft_512.pth' : 'full_sft_768_moe.pth';
    html += `<rect x="10" y="30" width="130" height="50" rx="8" fill="${isDark ? '#1e3a5f' : '#dbeafe'}" stroke="#3b82f6" stroke-width="2"/>`;
    html += `<text x="75" y="50" text-anchor="middle" fill="${fg}" font-size="11" font-weight="bold">MiniMind .pth</text>`;
    html += `<text x="75" y="66" text-anchor="middle" fill="${fg2}" font-size="9">out/${pthFile}</text>`;

    // 箭头 → 转换
    html += `<line x1="140" y1="55" x2="175" y2="55" stroke="${fg2}" stroke-width="1.5" marker-end="url(#arrDeploy)"/>`;

    // 转换脚本
    const scriptName = isDense ? 'convert_to_hf.py' : 'convert_model.py';
    const scriptDesc = isDense ? 'MiniMind → Llama 格式' : 'MiniMind → MiniMind-HF';
    html += `<rect x="180" y="25" width="160" height="60" rx="8" fill="${isDark ? '#312e81' : '#ede9fe'}" stroke="#8b5cf6" stroke-width="2"/>`;
    html += `<text x="260" y="48" text-anchor="middle" fill="${fg}" font-size="11" font-weight="bold">${scriptName}</text>`;
    html += `<text x="260" y="65" text-anchor="middle" fill="${fg2}" font-size="9">${scriptDesc}</text>`;

    // 箭头 → HF 目录
    html += `<line x1="340" y1="55" x2="380" y2="55" stroke="${fg2}" stroke-width="1.5" marker-end="url(#arrDeploy)"/>`;

    // HF 模型目录
    html += `<rect x="385" y="25" width="145" height="60" rx="8" fill="${isDark ? '#1a3a2a' : '#d1fae5'}" stroke="#10b981" stroke-width="2"/>`;
    html += `<text x="457" y="48" text-anchor="middle" fill="${fg}" font-size="11" font-weight="bold">minimind-hf/</text>`;
    html += `<text x="457" y="65" text-anchor="middle" fill="${fg2}" font-size="9">config + safetensors + tokenizer</text>`;

    // 箭头 → vLLM
    html += `<line x1="530" y1="55" x2="565" y2="55" stroke="${fg2}" stroke-width="1.5" marker-end="url(#arrDeploy)"/>`;

    // vLLM 服务
    html += `<rect x="570" y="25" width="140" height="60" rx="8" fill="${isDark ? '#4a1d1d' : '#fee2e2'}" stroke="#ef4444" stroke-width="2"/>`;
    html += `<text x="640" y="48" text-anchor="middle" fill="${fg}" font-size="11" font-weight="bold">vLLM Serve</text>`;
    html += `<text x="640" y="65" text-anchor="middle" fill="${fg2}" font-size="9">OpenAI 兼容 API</text>`;

    return html;
  }, [isDark, isDense]);

  const modelToggle = (
    <div style={{ display: 'flex', gap: 0, marginBottom: 16, borderRadius: 'var(--radius)', overflow: 'hidden', border: '2px solid var(--accent)', width: 'fit-content' }}>
      <button
        className="btn"
        style={{
          background: isDense ? 'var(--accent)' : 'transparent',
          color: isDense ? '#fff' : 'var(--accent)',
          border: 'none',
          borderRadius: 0,
          fontWeight: 'bold',
          padding: '6px 16px',
        }}
        onClick={() => setModelType('dense')}
      >
        Dense 模型
      </button>
      <button
        className="btn"
        style={{
          background: !isDense ? 'var(--accent)' : 'transparent',
          color: !isDense ? '#fff' : 'var(--accent)',
          border: 'none',
          borderRadius: 0,
          borderLeft: '2px solid var(--accent)',
          fontWeight: 'bold',
          padding: '6px 16px',
        }}
        onClick={() => setModelType('moe')}
      >
        MoE 模型
      </button>
    </div>
  );

  return (
    <>
      <h2>8. 模型转换与 vLLM 部署</h2>
      <p className="desc">
        训练产出的 <code>.pth</code> 文件需要转换为 HuggingFace 格式（<code>config.json</code> + <code>model.safetensors</code>），
        再通过 <code>vllm serve ./minimind-hf</code> 启动 OpenAI 兼容的 API 服务。
        Dense 和 MoE 模型的转换路径不同——切换下方按钮查看各自的流程。
        <br/>
        <small style={{ color: 'var(--fg2)' }}>
          关联源码：Dense → <code>scripts/convert_to_hf.py:15</code> <code>convert()</code> | MoE → <code>scripts/convert_model.py:16</code> <code>convert_torch2transformers_minimind()</code>
        </small>
      </p>

      {modelToggle}

      <Card title="Dense vs MoE 模型对比">
        <div style={{ background: 'var(--bg)', borderRadius: 'var(--radius)', border: '1px solid var(--border)', overflow: 'hidden' }}>
          {[
            { aspect: '架构', dense: '标准 Transformer（FFN）', moe: 'Transformer + Mixture of Experts' },
            { aspect: '参数量', dense: '26M (512d) / 104M (768d)', moe: '145M (768d + 8 专家)' },
            { aspect: '权重文件', dense: 'full_sft_512.pth', moe: 'full_sft_768_moe.pth' },
            { aspect: '转换脚本', dense: 'convert_to_hf.py', moe: 'convert_model.py' },
            { aspect: 'HF 架构', dense: 'LlamaForCausalLM', moe: 'MiniMindForCausalLM（自定义）' },
            { aspect: 'vLLM 加载', dense: '原生支持（Llama 格式）', moe: '--model-impl transformers + trust_remote_code' },
          ].map((row, i) => (
            <div key={i} style={{ display: 'grid', gridTemplateColumns: '100px 1fr 1fr', gap: 0, borderBottom: i < 5 ? '1px solid var(--border)' : 'none', fontSize: '0.82rem' }}>
              <div style={{ padding: '6px 10px', fontWeight: 'bold', color: 'var(--fg)', borderRight: '1px solid var(--border)', background: isDark ? '#1e293b' : '#f1f5f9' }}>{row.aspect}</div>
              <div style={{ padding: '6px 10px', borderRight: '1px solid var(--border)', background: isDense ? (isDark ? '#1e3a5f22' : '#dbeafe44') : 'transparent' }}>
                <code style={{ color: isDark ? '#60a5fa' : '#3b82f6', fontSize: '0.8rem' }}>{row.dense}</code>
              </div>
              <div style={{ padding: '6px 10px', background: !isDense ? (isDark ? '#3b1e5f22' : '#ede9fe44') : 'transparent' }}>
                <code style={{ color: isDark ? '#c084fc' : '#8b5cf6', fontSize: '0.8rem' }}>{row.moe}</code>
              </div>
            </div>
          ))}
        </div>
        <p style={{ marginTop: 10, fontSize: '0.82rem', color: 'var(--fg2)' }}>
          Dense 模型与 LlamaForCausalLM 结构完全兼容，可直接转换为 Llama 格式被 vLLM 原生支持。
          MoE 模型因包含路由器（Router）和多专家（Experts）等 Llama 不具备的组件，需要保留 MiniMind 自定义架构，
          通过 <code>trust_remote_code=True</code> 加载。
        </p>
      </Card>

      <Card title="部署流程总览">
        <p style={{ marginBottom: 10, fontSize: '0.9rem', color: 'var(--fg2)' }}>
          从训练权重到在线服务的完整流程。{isDense
            ? '当前展示 Dense 模型路径：通过 convert_to_hf.py 转换为标准 Llama 格式。'
            : '当前展示 MoE 模型路径：通过 convert_model.py 保留 MiniMind 自定义架构。'}
        </p>
        <svg
          width="100%"
          height={100}
          viewBox="0 0 720 100"
          dangerouslySetInnerHTML={{ __html: pipelineSvg }}
        />
      </Card>

      <Card title={isDense ? '模型格式转换（Dense → Llama）' : '模型格式转换（MoE → MiniMind-HF）'}>
        {isDense ? (
          <>
            <p style={{ marginBottom: 10, fontSize: '0.9rem', color: 'var(--fg2)' }}>
              Dense 模型通过 <code>convert_to_hf.py</code> 转换为 HuggingFace 标准目录结构。
              转换核心：将 MiniMind 架构映射为 LlamaForCausalLM（两者结构高度兼容），
              并利用 <code>tie_word_embeddings=True</code> 去除冗余的 lm_head.weight。
              点击下方步骤查看详情：
            </p>

            <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginBottom: 12 }}>
              {CONVERT_STEPS.map((s, i) => (
                <button
                  key={i}
                  className={`btn${selectedStep === i ? ' primary' : ''}`}
                  style={{ borderColor: s.color, color: selectedStep === i ? '#fff' : s.color, background: selectedStep === i ? s.color : 'transparent' }}
                  onClick={() => setSelectedStep(selectedStep === i ? null : i)}
                >
                  {i + 1}. {s.label}
                </button>
              ))}
            </div>
            {selectedStep !== null && (
              <div style={{ padding: 10, background: 'var(--bg)', border: `2px solid ${CONVERT_STEPS[selectedStep].color}`, borderRadius: 'var(--radius)', fontSize: '0.85rem', color: 'var(--fg2)' }}>
                <strong style={{ color: CONVERT_STEPS[selectedStep].color }}>{CONVERT_STEPS[selectedStep].label}</strong>：{CONVERT_STEPS[selectedStep].desc}
              </div>
            )}

            <div style={{ marginTop: 16 }}>
              <div className="label">转换命令</div>
              <pre style={{ background: 'var(--bg)', padding: 12, borderRadius: 'var(--radius)', border: '1px solid var(--border)', fontSize: '0.82rem', overflowX: 'auto' }}>
                <code>{`# 将 full_sft_512.pth 转换为 HuggingFace Llama 格式
python scripts/convert_to_hf.py \\
    --save_dir out \\
    --weight full_sft \\
    --hidden_size 512 \\
    --num_hidden_layers 8 \\
    --output_dir out/minimind-hf`}</code>
              </pre>
            </div>

            <SourcePanel
              title="对照源码：scripts/convert_to_hf.py:15-81"
              code={`def convert(args):
    # 1. 构建模型并加载权重
    config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=False,  # Dense 模型不启用 MoE
    )
    model = MiniMindForCausalLM(config)
    pth = os.path.join(args.save_dir, f"{args.weight}_{args.hidden_size}.pth")
    state_dict = torch.load(pth, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)

    # 2. 写 config.json (LlamaForCausalLM 格式)
    hf_config = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "vocab_size": config.vocab_size,
        "hidden_size": config.hidden_size,
        # ... 其他 Llama 兼容配置
        "tie_word_embeddings": True,
    }

    # 3. 保存权重为 safetensors (float16)
    sd = model.half().state_dict()
    if "lm_head.weight" in sd:
        del sd["lm_head.weight"]  # 与 embed_tokens 共享
    save_file(sd, os.path.join(out_dir, "model.safetensors"))

    # 4. 复制 tokenizer 文件到输出目录
    for fname in os.listdir(tokenizer_dir):
        if fname.startswith("tokenizer") or fname == "special_tokens_map.json":
            shutil.copy2(src, dst)`}
            />
          </>
        ) : (
          <>
            <p style={{ marginBottom: 10, fontSize: '0.9rem', color: 'var(--fg2)' }}>
              MoE 模型无法映射为标准 LlamaForCausalLM（因为 Llama 不包含 Router 和 Expert 模块），
              需要通过 <code>convert_model.py</code> 的 <code>convert_torch2transformers_minimind()</code> 函数，
              保留 MiniMind 自定义架构并注册到 AutoModelForCausalLM。
              转换后的模型需要 <code>trust_remote_code=True</code> 才能加载。
            </p>

            <div style={{ marginTop: 12, padding: 12, background: isDark ? '#3b1e5f22' : '#ede9fe44', border: '2px solid #8b5cf6', borderRadius: 'var(--radius)', fontSize: '0.85rem' }}>
              <strong style={{ color: '#8b5cf6' }}>MoE 转换要点</strong>
              <ul style={{ margin: '8px 0 0 0', paddingLeft: 20, color: 'var(--fg2)' }}>
                <li>权重文件名带 <code>_moe</code> 后缀：<code>full_sft_768_moe.pth</code></li>
                <li>使用 <code>MiniMindConfig.register_for_auto_class()</code> 注册自定义架构</li>
                <li>导出时使用 <code>save_pretrained()</code> 保存完整模型定义 + 权重</li>
                <li>加载时必须 <code>trust_remote_code=True</code>（包含自定义 Router/Expert 代码）</li>
              </ul>
            </div>

            <div style={{ marginTop: 16 }}>
              <div className="label">转换命令</div>
              <pre style={{ background: 'var(--bg)', padding: 12, borderRadius: 'var(--radius)', border: '1px solid var(--border)', fontSize: '0.82rem', overflowX: 'auto' }}>
                <code>{`# 修改 convert_model.py 的配置后执行
# lm_config = MiniMindConfig(hidden_size=768, num_hidden_layers=16,
#                             max_seq_len=8192, use_moe=True)
python scripts/convert_model.py`}</code>
              </pre>
            </div>

            <SourcePanel
              title="对照源码：scripts/convert_model.py:16-32 (MoE 转换)"
              code={`# MoE 模型需使用此函数转换（保留自定义架构）
def convert_torch2transformers_minimind(torch_path, transformers_path, dtype=torch.float16):
    # 注册自定义模型类到 AutoModel，使 from_pretrained 可识别
    MiniMindConfig.register_for_auto_class()
    MiniMindForCausalLM.register_for_auto_class("AutoModelForCausalLM")

    lm_model = MiniMindForCausalLM(lm_config)  # 包含 Router + Expert
    state_dict = torch.load(torch_path, map_location=device)
    lm_model.load_state_dict(state_dict, strict=False)
    lm_model = lm_model.to(dtype)

    # save_pretrained 会同时保存:
    #   - model.safetensors (权重)
    #   - config.json (含 MoE 特有参数: num_experts, top_k 等)
    #   - modeling_minimind.py (自定义模型代码，供 trust_remote_code 使用)
    lm_model.save_pretrained(transformers_path, safe_serialization=False)
    tokenizer = AutoTokenizer.from_pretrained('../model/')
    tokenizer.save_pretrained(transformers_path)`}
            />
          </>
        )}
      </Card>

      <Card title="HuggingFace 输出目录结构">
        <p style={{ marginBottom: 10, fontSize: '0.9rem', color: 'var(--fg2)' }}>
          转换后的目录包含以下文件，这是 vLLM 和 HuggingFace 生态的标准模型格式：
        </p>
        <div style={{ background: 'var(--bg)', padding: 12, borderRadius: 'var(--radius)', border: '1px solid var(--border)' }}>
          <div style={{ fontFamily: 'monospace', fontSize: '0.85rem' }}>
            <div style={{ color: 'var(--fg2)', marginBottom: 6 }}>{isDense ? 'minimind-hf/' : 'MiniMind2-MoE/'}</div>
            {HF_FILES.map((f, i) => (
              <div key={i} style={{ display: 'flex', gap: 12, padding: '4px 0 4px 20px', borderBottom: '1px solid var(--border)' }}>
                <span style={{ color: isDark ? '#60a5fa' : '#3b82f6', minWidth: 200, flexShrink: 0 }}>{f.name}</span>
                <span style={{ color: 'var(--fg2)', fontSize: '0.82rem' }}>{f.desc}</span>
              </div>
            ))}
            {!isDense && (
              <div style={{ display: 'flex', gap: 12, padding: '4px 0 4px 20px', borderBottom: '1px solid var(--border)' }}>
                <span style={{ color: isDark ? '#c084fc' : '#8b5cf6', minWidth: 200, flexShrink: 0 }}>modeling_minimind.py</span>
                <span style={{ color: 'var(--fg2)', fontSize: '0.82rem' }}>MoE 自定义模型代码（Router + Expert，供 trust_remote_code 加载）</span>
              </div>
            )}
            {!isDense && (
              <div style={{ display: 'flex', gap: 12, padding: '4px 0 4px 20px' }}>
                <span style={{ color: isDark ? '#c084fc' : '#8b5cf6', minWidth: 200, flexShrink: 0 }}>configuration_minimind.py</span>
                <span style={{ color: 'var(--fg2)', fontSize: '0.82rem' }}>MoE 配置类（含 num_experts、num_experts_per_tok 等参数）</span>
              </div>
            )}
          </div>
        </div>
      </Card>

      <Card title="vLLM 部署">
        <p style={{ marginBottom: 10, fontSize: '0.9rem', color: 'var(--fg2)' }}>
          <a href="https://github.com/vllm-project/vllm" target="_blank" rel="noreferrer" style={{ color: 'var(--accent)' }}>vLLM</a> 是高性能 LLM 推理引擎，通过 PagedAttention 优化显存管理，支持连续批处理，显著提升吞吐量。
          部署后提供 OpenAI 兼容的 Chat Completions API，可直接对接各类客户端。
        </p>

        <div className="viz-grid">
          <div>
            <div className="label">GPU 部署（CUDA）</div>
            <pre style={{ background: 'var(--bg)', padding: 12, borderRadius: 'var(--radius)', border: '1px solid var(--border)', fontSize: '0.82rem', overflowX: 'auto', margin: 0 }}>
              <code>{isDense
                ? `# Dense 模型：原生 Llama 格式，直接加载
vllm serve ./out/minimind-hf \\
    --served-model-name "minimind" \\
    --port 8998

# 或指定 transformers 后端
vllm serve ./out/minimind-hf \\
    --model-impl transformers \\
    --served-model-name "minimind" \\
    --port 8998`
                : `# MoE 模型：需要 trust_remote_code 加载自定义架构
vllm serve ./MiniMind2-MoE \\
    --model-impl transformers \\
    --served-model-name "minimind-moe" \\
    --trust-remote-code \\
    --port 8998`}</code>
            </pre>
          </div>
          <div>
            <div className="label">NPU 部署（昇腾 910B）</div>
            <pre style={{ background: 'var(--bg)', padding: 12, borderRadius: 'var(--radius)', border: '1px solid var(--border)', fontSize: '0.82rem', overflowX: 'auto', margin: 0 }}>
              <code>{isDense
                ? `# 使用 run_all_npu.sh 一键部署
bash scripts/run_all_npu.sh serve

# 等价于依次执行:
# 1. convert — 转换为 HF 格式
# 2. vllm   — 启动 vLLM Docker 服务
#
# 自定义选项:
bash scripts/run_all_npu.sh serve \\
    --vllm-port 8000 \\
    --max-model-len 2048`
                : `# MoE 模型使用 run_all_npu.sh 部署
# 注意: MoE 模型需先手动用
# convert_model.py 转换

# vLLM 启动时需加 trust-remote-code
# 当前 run_all_npu.sh 的 convert 阶段
# 仅支持 Dense 模型的自动转换`}</code>
            </pre>
          </div>
        </div>

        <div style={{ marginTop: 16 }}>
          <div className="label">测试 API 请求</div>
          <pre style={{ background: 'var(--bg)', padding: 12, borderRadius: 'var(--radius)', border: '1px solid var(--border)', fontSize: '0.82rem', overflowX: 'auto' }}>
            <code>{`curl http://localhost:8998/v1/chat/completions \\
    -H "Content-Type: application/json" \\
    -d '{
        "model": "${isDense ? 'minimind' : 'minimind-moe'}",
        "messages": [{"role": "user", "content": "你好"}],
        "max_tokens": 64
    }'`}</code>
          </pre>
        </div>

        <SourcePanel
          title="对照源码：scripts/run_all_npu.sh:377-457 (stage_vllm)"
          code={`stage_vllm() {
    local hf_dir="$PROJECT_DIR/out/minimind-hf"
    # 检查 HF 格式模型是否存在
    if [ ! -f "$hf_dir/config.json" ]; then
        log_error "请先运行 convert 阶段"
        return 1
    fi

    # 停止已有容器
    if docker ps -q --filter "name=$VLLM_CONTAINER_NAME" | grep -q .; then
        docker stop "$VLLM_CONTAINER_NAME"
    fi

    # 启动 vLLM Docker 容器 (昇腾 NPU)
    docker run -d --rm \\
        --name "$VLLM_CONTAINER_NAME" \\
        --device /dev/davinci0 \\
        -v "$hf_dir":/models/minimind:ro \\
        "$VLLM_IMAGE" \\
        vllm serve /models/minimind \\
            --host 0.0.0.0 --port "$VLLM_PORT" \\
            --dtype float16 --max-model-len "$MAX_MODEL_LEN"

    # 等待健康检查通过 (最长 120s)
    while ! curl -s "http://localhost:$VLLM_PORT/health"; do
        sleep 5
    done

    # 发送测试请求验证服务
    curl -s "http://localhost:$VLLM_PORT/v1/chat/completions" \\
        -d '{"model":"/models/minimind","messages":[{"role":"user","content":"你好"}]}'
}`}
        />
      </Card>
    </>
  );
}
