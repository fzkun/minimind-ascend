import { useState, useMemo, useCallback } from 'react';
import Card from '../components/Card';
import SourcePanel from '../components/SourcePanel';
import { useTheme } from '../context/ThemeContext';

type ModelType = 'dense' | 'moe';

interface PipelineStage {
  name: string;
  cmd: string;
  cmdMoe: string;
  desc: string;
  color: string;
  data?: string;
}

const PIPELINE: PipelineStage[] = [
  { name: 'download', cmd: 'bash scripts/run_all_npu.sh download', cmdMoe: 'bash scripts/run_all_npu.sh download', desc: '从 ModelScope 下载训练数据集（pretrain_hq.jsonl、sft_mini_512.jsonl、dpo.jsonl 等 6 个文件）', color: '#06b6d4', data: 'pretrain_hq.jsonl, sft_mini_512.jsonl, lora_identity.jsonl, dpo.jsonl, r1_mix_1024.jsonl, rlaif-mini.jsonl' },
  { name: 'build', cmd: 'bash scripts/run_all_npu.sh build', cmdMoe: 'bash scripts/run_all_npu.sh build', desc: '基于 Dockerfile.ascend 构建训练 Docker 镜像，安装 transformers、tokenizers 等依赖', color: '#8b5cf6' },
  { name: 'pretrain', cmd: 'bash scripts/run_all_npu.sh pretrain', cmdMoe: 'bash scripts/run_all_npu.sh --use-moe --hidden-size 768 --num-hidden-layers 16 pretrain', desc: '8 卡分布式预训练（torchrun nproc=8），学习语言模式。lr=5e-4, epochs=1', color: '#3b82f6' },
  { name: 'full_sft', cmd: 'bash scripts/run_all_npu.sh full_sft', cmdMoe: 'bash scripts/run_all_npu.sh --use-moe --hidden-size 768 --num-hidden-layers 16 full_sft', desc: '监督微调，学习指令跟随能力。lr=1e-6, epochs=2, 只对 assistant 回复计算损失', color: '#10b981' },
  { name: 'dpo', cmd: 'bash scripts/run_all_npu.sh dpo', cmdMoe: 'bash scripts/run_all_npu.sh --use-moe --hidden-size 768 --num-hidden-layers 16 dpo', desc: 'DPO 偏好对齐，通过 chosen/rejected 对比优化。lr=4e-8, β=0.1', color: '#ef4444' },
  { name: 'reason', cmd: 'bash scripts/run_all_npu.sh reason', cmdMoe: 'bash scripts/run_all_npu.sh --use-moe --hidden-size 768 --num-hidden-layers 16 reason', desc: '推理能力训练（R1 风格），使用思维链数据强化推理能力', color: '#f59e0b' },
  { name: 'eval', cmd: 'bash scripts/run_all_npu.sh eval', cmdMoe: 'bash scripts/run_all_npu.sh --use-moe --hidden-size 768 --num-hidden-layers 16 eval', desc: '单卡推理测试，验证训练效果', color: '#ec4899' },
];

const DOCKER_DEPS = [
  { pkg: 'transformers==4.45.2', purpose: 'HuggingFace 模型加载框架' },
  { pkg: 'tokenizers==0.20.4', purpose: 'BPE 分词器（Rust 加速）' },
  { pkg: 'torch_npu (预装)', purpose: '昇腾 NPU PyTorch 适配层' },
  { pkg: 'CANN (预装)', purpose: '昇腾计算加速库（Compute Architecture for Neural Networks）' },
  { pkg: 'wandb / swanlab', purpose: '训练指标可视化（可选）' },
];

const ENV_VARS = [
  { name: 'HCCL_CONNECT_TIMEOUT', value: '7200', desc: 'HCCL 集合通信连接超时（秒）' },
  { name: 'HCCL_EXEC_TIMEOUT', value: '7200', desc: 'HCCL 集合通信执行超时（秒）' },
  { name: 'TASK_QUEUE_ENABLE', value: '1', desc: '启用任务队列优化（提升多卡通信效率）' },
  { name: 'ASCEND_RT_VISIBLE_DEVICES', value: '0,1,2,...,7', desc: '对容器可见的 NPU 设备列表' },
];

export default function NPUDockerSection() {
  const { isDark } = useTheme();
  const [selectedStage, setSelectedStage] = useState<number | null>(null);
  const [showPresets, setShowPresets] = useState(false);
  const [modelType, setModelType] = useState<ModelType>('dense');

  const isDense = modelType === 'dense';

  // Pipeline SVG
  const pipelineSvg = useMemo(() => {
    const fg = isDark ? '#e2e8f0' : '#1a1a2e';
    const fg2 = isDark ? '#94a3b8' : '#555';
    let html = `<defs><marker id="arrNpu" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0 L10,5 L0,10 Z" fill="${fg2}"/></marker></defs>`;
    const w = 100, gap = 14, y = 30, h = 45;
    PIPELINE.forEach((s, i) => {
      const x = i * (w + gap) + 10;
      const opacity = selectedStage === i ? 0.4 : 0.15;
      html += `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="8" fill="${s.color}" opacity="${opacity}" stroke="${s.color}" stroke-width="2" style="cursor:pointer" data-stage="${i}"/>`;
      html += `<text x="${x + w / 2}" y="${y + h / 2 + 4}" text-anchor="middle" fill="${fg}" font-size="11" font-weight="bold" style="pointer-events:none">${s.name}</text>`;
      if (i < PIPELINE.length - 1) {
        html += `<line x1="${x + w}" y1="${y + h / 2}" x2="${x + w + gap}" y2="${y + h / 2}" stroke="${fg2}" stroke-width="1.5" marker-end="url(#arrNpu)"/>`;
      }
    });
    return html;
  }, [isDark, selectedStage]);

  const handleSvgClick = useCallback((e: React.MouseEvent<SVGSVGElement>) => {
    const target = e.target as SVGElement;
    const rect = target.closest('rect[data-stage]');
    if (rect) {
      setSelectedStage(parseInt(rect.getAttribute('data-stage')!));
    }
  }, []);

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
      <h2>9. 昇腾 910B Docker 复现</h2>
      <p className="desc">
        在华为昇腾 910B NPU 上通过 Docker 一键复现完整训练流程。只需一条命令 <code>bash scripts/run_all_npu.sh all</code> 即可从数据下载跑到推理测试。
        脚本自动处理镜像构建、8 卡 <code>torchrun</code> 分布式训练和断点续训。加 <code>--use-moe</code> 切换 MoE 架构。
        <br/>
        <small style={{ color: 'var(--fg2)' }}>
          关联源码：<code>scripts/run_all_npu.sh</code> 编排脚本 | <code>scripts/run_train_npu.sh</code> 8 卡启动器 | <code>Dockerfile.ascend</code> 镜像定义 | <code>trainer/trainer_utils.py:23</code> <code>is_npu_available()</code>
        </small>
      </p>

      {modelToggle}

      <Card title="Dense vs MoE 训练差异">
        <div style={{ background: 'var(--bg)', borderRadius: 'var(--radius)', border: '1px solid var(--border)', overflow: 'hidden' }}>
          {[
            { aspect: '参数量', dense: '26M (512d×8层)', moe: '145M (768d×16层 + 8专家)' },
            { aspect: '启用方式', dense: '默认（无需额外参数）', moe: '--use-moe' },
            { aspect: 'hidden_size', dense: '512', moe: '768' },
            { aspect: 'num_hidden_layers', dense: '8', moe: '16' },
            { aspect: '权重命名', dense: 'full_sft_512.pth', moe: 'full_sft_768_moe.pth' },
            { aspect: 'Checkpoint 命名', dense: 'full_sft_512_resume.pth', moe: 'full_sft_768_moe_resume.pth' },
            { aspect: '显存占用', dense: '较小，单卡可训练', moe: '较大，建议多卡分布式' },
            { aspect: 'MoE 辅助损失', dense: '无', moe: '负载均衡 loss（防止专家坍缩）' },
          ].map((row, i) => (
            <div key={i} style={{ display: 'grid', gridTemplateColumns: '140px 1fr 1fr', gap: 0, borderBottom: i < 7 ? '1px solid var(--border)' : 'none', fontSize: '0.82rem' }}>
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
          {isDense
            ? 'Dense 模型所有 token 经过相同的 FFN 层，结构简单，适合快速验证训练流程。'
            : 'MoE 模型每个 token 由 Router 动态选择 top-k 个 Expert 处理，参数量更大但计算量可控（稀疏激活）。训练时额外计算负载均衡辅助损失，防止所有 token 被路由到同一个 Expert。'}
        </p>
      </Card>

      <Card title="环境准备">
        <p style={{ marginBottom: 10, fontSize: '0.9rem', color: 'var(--fg2)' }}>
          在昇腾 910B 服务器上运行需要以下环境。Docker 镜像基于 <code>ascend-pretrain:latest</code>（预装 CANN + torch_npu + PyTorch），
          Dockerfile.ascend 在此基础上安装训练所需的额外依赖。Dense 和 MoE 使用相同的 Docker 镜像。
        </p>

        <div className="viz-grid">
          <div>
            <div className="label">Docker 镜像依赖</div>
            <div style={{ background: 'var(--bg)', borderRadius: 'var(--radius)', border: '1px solid var(--border)', overflow: 'hidden' }}>
              {DOCKER_DEPS.map((d, i) => (
                <div key={i} style={{ display: 'flex', justifyContent: 'space-between', padding: '6px 10px', borderBottom: i < DOCKER_DEPS.length - 1 ? '1px solid var(--border)' : 'none', fontSize: '0.82rem' }}>
                  <code style={{ color: isDark ? '#60a5fa' : '#3b82f6' }}>{d.pkg}</code>
                  <span style={{ color: 'var(--fg2)' }}>{d.purpose}</span>
                </div>
              ))}
            </div>
          </div>
          <div>
            <div className="label">关键环境变量</div>
            <div style={{ background: 'var(--bg)', borderRadius: 'var(--radius)', border: '1px solid var(--border)', overflow: 'hidden' }}>
              {ENV_VARS.map((v, i) => (
                <div key={i} style={{ padding: '6px 10px', borderBottom: i < ENV_VARS.length - 1 ? '1px solid var(--border)' : 'none', fontSize: '0.82rem' }}>
                  <code style={{ color: isDark ? '#34d399' : '#10b981' }}>{v.name}={v.value}</code>
                  <div style={{ color: 'var(--fg2)', fontSize: '0.78rem', marginTop: 2 }}>{v.desc}</div>
                </div>
              ))}
            </div>
          </div>
        </div>

        <SourcePanel
          title="对照源码：Dockerfile.ascend"
          code={`FROM ascend-pretrain:latest
WORKDIR /workspace/minimind
COPY . /workspace/minimind/

# 升级 transformers + tokenizers 以兼容新格式 tokenizer.json
RUN pip install --no-cache-dir --force-reinstall \\
    transformers==4.45.2 tokenizers==0.20.4 huggingface_hub>=0.24

# 修复 transformers 4.45 与 torch 2.1 的 pytree 兼容性问题
RUN SITE_PKG=$(python3 -c "import site; print(site.getsitepackages()[0])") && \\
    echo 'import torch.utils._pytree as _p; ...' > "$SITE_PKG/fix_pytree.py"

# 安装训练依赖: jieba, rich, wandb, swanlab, einops ...
# 确保 numpy<2（CANN 不兼容 numpy 2.x）
RUN pip install --no-cache-dir "numpy<2"

ENV HCCL_CONNECT_TIMEOUT=7200
ENV HCCL_EXEC_TIMEOUT=7200
ENV ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`}
        />
      </Card>

      <Card title="训练流水线">
        <p style={{ marginBottom: 10, fontSize: '0.9rem', color: 'var(--fg2)' }}>
          <code>run_all_npu.sh</code> 将完整训练流程编排为多个阶段，可按需选择执行。
          每个训练阶段在 Docker 容器中通过 <code>torchrun --nproc_per_node=8</code> 启动 8 卡分布式训练。
          {!isDense && ' MoE 模型需要额外传入 --use-moe 标志，脚本会自动为所有阶段附加 --use_moe 1 参数。'}
          点击各阶段查看详情：
        </p>
        <svg
          width="100%"
          height={100}
          viewBox="0 0 808 100"
          onClick={handleSvgClick}
          dangerouslySetInnerHTML={{ __html: pipelineSvg }}
        />
        <div style={{ marginTop: 10, fontSize: '0.85rem', color: 'var(--fg2)', minHeight: 50 }}>
          {selectedStage !== null && (
            <div style={{ padding: 10, background: 'var(--bg)', border: `2px solid ${PIPELINE[selectedStage].color}`, borderRadius: 'var(--radius)' }}>
              <strong style={{ color: PIPELINE[selectedStage].color }}>{PIPELINE[selectedStage].name}</strong>
              ：{PIPELINE[selectedStage].desc}
              <pre style={{ marginTop: 8, fontSize: '0.82rem', color: isDark ? '#e2e8f0' : '#1a1a2e' }}>
                <code>$ {isDense ? PIPELINE[selectedStage].cmd : PIPELINE[selectedStage].cmdMoe}</code>
              </pre>
              {PIPELINE[selectedStage].data && (
                <div style={{ marginTop: 6, fontSize: '0.8rem', color: 'var(--fg2)' }}>
                  数据文件：{PIPELINE[selectedStage].data}
                </div>
              )}
            </div>
          )}
        </div>
      </Card>

      <Card title="快速开始">
        <p style={{ marginBottom: 10, fontSize: '0.9rem', color: 'var(--fg2)' }}>
          最常用的命令集合。<code>run_all_npu.sh</code> 提供了预设组合（all、core、serve、rl），
          也可以自由组合各阶段名。脚本会自动处理 Docker 容器启动、设备挂载和目录映射。
        </p>

        <div style={{ marginBottom: 12 }}>
          <button className="btn" onClick={() => setShowPresets(!showPresets)}>
            {showPresets ? '收起' : '展开'}预设组合说明
          </button>
        </div>
        {showPresets && (
          <div style={{ background: 'var(--bg)', padding: 12, borderRadius: 'var(--radius)', border: '1px solid var(--border)', marginBottom: 16, fontSize: '0.85rem' }}>
            <div style={{ display: 'grid', gridTemplateColumns: 'auto 1fr', gap: '6px 16px' }}>
              <code style={{ color: isDark ? '#60a5fa' : '#3b82f6' }}>all</code>
              <span style={{ color: 'var(--fg2)' }}>download → build → pretrain → full_sft → dpo → reason → eval</span>
              <code style={{ color: isDark ? '#34d399' : '#10b981' }}>core</code>
              <span style={{ color: 'var(--fg2)' }}>download → build → pretrain → full_sft → eval（最小可用流程）</span>
              <code style={{ color: isDark ? '#fbbf24' : '#f59e0b' }}>serve</code>
              <span style={{ color: 'var(--fg2)' }}>convert → vllm（模型转换 + 部署服务）</span>
              <code style={{ color: isDark ? '#f87171' : '#ef4444' }}>rl</code>
              <span style={{ color: 'var(--fg2)' }}>ppo → grpo → spo（强化学习系列）</span>
            </div>
          </div>
        )}

        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          <div>
            <div className="label">{isDense ? '一键完整训练 — Dense（推荐）' : '一键完整训练 — MoE'}</div>
            <pre style={{ background: 'var(--bg)', padding: 12, borderRadius: 'var(--radius)', border: `1px solid ${isDense ? 'var(--border)' : '#8b5cf6'}`, fontSize: '0.82rem', overflowX: 'auto', margin: 0 }}>
              <code>{isDense
                ? `# Dense 模型完整训练（26M 参数，默认 512 维 8 层）
# download → build → pretrain → sft → dpo → reason → eval
bash scripts/run_all_npu.sh all`
                : `# MoE 模型完整训练（145M 参数，768 维 16 层 + 8 专家）
# --use-moe 为所有训练阶段自动附加 --use_moe 1
# 权重文件名自动添加 _moe 后缀
bash scripts/run_all_npu.sh \\
    --use-moe \\
    --hidden-size 768 \\
    --num-hidden-layers 16 \\
    all`}</code>
            </pre>
          </div>

          <div>
            <div className="label">最小可用流程</div>
            <pre style={{ background: 'var(--bg)', padding: 12, borderRadius: 'var(--radius)', border: '1px solid var(--border)', fontSize: '0.82rem', overflowX: 'auto', margin: 0 }}>
              <code>{isDense
                ? `# Dense: 仅预训练 + 监督微调 + 推理测试
bash scripts/run_all_npu.sh core`
                : `# MoE: 仅预训练 + 监督微调 + 推理测试
bash scripts/run_all_npu.sh \\
    --use-moe --hidden-size 768 --num-hidden-layers 16 \\
    core`}</code>
            </pre>
          </div>

          <div>
            <div className="label">单独执行某个阶段</div>
            <pre style={{ background: 'var(--bg)', padding: 12, borderRadius: 'var(--radius)', border: '1px solid var(--border)', fontSize: '0.82rem', overflowX: 'auto', margin: 0 }}>
              <code>{isDense
                ? `# Dense 模型示例
bash scripts/run_all_npu.sh build
bash scripts/run_all_npu.sh pretrain
bash scripts/run_all_npu.sh pretrain full_sft eval`
                : `# MoE 模型示例（每条命令都需要 --use-moe）
bash scripts/run_all_npu.sh build
bash scripts/run_all_npu.sh --use-moe --hidden-size 768 --num-hidden-layers 16 pretrain
bash scripts/run_all_npu.sh --use-moe --hidden-size 768 --num-hidden-layers 16 pretrain full_sft eval`}</code>
            </pre>
          </div>

          <div>
            <div className="label">高级选项</div>
            <pre style={{ background: 'var(--bg)', padding: 12, borderRadius: 'var(--radius)', border: '1px solid var(--border)', fontSize: '0.82rem', overflowX: 'auto', margin: 0 }}>
              <code>{isDense
                ? `# 断点续训（自动从最近的 checkpoint 恢复）
bash scripts/run_all_npu.sh --resume pretrain full_sft

# 自定义模型维度（768 维，16 层 → 104M 参数 Dense）
bash scripts/run_all_npu.sh \\
    --hidden-size 768 \\
    --num-hidden-layers 16 \\
    all

# dry-run：只打印命令不实际执行
bash scripts/run_all_npu.sh --dry-run all`
                : `# MoE 断点续训
bash scripts/run_all_npu.sh \\
    --resume --use-moe \\
    --hidden-size 768 --num-hidden-layers 16 \\
    pretrain full_sft

# dry-run：只打印命令不实际执行
bash scripts/run_all_npu.sh \\
    --dry-run --use-moe \\
    --hidden-size 768 --num-hidden-layers 16 \\
    all`}</code>
            </pre>
          </div>
        </div>

        <SourcePanel
          title="对照源码：scripts/run_train_npu.sh（8 卡训练启动器）"
          code={`#!/bin/bash
# MiniMind 昇腾910B 8卡分布式训练启动脚本
# 用法: bash scripts/run_train_npu.sh pretrain --epochs 1 --batch_size 32

TRAIN_MODE=\${1:-pretrain}
shift

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR/trainer"

# 使用 torchrun 启动 8 卡分布式训练
# DDP 后端使用 HCCL（华为集合通信库）替代 NCCL
torchrun --nnodes=1 --nproc_per_node=8 --master_port=29500 \\
    train_\${TRAIN_MODE}.py "$@"
# MoE 模型训练时，run_all_npu.sh 会自动传入 --use_moe 1 参数`}
        />
      </Card>

      <Card title="Docker 容器启动细节">
        <p style={{ marginBottom: 10, fontSize: '0.9rem', color: 'var(--fg2)' }}>
          在 Docker 中运行 NPU 训练需要挂载昇腾驱动、NPU 设备节点和数据目录。
          <code>run_all_npu.sh</code> 自动处理这些挂载，但如果需要手动启动容器，可以参考以下命令：
        </p>
        <pre style={{ background: 'var(--bg)', padding: 12, borderRadius: 'var(--radius)', border: '1px solid var(--border)', fontSize: '0.78rem', overflowX: 'auto' }}>
          <code>{isDense
            ? `docker run -it --rm --network=host --shm-size=500g \\
    --device /dev/davinci0 \\
    --device /dev/davinci1 \\
    --device /dev/davinci2 \\
    --device /dev/davinci3 \\
    --device /dev/davinci4 \\
    --device /dev/davinci5 \\
    --device /dev/davinci6 \\
    --device /dev/davinci7 \\
    --device /dev/davinci_manager \\
    --device /dev/devmm_svm \\
    --device /dev/hisi_hdc \\
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \\
    -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi:ro \\
    -v $(pwd)/dataset:/workspace/minimind/dataset \\
    -v $(pwd)/out:/workspace/minimind/out \\
    -v $(pwd)/checkpoints:/workspace/minimind/checkpoints \\
    minimind-npu \\
    bash scripts/run_train_npu.sh pretrain --epochs 1 --batch_size 32`
            : `docker run -it --rm --network=host --shm-size=500g \\
    --device /dev/davinci0 \\
    --device /dev/davinci1 \\
    --device /dev/davinci2 \\
    --device /dev/davinci3 \\
    --device /dev/davinci4 \\
    --device /dev/davinci5 \\
    --device /dev/davinci6 \\
    --device /dev/davinci7 \\
    --device /dev/davinci_manager \\
    --device /dev/devmm_svm \\
    --device /dev/hisi_hdc \\
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \\
    -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi:ro \\
    -v $(pwd)/dataset:/workspace/minimind/dataset \\
    -v $(pwd)/out:/workspace/minimind/out \\
    -v $(pwd)/checkpoints:/workspace/minimind/checkpoints \\
    minimind-npu \\
    bash scripts/run_train_npu.sh pretrain \\
        --epochs 1 --batch_size 32 \\
        --hidden_size 768 --num_hidden_layers 16 \\
        --use_moe 1`}</code>
        </pre>
        <div style={{ marginTop: 12, fontSize: '0.85rem', color: 'var(--fg2)' }}>
          <div style={{ display: 'grid', gridTemplateColumns: 'auto 1fr', gap: '4px 12px' }}>
            <code style={{ color: isDark ? '#f87171' : '#ef4444' }}>/dev/davinci0~7</code>
            <span>8 块昇腾 910B NPU 设备节点</span>
            <code style={{ color: isDark ? '#f87171' : '#ef4444' }}>/dev/davinci_manager</code>
            <span>NPU 设备管理节点</span>
            <code style={{ color: isDark ? '#60a5fa' : '#3b82f6' }}>Ascend/driver</code>
            <span>昇腾驱动（只读挂载）</span>
            <code style={{ color: isDark ? '#34d399' : '#10b981' }}>dataset/</code>
            <span>训练数据（双向挂载，支持下载）</span>
            <code style={{ color: isDark ? '#34d399' : '#10b981' }}>out/</code>
            <span>模型权重输出目录</span>
            <code style={{ color: isDark ? '#34d399' : '#10b981' }}>checkpoints/</code>
            <span>断点续训检查点</span>
            <code style={{ color: isDark ? '#fbbf24' : '#f59e0b' }}>--shm-size=500g</code>
            <span>共享内存（DDP 多进程通信需要大共享内存）</span>
          </div>
        </div>

        <SourcePanel
          title="对照源码：scripts/run_all_npu.sh:125-156 (run_docker)"
          code={`run_docker() {
    local extra_mounts=("$@")
    local tty_flag="-i"
    [ -t 0 ] && tty_flag="-it"
    local cmd_args=(
        docker run $tty_flag --rm --network=host --shm-size=500g
        --device /dev/davinci0
        --device /dev/davinci1
        # ... davinci2~7, davinci_manager, devmm_svm, hisi_hdc
        -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro
        -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi:ro
        -v "$PROJECT_DIR/dataset":/workspace/minimind/dataset
        -v "$PROJECT_DIR/out":/workspace/minimind/out
        -v "$PROJECT_DIR/checkpoints":/workspace/minimind/checkpoints
    )
    for m in "\${extra_mounts[@]}"; do
        cmd_args+=(-v "$m")
    done
    cmd_args+=("$IMAGE_NAME")
}`}
        />
      </Card>

      <Card title="NPU/CUDA 兼容适配">
        <p style={{ marginBottom: 10, fontSize: '0.9rem', color: 'var(--fg2)' }}>
          MiniMind 的所有训练脚本（Dense 和 MoE）都同时支持 CUDA GPU 和昇腾 NPU。训练代码通过统一的适配模式，
          在运行时自动检测硬件并选择对应的后端，无需修改训练代码即可在两种硬件上运行：
        </p>
        <div style={{ background: 'var(--bg)', borderRadius: 'var(--radius)', border: '1px solid var(--border)', overflow: 'hidden' }}>
          {[
            { aspect: '设备选择', cuda: 'cuda:{rank}', npu: 'npu:{rank}', desc: '自动检测 torch_npu 是否可用' },
            { aspect: 'DDP 后端', cuda: 'nccl', npu: 'hccl', desc: 'NCCL vs 华为 HCCL 集合通信' },
            { aspect: '混合精度', cuda: 'torch.cuda.amp.autocast()', npu: 'torch.amp.autocast("npu")', desc: '' },
            { aspect: 'GradScaler', cuda: 'torch.cuda.amp.GradScaler()', npu: 'torch.amp.GradScaler("npu")', desc: '' },
            { aspect: '默认精度', cuda: 'bfloat16', npu: 'float16', desc: 'NPU 对 bf16 支持有限' },
            { aspect: 'torch.compile', cuda: '启用', npu: '跳过', desc: 'NPU 暂不支持 compile' },
          ].map((row, i) => (
            <div key={i} style={{ display: 'grid', gridTemplateColumns: '120px 1fr 1fr', gap: 0, borderBottom: i < 5 ? '1px solid var(--border)' : 'none', fontSize: '0.82rem' }}>
              <div style={{ padding: '6px 10px', fontWeight: 'bold', color: 'var(--fg)', borderRight: '1px solid var(--border)', background: isDark ? '#1e293b' : '#f1f5f9' }}>{row.aspect}</div>
              <div style={{ padding: '6px 10px', borderRight: '1px solid var(--border)' }}>
                <code style={{ color: isDark ? '#60a5fa' : '#3b82f6' }}>{row.cuda}</code>
              </div>
              <div style={{ padding: '6px 10px' }}>
                <code style={{ color: isDark ? '#34d399' : '#10b981' }}>{row.npu}</code>
                {row.desc && <span style={{ color: 'var(--fg2)', fontSize: '0.78rem', marginLeft: 8 }}>{row.desc}</span>}
              </div>
            </div>
          ))}
        </div>

        <SourcePanel
          title="对照源码：trainer/trainer_utils.py (NPU 检测)"
          code={`def is_npu_available():
    """检测昇腾 NPU (torch_npu) 是否可用"""
    try:
        import torch_npu
        return torch.npu.is_available()
    except ImportError:
        return False

def init_distributed_mode():
    """DDP 初始化：根据硬件自动选择 NCCL/HCCL 后端"""
    backend = 'hccl' if is_npu_available() else 'nccl'
    dist.init_process_group(backend=backend)

    local_rank = int(os.environ['LOCAL_RANK'])
    if is_npu_available():
        device = torch.device(f'npu:{local_rank}')
        torch.npu.set_device(device)
    else:
        device = torch.device(f'cuda:{local_rank}')
        torch.cuda.set_device(device)
    return device`}
        />
      </Card>
    </>
  );
}
