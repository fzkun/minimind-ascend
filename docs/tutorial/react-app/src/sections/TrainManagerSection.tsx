import { useState, useRef, useCallback, useEffect, useMemo } from 'react';

// ==================== 类型 ====================
interface WeightFile {
  name: string;
  size_mb: number;
}

interface DatasetFile {
  name: string;
  size_mb: number;
}

interface TrainProgress {
  epoch: number;
  totalEpochs: number;
  step: number;
  totalSteps: number;
  percent: number;
}

// ==================== 阶段默认参数 ====================
const STAGE_DEFAULTS: Record<string, { epochs: number; batch_size: number; lr: number; data_path: string; from_weight: string }> = {
  pretrain:      { epochs: 1, batch_size: 32, lr: 1e-4, data_path: '', from_weight: '' },
  full_sft:      { epochs: 2, batch_size: 16, lr: 1e-5, data_path: '', from_weight: 'pretrain' },
  lora:          { epochs: 50, batch_size: 32, lr: 1e-4, data_path: '', from_weight: 'full_sft' },
  dpo:           { epochs: 1, batch_size: 4, lr: 5e-6, data_path: '', from_weight: 'full_sft' },
  ppo:           { epochs: 1, batch_size: 2, lr: 1e-6, data_path: '', from_weight: 'full_sft' },
  grpo:          { epochs: 1, batch_size: 2, lr: 1e-6, data_path: '', from_weight: 'full_sft' },
  spo:           { epochs: 1, batch_size: 2, lr: 1e-6, data_path: '', from_weight: 'full_sft' },
  distillation:  { epochs: 6, batch_size: 32, lr: 1e-4, data_path: '', from_weight: 'full_sft' },
  reason:        { epochs: 1, batch_size: 8, lr: 1e-5, data_path: '', from_weight: 'full_sft' },
};

const STAGES = Object.keys(STAGE_DEFAULTS);

// ==================== 组件 ====================
export default function TrainManagerSection() {
  // 连接配置
  const [managerUrl, setManagerUrl] = useState('');
  const [connected, setConnected] = useState(false);
  const [serverStatus, setServerStatus] = useState<'idle' | 'running' | 'finished' | 'error'>('idle');
  const [serverStage, setServerStage] = useState('');
  const [elapsed, setElapsed] = useState<number | null>(null);

  // 训练参数
  const [stage, setStage] = useState('full_sft');
  const [hiddenSize, setHiddenSize] = useState(512);
  const [numLayers, setNumLayers] = useState(8);
  const [useMoe, setUseMoe] = useState(false);
  const [epochs, setEpochs] = useState(2);
  const [batchSize, setBatchSize] = useState(16);
  const [learningRate, setLearningRate] = useState(1e-5);
  const [dataPath, setDataPath] = useState('');
  const [fromWeight, setFromWeight] = useState('pretrain');
  const [saveWeight, setSaveWeight] = useState('');
  const [extraArgs, setExtraArgs] = useState('');

  // 数据列表
  const [weights, setWeights] = useState<WeightFile[]>([]);
  const [datasets, setDatasets] = useState<DatasetFile[]>([]);
  const [weightsOpen, setWeightsOpen] = useState(false);

  // 日志
  const [logs, setLogs] = useState<string[]>([]);
  const [logFilter, setLogFilter] = useState('');
  const [progress, setProgress] = useState<TrainProgress | null>(null);
  const [lossValues, setLossValues] = useState<number[]>([]);

  // NPU
  const [npuRaw, setNpuRaw] = useState('');

  const logsRef = useRef<HTMLDivElement>(null);
  const eventSourceRef = useRef<EventSource | null>(null);
  const pollRef = useRef<number | null>(null);

  // ==================== 阶段切换时更新默认参数 ====================
  const handleStageChange = useCallback((newStage: string) => {
    setStage(newStage);
    const defaults = STAGE_DEFAULTS[newStage];
    if (defaults) {
      setEpochs(defaults.epochs);
      setBatchSize(defaults.batch_size);
      setLearningRate(defaults.lr);
      setFromWeight(defaults.from_weight);
      if (defaults.data_path) setDataPath(defaults.data_path);
    }
  }, []);

  // ==================== 轮询状态 ====================
  useEffect(() => {
    const poll = async () => {
      try {
        const resp = await fetch(`${managerUrl}/api/status`);
        if (!resp.ok) throw new Error();
        const data = await resp.json();
        setConnected(true);
        setServerStatus(data.status);
        setServerStage(data.stage || '');
        setElapsed(data.elapsed_seconds);
      } catch {
        setConnected(false);
      }
    };
    poll();
    pollRef.current = window.setInterval(poll, 3000);
    return () => { if (pollRef.current) clearInterval(pollRef.current); };
  }, [managerUrl]);

  // ==================== 获取权重和数据集列表 ====================
  const fetchLists = useCallback(async () => {
    try {
      const [wResp, dResp] = await Promise.all([
        fetch(`${managerUrl}/api/weights`),
        fetch(`${managerUrl}/api/datasets`),
      ]);
      if (wResp.ok) {
        const wData = await wResp.json();
        setWeights(wData.weights || []);
      }
      if (dResp.ok) {
        const dData = await dResp.json();
        setDatasets(dData.datasets || []);
      }
    } catch { /* ignore */ }
  }, [managerUrl]);

  useEffect(() => {
    if (connected) fetchLists();
  }, [connected, fetchLists]);

  // ==================== NPU 状态 ====================
  const fetchNpu = useCallback(async () => {
    try {
      const resp = await fetch(`${managerUrl}/api/npu`);
      if (resp.ok) {
        const data = await resp.json();
        setNpuRaw(data.raw || '');
      }
    } catch { /* ignore */ }
  }, [managerUrl]);

  useEffect(() => {
    if (!connected) return;
    fetchNpu();
    const id = setInterval(fetchNpu, 10000);
    return () => clearInterval(id);
  }, [connected, fetchNpu]);

  // ==================== 日志解析 ====================
  const parseLogLine = useCallback((line: string) => {
    // 解析进度: Epoch:[1/3](150/4500) 或 Epoch: [1/3]  Step: 150/4500
    const epochMatch = line.match(/Epoch[:\s]*\[?(\d+)\/(\d+)\]?\s*(?:\(|Step[:\s]*)?(\d+)\/(\d+)/i);
    if (epochMatch) {
      const epoch = parseInt(epochMatch[1]);
      const totalEpochs = parseInt(epochMatch[2]);
      const step = parseInt(epochMatch[3]);
      const totalSteps = parseInt(epochMatch[4]);
      const percent = ((epoch - 1) / totalEpochs + step / totalSteps / totalEpochs) * 100;
      setProgress({ epoch, totalEpochs, step, totalSteps, percent: Math.min(100, percent) });
    }

    // 解析 loss
    const lossMatch = line.match(/loss[=:\s]+([0-9]+\.?[0-9]*)/i);
    if (lossMatch) {
      const loss = parseFloat(lossMatch[1]);
      if (!isNaN(loss) && loss < 100) {
        setLossValues(prev => [...prev.slice(-99), loss]);
      }
    }
  }, []);

  // ==================== SSE 日志流 ====================
  const connectLogs = useCallback(() => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
    }
    const es = new EventSource(`${managerUrl}/api/train/logs`);
    es.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        if (msg.type === 'log') {
          setLogs(prev => [...prev, msg.data]);
          parseLogLine(msg.data);
          // 自动滚动
          requestAnimationFrame(() => {
            if (logsRef.current) {
              logsRef.current.scrollTop = logsRef.current.scrollHeight;
            }
          });
        }
      } catch { /* ignore */ }
    };
    es.onerror = () => {
      es.close();
      eventSourceRef.current = null;
    };
    eventSourceRef.current = es;
  }, [managerUrl, parseLogLine]);

  // 训练开始后自动连接日志
  useEffect(() => {
    if (serverStatus === 'running') {
      connectLogs();
    }
    return () => {
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
        eventSourceRef.current = null;
      }
    };
  }, [serverStatus, connectLogs]);

  // ==================== 开始训练 ====================
  const startTrain = useCallback(async () => {
    try {
      setLogs([]);
      setProgress(null);
      setLossValues([]);

      const body = {
        stage,
        hidden_size: hiddenSize,
        num_hidden_layers: numLayers,
        use_moe: useMoe,
        epochs,
        batch_size: batchSize,
        learning_rate: learningRate,
        data_path: dataPath,
        from_weight: fromWeight,
        save_weight: saveWeight,
        extra_args: extraArgs,
      };

      const resp = await fetch(`${managerUrl}/api/train/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      const data = await resp.json();
      if (!data.ok) {
        alert('启动失败: ' + data.error);
        return;
      }
      setLogs([`[UI] 训练已启动: ${data.command}`]);
      connectLogs();
    } catch (e) {
      alert('请求失败: ' + (e instanceof Error ? e.message : e));
    }
  }, [stage, hiddenSize, numLayers, useMoe, epochs, batchSize, learningRate, dataPath, fromWeight, saveWeight, extraArgs, managerUrl, connectLogs]);

  // ==================== 停止训练 ====================
  const stopTrain = useCallback(async () => {
    try {
      const resp = await fetch(`${managerUrl}/api/train/stop`, { method: 'POST' });
      const data = await resp.json();
      if (!data.ok) {
        alert('停止失败: ' + data.error);
      }
    } catch (e) {
      alert('请求失败: ' + (e instanceof Error ? e.message : e));
    }
  }, [managerUrl]);

  // ==================== 状态灯颜色 ====================
  const statusDotClass = useMemo(() => {
    if (!connected) return 'gray';
    switch (serverStatus) {
      case 'running': return 'green';
      case 'error': return 'red';
      case 'finished': return 'yellow';
      default: return 'green';
    }
  }, [connected, serverStatus]);

  const statusText = useMemo(() => {
    if (!connected) return '未连接';
    const elapsedStr = elapsed ? ` (${Math.floor(elapsed / 60)}分${elapsed % 60}秒)` : '';
    switch (serverStatus) {
      case 'running': return `训练中: ${serverStage}${elapsedStr}`;
      case 'error': return '训练异常';
      case 'finished': return '训练完成';
      default: return '空闲';
    }
  }, [connected, serverStatus, serverStage, elapsed]);

  // ==================== Loss 趋势 SVG ====================
  const lossSvg = useMemo(() => {
    if (lossValues.length < 2) return null;
    const w = 280, h = 70, pad = 4;
    const maxV = Math.max(...lossValues);
    const minV = Math.min(...lossValues);
    const range = maxV - minV || 1;
    const points = lossValues.map((v, i) => {
      const x = pad + (i / (lossValues.length - 1)) * (w - pad * 2);
      const y = pad + (1 - (v - minV) / range) * (h - pad * 2);
      return `${x},${y}`;
    }).join(' ');

    return (
      <div className="tm-loss-chart">
        <div style={{ fontSize: '0.72rem', color: 'var(--fg3)', marginBottom: 2 }}>
          Loss 趋势 (最近{lossValues.length}步, 最新: {lossValues[lossValues.length - 1].toFixed(4)})
        </div>
        <svg viewBox={`0 0 ${w} ${h}`}>
          <polyline
            points={points}
            fill="none"
            stroke="var(--accent)"
            strokeWidth="1.5"
            strokeLinejoin="round"
          />
        </svg>
      </div>
    );
  }, [lossValues]);

  // ==================== 日志过滤 ====================
  const filteredLogs = useMemo(() => {
    if (!logFilter) return logs;
    const lower = logFilter.toLowerCase();
    return logs.filter(l => l.toLowerCase().includes(lower));
  }, [logs, logFilter]);

  // ==================== 格式化时间 ====================
  const formatSize = (mb: number) => mb >= 1024 ? `${(mb / 1024).toFixed(1)} GB` : `${mb.toFixed(1)} MB`;

  return (
    <>
      <h2>11. 训练管理 (Train Manager)</h2>
      <p className="desc">通过网页执行 Docker 容器内的训练脚本，实时查看日志、进度和 Loss 趋势</p>

      <div className="tm-layout">
        {/* ==================== 左侧面板 ==================== */}
        <div className="tm-sidebar">
          {/* 连接配置 */}
          <h4>管理服务</h4>
          <label>服务地址</label>
          <input type="text" value={managerUrl} onChange={e => setManagerUrl(e.target.value)} />
          <div className="tm-status-row">
            <span className={`tm-status-dot ${statusDotClass}`} />
            <span className="tm-status-text">{statusText}</span>
          </div>

          {/* 训练阶段 */}
          <h4>训练配置</h4>
          <label>训练阶段</label>
          <select value={stage} onChange={e => handleStageChange(e.target.value)}>
            {STAGES.map(s => <option key={s} value={s}>{s}</option>)}
          </select>

          {/* 模型配置 */}
          <div className="tm-field-row">
            <div>
              <label>hidden_size</label>
              <select value={hiddenSize} onChange={e => { setHiddenSize(Number(e.target.value)); setNumLayers(Number(e.target.value) === 768 ? 16 : 8); }}>
                <option value={512}>512 (Small)</option>
                <option value={768}>768 (Base)</option>
              </select>
            </div>
            <div>
              <label>num_hidden_layers</label>
              <select value={numLayers} onChange={e => setNumLayers(Number(e.target.value))}>
                <option value={8}>8</option>
                <option value={16}>16</option>
              </select>
            </div>
          </div>

          <div className="tm-switch-row">
            <span>use_moe (MoE架构)</span>
            <button
              className={`tm-toggle${useMoe ? ' active' : ''}`}
              onClick={() => setUseMoe(!useMoe)}
            />
          </div>

          {/* 训练参数 */}
          <div className="tm-field-row">
            <div>
              <label>epochs</label>
              <input type="number" min={1} value={epochs} onChange={e => setEpochs(Number(e.target.value))} />
            </div>
            <div>
              <label>batch_size</label>
              <input type="number" min={1} value={batchSize} onChange={e => setBatchSize(Number(e.target.value))} />
            </div>
          </div>
          <label>learning_rate</label>
          <input type="text" value={learningRate} onChange={e => setLearningRate(Number(e.target.value))} />

          {/* 数据集 */}
          <label>data_path (可选)</label>
          <select value={dataPath} onChange={e => setDataPath(e.target.value)}>
            <option value="">默认</option>
            {datasets.map(d => (
              <option key={d.name} value={`../dataset/${d.name}`}>{d.name} ({formatSize(d.size_mb)})</option>
            ))}
          </select>

          {/* 权重 */}
          <label>from_weight (加载基础权重)</label>
          <select value={fromWeight} onChange={e => setFromWeight(e.target.value)}>
            <option value="">无</option>
            {weights.map(w => (
              <option key={w.name} value={w.name.replace(/\.pth$/, '').replace(/_\d+(_moe)?$/, '')}>{w.name} ({formatSize(w.size_mb)})</option>
            ))}
            <option value="pretrain">pretrain</option>
            <option value="full_sft">full_sft</option>
          </select>

          <label>save_weight (自定义保存名称)</label>
          <input type="text" value={saveWeight} onChange={e => setSaveWeight(e.target.value)} placeholder="留空使用默认" />

          <label>extra_args (额外参数)</label>
          <input type="text" value={extraArgs} onChange={e => setExtraArgs(e.target.value)} placeholder="如: --max_seq_len 512" />

          {/* 操作按钮 */}
          <div style={{ marginTop: 12 }}>
            <button
              className="tm-btn tm-btn-start"
              disabled={!connected || serverStatus === 'running'}
              onClick={startTrain}
            >
              开始训练
            </button>
            <button
              className="tm-btn tm-btn-stop"
              disabled={serverStatus !== 'running'}
              onClick={stopTrain}
            >
              停止训练
            </button>
          </div>

          {/* NPU 状态 */}
          <h4>NPU 状态</h4>
          <button className="pg-quick-btn" onClick={fetchNpu} style={{ marginBottom: 6 }}>刷新</button>
          {npuRaw && <div className="tm-npu-card">{npuRaw}</div>}

          {/* 权重列表折叠 */}
          <div className="tm-weights-section">
            <div
              className={`tm-weights-toggle${weightsOpen ? ' open' : ''}`}
              onClick={() => { setWeightsOpen(!weightsOpen); fetchLists(); }}
            >
              <span className="arrow">&#9654;</span>
              权重文件 ({weights.length})
            </div>
            {weightsOpen && weights.map(w => (
              <div key={w.name} className="tm-weight-item">
                <span>{w.name}</span>
                <span className="tm-weight-size">{formatSize(w.size_mb)}</span>
              </div>
            ))}
          </div>
        </div>

        {/* ==================== 右侧日志区 ==================== */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
          {/* 进度条 */}
          {progress && (
            <div className="tm-progress">
              <div className="tm-progress-label">
                <span>Epoch {progress.epoch}/{progress.totalEpochs} Step {progress.step}/{progress.totalSteps}</span>
                <span>{progress.percent.toFixed(1)}%</span>
              </div>
              <div className="tm-progress-bar">
                <div className="tm-progress-fill" style={{ width: `${progress.percent}%` }} />
              </div>
            </div>
          )}

          {/* Loss 趋势 */}
          {lossSvg}

          {/* 日志搜索 */}
          <div className="tm-log-search">
            <input
              type="text"
              placeholder="搜索日志..."
              value={logFilter}
              onChange={e => setLogFilter(e.target.value)}
            />
            <button className="pg-quick-btn" onClick={() => setLogs([])}>清空</button>
          </div>

          {/* 日志流 */}
          <div className="tm-logs" ref={logsRef}>
            {filteredLogs.length === 0 && (
              <span style={{ color: '#64748b' }}>等待日志...</span>
            )}
            {filteredLogs.map((line, i) => {
              const isManager = line.startsWith('[Manager]') || line.startsWith('[UI]');
              const isError = /error|exception|traceback/i.test(line);
              const cls = `tm-log-line${isManager ? ' manager' : ''}${isError ? ' error' : ''}`;

              // 高亮搜索关键词
              if (logFilter) {
                const idx = line.toLowerCase().indexOf(logFilter.toLowerCase());
                if (idx >= 0) {
                  const before = line.slice(0, idx);
                  const match = line.slice(idx, idx + logFilter.length);
                  const after = line.slice(idx + logFilter.length);
                  return (
                    <div key={i} className={cls}>
                      {before}<span className="tm-log-highlight">{match}</span>{after}
                    </div>
                  );
                }
              }
              return <div key={i} className={cls}>{line}</div>;
            })}
          </div>
        </div>
      </div>
    </>
  );
}
