import { useState, useRef, useCallback, useEffect } from 'react';

// ==================== 工具定义 ====================
const TOOLS = [
  {
    type: 'function' as const,
    function: {
      name: 'get_weather',
      description: '获取指定城市的天气信息',
      parameters: {
        type: 'object',
        properties: { city: { type: 'string', description: '城市名称' } },
        required: ['city'],
      },
    },
  },
  {
    type: 'function' as const,
    function: {
      name: 'calculate',
      description: '计算数学表达式的结果',
      parameters: {
        type: 'object',
        properties: { expression: { type: 'string', description: '数学表达式' } },
        required: ['expression'],
      },
    },
  },
  {
    type: 'function' as const,
    function: {
      name: 'search',
      description: '搜索互联网获取相关信息',
      parameters: {
        type: 'object',
        properties: { query: { type: 'string', description: '搜索关键词' } },
        required: ['query'],
      },
    },
  },
  {
    type: 'function' as const,
    function: {
      name: 'get_time',
      description: '获取当前时间或指定时区的时间',
      parameters: {
        type: 'object',
        properties: { timezone: { type: 'string', description: '时区，如Asia/Shanghai', default: 'Asia/Shanghai' } },
        required: [],
      },
    },
  },
];

// ==================== 预设模型 ====================
const PRESETS = [
  { label: 'MiniMind-MoE (vLLM)', url: '/v1/chat/completions', model: 'minimind-moe', sendTools: true },
  { label: 'MiniMind-Dense (vLLM)', url: '/v1/chat/completions', model: 'minimind', sendTools: true },
  { label: 'MiniMind (本地API:8998)', url: 'http://127.0.0.1:8998/v1/chat/completions', model: 'minimind', sendTools: true },
  { label: '自定义', url: '', model: '', sendTools: true },
];

// ==================== 类型 ====================
type CallMode = 'normal' | 'mcp';

interface ToolCall {
  name: string;
  arguments: Record<string, unknown>;
}

interface TraceStep {
  type: 'user' | 'llm' | 'tool' | 'tool_result' | 'reply';
  label: string;
  time: number; // ms since trace start
  data?: string;
}

interface CallTrace {
  steps: TraceStep[];
  totalMs: number;
  mode: CallMode;
}

interface ChatMessage {
  role: 'user' | 'assistant' | 'tool' | 'system' | 'trace';
  content: string;
  toolCalls?: ToolCall[];
  toolName?: string;
  trace?: CallTrace;
  mcpRequest?: object;
  mcpResponse?: object;
  isMcp?: boolean;
}

interface ApiToolCall {
  id?: string;
  type?: string;
  function: { name: string; arguments: string | Record<string, unknown> };
}

interface Metrics {
  detect: number;
  select: number;
  json: number;
  params: number;
  noCall: number;
  shouldCallCount: number;
  shouldNotCallCount: number;
}

// 流程动画步骤 ID
type FlowStep = 'idle' | 'user' | 'llm1' | 'tool' | 'llm2' | 'reply';

// ==================== 工具函数 ====================
function parseToolCalls(text: string): ToolCall[] {
  const re = /<tool_call>\s*([\s\S]*?)\s*<\/tool_call>/g;
  const calls: ToolCall[] = [];
  let m;
  while ((m = re.exec(text)) !== null) {
    try {
      const data = JSON.parse(m[1].trim());
      calls.push({ name: data.name || '', arguments: data.arguments || {} });
    } catch { /* skip */ }
  }
  return calls;
}

function simulateTool(name: string, args: Record<string, unknown>): string {
  switch (name) {
    case 'get_weather':
      return `${args.city || '未知'}：晴，气温25°C，湿度40%`;
    case 'calculate':
      try {
        const expr = String(args.expression || '0');
        if (/^[\d\s+\-*/().%]+$/.test(expr)) {
          return String(Function('"use strict"; return (' + expr + ')')());
        }
        return '不支持的表达式';
      } catch { return '计算错误'; }
    case 'search':
      return `搜索"${args.query || ''}"的结果：找到3条相关信息。`;
    case 'get_time':
      return `当前时间（${args.timezone || 'Asia/Shanghai'}）：${new Date().toLocaleString('zh-CN')}`;
    default:
      return `未知工具: ${name}`;
  }
}

async function mcpCallTool(name: string, args: Record<string, unknown>): Promise<{ result: string; request: object; response: object }> {
  const mcpReq = {
    jsonrpc: '2.0',
    method: 'tools/call',
    params: { name, arguments: args },
  };
  const resp = await fetch('/api/mcp/tools/call', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name, arguments: args }),
  });
  if (!resp.ok) throw new Error(`MCP HTTP ${resp.status}`);
  const data = await resp.json();
  const text = data.result?.content?.[0]?.text ?? JSON.stringify(data.result);
  return { result: text, request: mcpReq, response: data };
}

function escapeHtml(t: string): string {
  return t.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function stripToolCallTags(text: string): string {
  return text.replace(/<tool_call>[\s\S]*?<\/tool_call>/g, '').trim();
}

function fmtMs(ms: number): string {
  return ms < 1000 ? `${Math.round(ms)}ms` : `${(ms / 1000).toFixed(2)}s`;
}

// ==================== 流程动画组件 ====================
function FlowAnimation({ activeStep, mode }: { activeStep: FlowStep; mode: CallMode }) {
  const nodes: { id: FlowStep; label: string; icon: string }[] = [
    { id: 'user', label: 'User', icon: '👤' },
    { id: 'llm1', label: 'LLM', icon: '🧠' },
    { id: 'tool', label: mode === 'mcp' ? 'MCP Server' : 'Tool (本地)', icon: mode === 'mcp' ? '🔌' : '🔧' },
    { id: 'llm2', label: 'LLM', icon: '🧠' },
    { id: 'reply', label: 'Reply', icon: '💬' },
  ];

  // 步骤顺序索引（用于判断 done / active）
  const stepOrder: FlowStep[] = ['user', 'llm1', 'tool', 'llm2', 'reply'];
  const activeIdx = stepOrder.indexOf(activeStep);

  const nodeState = (id: FlowStep) => {
    const idx = stepOrder.indexOf(id);
    if (activeStep === 'idle') return '';
    if (idx < activeIdx) return 'done';
    if (idx === activeIdx) return 'active';
    return '';
  };

  const arrowState = (afterIdx: number) => {
    if (activeStep === 'idle') return '';
    if (afterIdx < activeIdx) return 'done';
    if (afterIdx === activeIdx) return 'active';
    return '';
  };

  return (
    <div className="pg-flow-body">
      {nodes.map((n, i) => (
        <div key={n.id} style={{ display: 'flex', alignItems: 'center' }}>
          <div className={`pg-flow-node ${nodeState(n.id)}`}>
            <span className="pg-flow-icon">{n.icon}</span>
            {n.label}
          </div>
          {i < nodes.length - 1 && (
            <div className={`pg-flow-arrow ${arrowState(i)}`}>
              <svg viewBox="0 0 36 16">
                <line x1="2" y1="8" x2="28" y2="8" strokeWidth="2" />
                <polygon points="28,4 34,8 28,12" />
              </svg>
            </div>
          )}
        </div>
      ))}
    </div>
  );
}

// ==================== 时间线组件 ====================
function TraceCard({ trace }: { trace: CallTrace }) {
  const dotClass = (type: TraceStep['type']) => {
    switch (type) {
      case 'tool': return 'tool';
      case 'tool_result': return trace.mode === 'mcp' ? 'mcp' : 'result';
      case 'reply': return 'reply';
      default: return '';
    }
  };

  return (
    <div className="pg-trace">
      <div className="pg-trace-title">
        调用过程
        <span className="pg-trace-total">总耗时: {fmtMs(trace.totalMs)}</span>
        {trace.mode === 'mcp' && <span className="pg-chip-badge">MCP</span>}
      </div>
      {trace.steps.map((s, i) => (
        <div key={i} className="pg-trace-step">
          <div className={`pg-trace-dot ${dotClass(s.type)}`} />
          <div className="pg-trace-line" />
          <div className="pg-trace-label">
            {s.label}
            {s.data && <> <code>{s.data}</code></>}
          </div>
          <div className="pg-trace-time">+{fmtMs(s.time)}</div>
        </div>
      ))}
    </div>
  );
}

// ==================== 组件 ====================
export default function ToolPlaygroundSection() {
  const [presetIdx, setPresetIdx] = useState(0);
  const [apiUrl, setApiUrl] = useState(PRESETS[0].url);
  const [modelName, setModelName] = useState(PRESETS[0].model);
  const [temperature, setTemperature] = useState(0.1);
  const [maxTokens, setMaxTokens] = useState(512);
  const [enabledTools, setEnabledTools] = useState<Set<string>>(() => new Set(TOOLS.map(t => t.function.name)));
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState('就绪');
  const [statusType, setStatusType] = useState<'' | 'ok' | 'error'>('');
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [inputVal, setInputVal] = useState('');

  // 新增状态
  const [callMode, setCallMode] = useState<CallMode>('normal');
  const [flowStep, setFlowStep] = useState<FlowStep>('idle');
  const [flowOpen, setFlowOpen] = useState(true);
  const [mcpTools, setMcpTools] = useState<Array<{ name: string; description: string }>>([]);

  const messagesRef = useRef<HTMLDivElement>(null);
  const conversationRef = useRef<Array<{ role: string; content: string; tool_calls?: ApiToolCall[] }>>([]);

  const scrollToBottom = useCallback(() => {
    if (messagesRef.current) {
      messagesRef.current.scrollTop = messagesRef.current.scrollHeight;
    }
  }, []);

  useEffect(() => { scrollToBottom(); }, [messages, scrollToBottom]);

  // MCP 模式切换时获取工具列表
  useEffect(() => {
    if (callMode === 'mcp') {
      fetch('/api/mcp/tools/list')
        .then(r => r.json())
        .then(data => {
          const tools = data.result?.tools || [];
          setMcpTools(tools.map((t: { name: string; description: string }) => ({ name: t.name, description: t.description })));
          setEnabledTools(new Set(tools.map((t: { name: string }) => t.name)));
        })
        .catch(() => setMcpTools([]));
    } else {
      setMcpTools([]);
      setEnabledTools(new Set(TOOLS.map(t => t.function.name)));
    }
  }, [callMode]);

  const addMsg = useCallback((msg: ChatMessage) => {
    setMessages(prev => [...prev, msg]);
  }, []);

  const setStatusMsg = useCallback((text: string, type: '' | 'ok' | 'error' = '') => {
    setStatus(text);
    setStatusType(type);
  }, []);

  const getActiveTools = useCallback(() => {
    return TOOLS.filter(t => enabledTools.has(t.function.name));
  }, [enabledTools]);

  const toggleTool = useCallback((name: string) => {
    setEnabledTools(prev => {
      const next = new Set(prev);
      if (next.has(name)) next.delete(name);
      else next.add(name);
      return next;
    });
  }, []);

  const handlePresetChange = useCallback((idx: number) => {
    setPresetIdx(idx);
    if (idx < PRESETS.length - 1) {
      setApiUrl(PRESETS[idx].url);
      setModelName(PRESETS[idx].model);
    }
  }, []);

  // ==================== API 调用 ====================
  const callAPI = useCallback(async (
    msgs: Array<{ role: string; content: string; tool_calls?: ApiToolCall[] }>,
    tools: typeof TOOLS,
  ) => {
    const body: Record<string, unknown> = {
      model: modelName,
      messages: msgs,
      temperature,
      top_p: 0.9,
      max_tokens: maxTokens,
      repetition_penalty: 1.3,
      stream: false,
    };
    const canSendTools = PRESETS[presetIdx]?.sendTools !== false;
    if (tools.length > 0 && canSendTools) body.tools = tools;

    const resp = await fetch(apiUrl, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}: ${await resp.text()}`);
    return resp.json();
  }, [apiUrl, modelName, temperature, maxTokens, presetIdx]);

  // ==================== 工具调用循环 ====================
  const sendWithTools = useCallback(async (userMsg: string, conv?: typeof conversationRef.current) => {
    const tools = getActiveTools();
    const conversation = conv || conversationRef.current;
    const isMcp = callMode === 'mcp';

    // 时间线记录
    const traceStart = performance.now();
    const traceSteps: TraceStep[] = [];
    const addTrace = (type: TraceStep['type'], label: string, data?: string) => {
      traceSteps.push({ type, label, time: performance.now() - traceStart, data });
    };

    // ① 用户输入
    setFlowStep('user');
    conversation.push({ role: 'user', content: userMsg });
    addMsg({ role: 'user', content: userMsg });
    addTrace('user', '用户输入', userMsg);

    const maxRounds = 3;
    let hadToolCall = false;

    for (let round = 0; round < maxRounds; round++) {
      // ② LLM 思考
      setFlowStep(hadToolCall ? 'llm2' : 'llm1');
      setStatusMsg(`请求中... (第${round + 1}轮)`);

      const t0 = performance.now();
      const data = await callAPI(conversation, tools);
      const llmMs = performance.now() - t0;
      const choice = data.choices[0];
      const content: string = choice.message.content || '';

      const apiToolCalls: ApiToolCall[] | null = choice.message.tool_calls || null;
      const parsedCalls = parseToolCalls(content);

      const detectedCalls: ToolCall[] = [];
      if (apiToolCalls && apiToolCalls.length > 0) {
        apiToolCalls.forEach(tc => {
          const args = typeof tc.function.arguments === 'string' ? JSON.parse(tc.function.arguments) : tc.function.arguments;
          detectedCalls.push({ name: tc.function.name, arguments: args });
        });
      } else if (parsedCalls.length > 0) {
        detectedCalls.push(...parsedCalls);
      }

      if (detectedCalls.length > 0) {
        // LLM 检测到需要工具调用
        addTrace('llm', `LLM 思考 (第${round + 1}轮)`, `检测到工具调用: ${detectedCalls.map(c => c.name).join(', ')}`);

        if (apiToolCalls && apiToolCalls.length > 0) {
          conversation.push({ role: 'assistant', content, tool_calls: apiToolCalls });
        } else {
          conversation.push({ role: 'assistant', content });
        }
        addMsg({ role: 'assistant', content, toolCalls: detectedCalls, isMcp });

        // ③ 工具调用
        setFlowStep('tool');
        for (const call of detectedCalls) {
          addTrace('tool', '工具调用', `${call.name}(${JSON.stringify(call.arguments)})`);

          const t1 = performance.now();
          let result: string;
          let mcpReq: object | undefined;
          let mcpResp: object | undefined;

          if (isMcp) {
            const mcpResult = await mcpCallTool(call.name, call.arguments);
            result = mcpResult.result;
            mcpReq = mcpResult.request;
            mcpResp = mcpResult.response;
          } else {
            result = simulateTool(call.name, call.arguments);
          }
          const toolMs = performance.now() - t1;

          addTrace('tool_result', isMcp ? `MCP 返回 (+${fmtMs(toolMs)})` : `工具返回 (+${fmtMs(toolMs)})`, result);

          conversation.push({ role: 'tool', content: result });
          addMsg({
            role: 'tool',
            content: `${call.name} → ${result}`,
            toolName: call.name,
            isMcp,
            mcpRequest: mcpReq,
            mcpResponse: mcpResp,
          });
        }
        hadToolCall = true;
        continue;
      } else {
        // ⑤ 无工具调用 — 最终回答
        addTrace('llm', hadToolCall ? `LLM 最终回答 (+${fmtMs(llmMs)})` : `LLM 回答 (+${fmtMs(llmMs)})`, content.slice(0, 60) + (content.length > 60 ? '...' : ''));
        setFlowStep('reply');
        conversation.push({ role: 'assistant', content });
        addMsg({ role: 'assistant', content });
        break;
      }
    }

    // 插入时间线
    const totalMs = performance.now() - traceStart;
    const trace: CallTrace = { steps: traceSteps, totalMs, mode: callMode };
    addMsg({ role: 'trace', content: '', trace });

    // 流程动画延迟恢复 idle
    setTimeout(() => setFlowStep('idle'), 1500);
  }, [getActiveTools, callAPI, addMsg, setStatusMsg, callMode]);

  // ==================== 发送 ====================
  const handleSend = useCallback(async () => {
    if (busy) return;
    const text = inputVal.trim();
    if (!text) return;
    setInputVal('');
    setBusy(true);
    try {
      await sendWithTools(text);
      setStatusMsg('完成', 'ok');
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      setStatusMsg('错误: ' + msg, 'error');
      addMsg({ role: 'system', content: '请求失败: ' + msg });
      setFlowStep('idle');
    }
    setBusy(false);
  }, [busy, inputVal, sendWithTools, setStatusMsg, addMsg]);

  const handleQuick = useCallback((text: string) => {
    if (busy) return;
    setInputVal('');
    setBusy(true);
    (async () => {
      try {
        await sendWithTools(text);
        setStatusMsg('完成', 'ok');
      } catch (e) {
        const msg = e instanceof Error ? e.message : String(e);
        setStatusMsg('错误: ' + msg, 'error');
        addMsg({ role: 'system', content: '请求失败: ' + msg });
        setFlowStep('idle');
      }
      setBusy(false);
    })();
  }, [busy, sendWithTools, setStatusMsg, addMsg]);

  const clearChat = useCallback(() => {
    setMessages([]);
    conversationRef.current = [];
    setMetrics(null);
    setFlowStep('idle');
    setStatusMsg('对话已清空');
  }, [setStatusMsg]);

  // ==================== 自动评测 ====================
  const runAutoTest = useCallback(async () => {
    if (busy) return;
    setBusy(true);
    setMessages([]);
    setMetrics(null);

    const tests = [
      { name: '天气查询', query: '北京今天天气怎么样？', expectedTool: 'get_weather', shouldCall: true },
      { name: '数学计算', query: '帮我算一下 123 * 456 等于多少？', expectedTool: 'calculate', shouldCall: true },
      { name: '时间查询', query: '现在几点了？', expectedTool: 'get_time', shouldCall: true },
      { name: '搜索查询', query: '帮我搜一下Python最新版本是什么？', expectedTool: 'search', shouldCall: true },
      { name: '普通对话', query: '你好，请介绍一下你自己。', expectedTool: null as string | null, shouldCall: false },
      { name: '多参数天气', query: '上海的天气如何？', expectedTool: 'get_weather', shouldCall: true },
    ];

    const results = { detect: 0, select: 0, json: 0, params: 0, noCall: 0 };
    let shouldCallCount = 0, shouldNotCallCount = 0, calledCount = 0;

    for (const tc of tests) {
      addMsg({ role: 'system', content: `--- 测试: ${tc.name} ---` });
      try {
        const tempConv: typeof conversationRef.current = [];
        await sendWithTools(tc.query, tempConv);

        const assistantMsgs = tempConv.filter(m => m.role === 'assistant');
        const allToolCalls: ToolCall[] = [];
        assistantMsgs.forEach(m => {
          if (m.tool_calls) {
            m.tool_calls.forEach(tc2 => {
              const args = typeof tc2.function.arguments === 'string'
                ? JSON.parse(tc2.function.arguments)
                : tc2.function.arguments;
              allToolCalls.push({ name: tc2.function.name, arguments: args });
            });
          }
          parseToolCalls(m.content || '').forEach(c => allToolCalls.push(c));
        });

        const hasCalled = allToolCalls.length > 0;
        const selectedTool = hasCalled ? allToolCalls[0].name : null;

        if (tc.shouldCall) {
          shouldCallCount++;
          if (hasCalled) { results.detect++; calledCount++; }
          if (selectedTool === tc.expectedTool) results.select++;
          if (hasCalled) { results.json++; results.params++; }
        } else {
          shouldNotCallCount++;
          if (!hasCalled) results.noCall++;
        }
      } catch (e) {
        const msg = e instanceof Error ? e.message : String(e);
        addMsg({ role: 'system', content: `测试失败: ${msg}` });
      }
    }

    setMetrics({ ...results, shouldCallCount, shouldNotCallCount });

    const summary = [
      '',
      '========== 评估指标汇总 ==========',
      `工具调用检测率:   ${results.detect}/${shouldCallCount} = ${(results.detect / shouldCallCount * 100).toFixed(0)}%`,
      `工具选择正确率:   ${results.select}/${shouldCallCount} = ${(results.select / shouldCallCount * 100).toFixed(0)}%`,
      `JSON有效率:       ${results.json}/${calledCount || 1} = ${(results.json / (calledCount || 1) * 100).toFixed(0)}%`,
      `参数完整率:       ${results.params}/${calledCount || 1} = ${(results.params / (calledCount || 1) * 100).toFixed(0)}%`,
      `无需工具正确率:   ${results.noCall}/${shouldNotCallCount} = ${(results.noCall / shouldNotCallCount * 100).toFixed(0)}%`,
      '==================================',
    ].join('\n');
    addMsg({ role: 'system', content: summary });
    setStatusMsg('自动评测完成', 'ok');
    setBusy(false);
  }, [busy, sendWithTools, addMsg, setStatusMsg]);

  // ==================== 渲染消息 ====================
  const renderMessage = useCallback((msg: ChatMessage, i: number) => {
    // 时间线卡片
    if (msg.role === 'trace' && msg.trace) {
      return <TraceCard key={i} trace={msg.trace} />;
    }

    const cls = `pg-msg pg-msg-${msg.role}`;
    return (
      <div key={i} className={cls}>
        <div className="pg-msg-role">
          {msg.role}
          {msg.isMcp && <span className="pg-chip-badge">MCP</span>}
        </div>
        {msg.toolCalls && msg.toolCalls.length > 0 ? (
          <>
            {msg.toolCalls.map((tc, j) => (
              <div key={j} className={`pg-tool-call${msg.isMcp ? ' mcp' : ''}`}>
                <span className="pg-tc-name">{tc.name}</span>
                <span className="pg-tc-args">({JSON.stringify(tc.arguments)})</span>
              </div>
            ))}
            {stripToolCallTags(msg.content) && (
              <div className="pg-msg-content" dangerouslySetInnerHTML={{ __html: escapeHtml(stripToolCallTags(msg.content)) }} />
            )}
          </>
        ) : msg.role === 'tool' ? (
          <>
            <div className={`pg-tool-result${msg.isMcp ? ' mcp' : ''}`}>{msg.content}</div>
            {msg.isMcp && msg.mcpRequest && (
              <details className="pg-mcp-detail">
                <summary>MCP 请求 JSON</summary>
                <pre className="pg-mcp-json">{JSON.stringify(msg.mcpRequest, null, 2)}</pre>
              </details>
            )}
            {msg.isMcp && msg.mcpResponse && (
              <details className="pg-mcp-detail">
                <summary>MCP 响应 JSON</summary>
                <pre className="pg-mcp-json">{JSON.stringify(msg.mcpResponse, null, 2)}</pre>
              </details>
            )}
          </>
        ) : (
          <div className="pg-msg-content" dangerouslySetInnerHTML={{ __html: escapeHtml(msg.content) }} />
        )}
      </div>
    );
  }, []);

  // ==================== 渲染指标 ====================
  const renderMetrics = useCallback(() => {
    if (!metrics) return null;
    const items = [
      { label: '工具检测率', value: `${(metrics.detect / metrics.shouldCallCount * 100).toFixed(0)}%` },
      { label: '选择正确率', value: `${(metrics.select / metrics.shouldCallCount * 100).toFixed(0)}%` },
      { label: 'JSON有效率', value: `${(metrics.json / (metrics.detect || 1) * 100).toFixed(0)}%` },
      { label: '参数完整率', value: `${(metrics.params / (metrics.detect || 1) * 100).toFixed(0)}%` },
      { label: '无工具正确率', value: `${(metrics.noCall / metrics.shouldNotCallCount * 100).toFixed(0)}%` },
    ];
    return (
      <div className="pg-metrics">
        {items.map((it, i) => (
          <div key={i} className="pg-metric-card">
            <div className="pg-metric-label">{it.label}</div>
            <div className="pg-metric-value">{it.value}</div>
          </div>
        ))}
      </div>
    );
  }, [metrics]);

  // 工具列表来源
  const toolList = callMode === 'mcp' && mcpTools.length > 0
    ? mcpTools
    : TOOLS.map(t => ({ name: t.function.name, description: t.function.description }));

  return (
    <>
      <h2>10. 工具测试 (Tool Calling)</h2>
      <p className="desc">测试模型的工具调用 / Function Calling 能力，支持多轮工具调用循环和自动评测</p>

      {/* 配置区 */}
      <div className="pg-config">
        <label>模型预设:</label>
        <select value={presetIdx} onChange={e => handlePresetChange(Number(e.target.value))}>
          {PRESETS.map((p, i) => <option key={i} value={i}>{p.label}</option>)}
        </select>

        <label>API地址:</label>
        <input
          type="text"
          value={apiUrl}
          onChange={e => setApiUrl(e.target.value)}
          style={{ flex: 1, minWidth: 200 }}
          readOnly={presetIdx < PRESETS.length - 1}
        />
        <label>模型:</label>
        <input
          type="text"
          value={modelName}
          onChange={e => setModelName(e.target.value)}
          style={{ width: 120 }}
          readOnly={presetIdx < PRESETS.length - 1}
        />

        <label>调用模式:</label>
        <div className="pg-mode-switch">
          <button className={callMode === 'normal' ? 'active' : ''} onClick={() => setCallMode('normal')}>普通 Tool Calling</button>
          <button className={callMode === 'mcp' ? 'active' : ''} onClick={() => setCallMode('mcp')}>MCP Tool Calling</button>
        </div>

        <div className="pg-range-group">
          <label>Temp:</label>
          <input type="range" min="0" max="1" step="0.05" value={temperature} onChange={e => setTemperature(Number(e.target.value))} />
          <span className="pg-range-val">{temperature.toFixed(2)}</span>
        </div>

        <div className="pg-range-group">
          <label>MaxTok:</label>
          <input type="range" min="64" max="2048" step="64" value={maxTokens} onChange={e => setMaxTokens(Number(e.target.value))} />
          <span className="pg-range-val">{maxTokens}</span>
        </div>
      </div>

      {/* 工具 chip */}
      <div className="pg-chips">
        <span style={{ fontSize: '0.78rem', color: 'var(--fg3)', lineHeight: '28px', marginRight: 4 }}>
          启用工具{callMode === 'mcp' ? ' (MCP)' : ''}:
        </span>
        {toolList.map(t => (
          <button
            key={t.name}
            className={`pg-chip${enabledTools.has(t.name) ? ' active' : ''}`}
            onClick={() => toggleTool(t.name)}
            title={t.description}
          >
            {t.name}
            {callMode === 'mcp' && <span className="pg-chip-badge">MCP</span>}
          </button>
        ))}
      </div>

      {/* 流程动画面板 */}
      <div className="pg-flow-panel">
        <div className={`pg-flow-header${flowOpen ? ' open' : ''}`} onClick={() => setFlowOpen(v => !v)}>
          <span><span className="arrow">▶</span> 调用流程</span>
          <span style={{ fontSize: '0.72rem', color: 'var(--fg3)' }}>
            {flowStep === 'idle' ? '等待请求...' : `步骤: ${flowStep}`}
          </span>
        </div>
        {flowOpen && <FlowAnimation activeStep={flowStep} mode={callMode} />}
      </div>

      {/* 快捷按钮 */}
      <div className="pg-quick-btns">
        <span style={{ fontSize: '0.78rem', color: 'var(--fg3)', lineHeight: '28px', marginRight: 4 }}>快捷测试:</span>
        <button className="pg-quick-btn" disabled={busy} onClick={() => handleQuick('北京今天天气怎么样？')}>天气查询</button>
        <button className="pg-quick-btn" disabled={busy} onClick={() => handleQuick('帮我算一下 123 * 456 等于多少')}>数学计算</button>
        <button className="pg-quick-btn" disabled={busy} onClick={() => handleQuick('现在几点了？')}>时间查询</button>
        <button className="pg-quick-btn" disabled={busy} onClick={() => handleQuick('帮我搜一下Python最新版本')}>搜索</button>
        <button className="pg-quick-btn" disabled={busy} onClick={() => handleQuick('你好，介绍一下你自己')}>普通对话</button>
        <button className="pg-quick-btn" disabled={busy} onClick={runAutoTest} style={{ borderColor: 'var(--accent)', color: 'var(--accent)' }}>自动评测</button>
        <button className="pg-quick-btn" onClick={clearChat} style={{ marginLeft: 'auto' }}>清空对话</button>
      </div>

      {/* 聊天区 */}
      <div className="pg-chat">
        <div className="pg-messages" ref={messagesRef}>
          {messages.length === 0 && (
            <div style={{ color: 'var(--fg3)', textAlign: 'center', padding: 40, fontSize: '0.88rem' }}>
              选择模型预设并发送消息，或点击快捷测试按钮开始
            </div>
          )}
          {messages.map(renderMessage)}
        </div>
        <div className="pg-input-area">
          <input
            type="text"
            placeholder="输入消息..."
            value={inputVal}
            onChange={e => setInputVal(e.target.value)}
            onKeyDown={e => { if (e.key === 'Enter') handleSend(); }}
            disabled={busy}
          />
          <button onClick={handleSend} disabled={busy}>发送</button>
        </div>
      </div>

      {/* 状态 */}
      <div className={`pg-status${statusType ? ' ' + statusType : ''}`}>{status}</div>

      {/* 评测指标 */}
      {renderMetrics()}
    </>
  );
}
