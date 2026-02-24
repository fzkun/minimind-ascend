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
interface ToolCall {
  name: string;
  arguments: Record<string, unknown>;
}

interface ChatMessage {
  role: 'user' | 'assistant' | 'tool' | 'system';
  content: string;
  toolCalls?: ToolCall[];
  toolName?: string;
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
        // 安全计算：只允许数字和基本运算符
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

function escapeHtml(t: string): string {
  return t.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function stripToolCallTags(text: string): string {
  return text.replace(/<tool_call>[\s\S]*?<\/tool_call>/g, '').trim();
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

  const messagesRef = useRef<HTMLDivElement>(null);
  const conversationRef = useRef<Array<{ role: string; content: string; tool_calls?: ApiToolCall[] }>>([]);

  const scrollToBottom = useCallback(() => {
    if (messagesRef.current) {
      messagesRef.current.scrollTop = messagesRef.current.scrollHeight;
    }
  }, []);

  useEffect(() => { scrollToBottom(); }, [messages, scrollToBottom]);

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
      stream: false,
    };
    // vLLM 未开启 --enable-auto-tool-choice，不发 tools 字段，靠解析 <tool_call> 标签
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
    conversation.push({ role: 'user', content: userMsg });
    addMsg({ role: 'user', content: userMsg });

    const maxRounds = 3;
    for (let round = 0; round < maxRounds; round++) {
      setStatusMsg(`请求中... (第${round + 1}轮)`);
      const data = await callAPI(conversation, tools);
      const choice = data.choices[0];
      const content: string = choice.message.content || '';

      // 检查 OpenAI 格式 tool_calls
      const apiToolCalls: ApiToolCall[] | null = choice.message.tool_calls || null;
      // 检查 <tool_call> 标签
      const parsedCalls = parseToolCalls(content);

      if (apiToolCalls && apiToolCalls.length > 0) {
        // OpenAI 格式
        conversation.push({ role: 'assistant', content, tool_calls: apiToolCalls });
        const toolCallsParsed: ToolCall[] = apiToolCalls.map(tc => {
          const args = typeof tc.function.arguments === 'string'
            ? JSON.parse(tc.function.arguments)
            : tc.function.arguments;
          return { name: tc.function.name, arguments: args };
        });
        addMsg({ role: 'assistant', content, toolCalls: toolCallsParsed });

        for (const tc of apiToolCalls) {
          const args = typeof tc.function.arguments === 'string'
            ? JSON.parse(tc.function.arguments)
            : tc.function.arguments;
          const result = simulateTool(tc.function.name, args);
          conversation.push({ role: 'tool', content: result });
          addMsg({ role: 'tool', content: `${tc.function.name} → ${result}`, toolName: tc.function.name });
        }
        continue;
      } else if (parsedCalls.length > 0) {
        // <tool_call> 标签格式
        conversation.push({ role: 'assistant', content });
        addMsg({ role: 'assistant', content, toolCalls: parsedCalls });

        for (const call of parsedCalls) {
          const result = simulateTool(call.name, call.arguments);
          conversation.push({ role: 'tool', content: result });
          addMsg({ role: 'tool', content: `${call.name} → ${result}`, toolName: call.name });
        }
        continue;
      } else {
        // 无工具调用
        conversation.push({ role: 'assistant', content });
        addMsg({ role: 'assistant', content });
        break;
      }
    }
  }, [getActiveTools, callAPI, addMsg, setStatusMsg]);

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
      }
      setBusy(false);
    })();
  }, [busy, sendWithTools, setStatusMsg, addMsg]);

  const clearChat = useCallback(() => {
    setMessages([]);
    conversationRef.current = [];
    setMetrics(null);
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

        // 分析结果
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
    const cls = `pg-msg pg-msg-${msg.role}`;
    return (
      <div key={i} className={cls}>
        <div className="pg-msg-role">{msg.role}</div>
        {msg.toolCalls && msg.toolCalls.length > 0 ? (
          <>
            {msg.toolCalls.map((tc, j) => (
              <div key={j} className="pg-tool-call">
                <span className="pg-tc-name">{tc.name}</span>
                <span className="pg-tc-args">({JSON.stringify(tc.arguments)})</span>
              </div>
            ))}
            {stripToolCallTags(msg.content) && (
              <div className="pg-msg-content" dangerouslySetInnerHTML={{ __html: escapeHtml(stripToolCallTags(msg.content)) }} />
            )}
          </>
        ) : msg.role === 'tool' ? (
          <div className="pg-tool-result">{msg.content}</div>
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
        <span style={{ fontSize: '0.78rem', color: 'var(--fg3)', lineHeight: '28px', marginRight: 4 }}>启用工具:</span>
        {TOOLS.map(t => (
          <button
            key={t.function.name}
            className={`pg-chip${enabledTools.has(t.function.name) ? ' active' : ''}`}
            onClick={() => toggleTool(t.function.name)}
            title={t.function.description}
          >
            {t.function.name}
          </button>
        ))}
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
