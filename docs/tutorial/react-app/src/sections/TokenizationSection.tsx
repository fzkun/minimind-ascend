import { useState, useRef, useEffect, useCallback } from 'react';
import Card from '../components/Card';
import SourcePanel from '../components/SourcePanel';
import { tokenColor } from '../utils';

// BPE merge simulation
function computeBPESteps(): string[][] {
  const text = '学习人工智能';
  const steps: string[][] = [];
  let tokens = text.split('');
  steps.push(tokens.slice());
  const merges: [string[], string | null][] = [
    [['人', '工'], '人工'],
    [['智', '能'], '智能'],
    [['人工', '智', '能'], null],
    [['人工', '智能'], '人工智能'],
    [['学', '习'], '学习'],
    [['学习', '人工智能'], null],
  ];
  for (const [pair, result] of merges) {
    if (!result) continue;
    const newTokens: string[] = [];
    let i = 0;
    while (i < tokens.length) {
      if (i < tokens.length - 1 && tokens[i] === pair[0] && tokens[i + 1] === pair[1]) {
        newTokens.push(result);
        i += 2;
      } else {
        newTokens.push(tokens[i]);
        i++;
      }
    }
    tokens = newTokens;
    steps.push(tokens.slice());
  }
  return steps;
}

const BPE_STEPS = computeBPESteps();

// Token vocabulary
const VOCAB: Record<string, number> = {
  '你': 340, '好': 590, '我': 280, '是': 460, '的': 350, '了': 370, '不': 410,
  '在': 430, '有': 450, '这': 470, '人': 310, '大': 480, '中': 300, '上': 490,
  '一': 260, '个': 500, '到': 510, '来': 520, '时': 530, '会': 540, '说': 550,
  '他': 560, '她': 570, '地': 580, '出': 600, '就': 610, '也': 620, '和': 630,
  '对': 640, '能': 650, '都': 660, '学': 670, '习': 680, '工': 690, '智': 700,
  '机': 710, '器': 720, '语': 730, '言': 740, '模': 750, '型': 760,
  '世': 770, '界': 780, '想': 790, '要': 800, '生': 810, '活': 820,
  '做': 830, '可': 840, '以': 850, '吗': 860, '什': 870, '么': 880,
  '怎': 890, '样': 900, '很': 910, '多': 920, '最': 930, '小': 940,
  '天': 950, '年': 960, '月': 970, '日': 980, '。': 16, '，': 14, '？': 33, '！': 3,
  '、': 100, '\u201c': 4, '\u201d': 101, '（': 10, '）': 11, '\n': 201,
  'Mini': 1100, 'Mind': 1200, 'mind': 1201, 'Hello': 1300, 'World': 1400,
  'the': 400, 'is': 461, 'a': 66, 'of': 402, 'to': 403, 'and': 501,
  'in': 404, 'that': 405, 'I': 44, 'you': 406, 'it': 407,
  ' ': 220, '.': 16, ',': 14, '?': 33, '!': 3,
};
const COMPOUND: Record<string, number> = {
  '你好': 1500, '什么': 1501, '怎样': 1502, '可以': 1503,
  '人工': 1504, '智能': 1505, '学习': 1506, '语言': 1507,
  '模型': 1508, '世界': 1509, '生活': 1510, 'MiniMind': 1511,
  '人工智能': 1600,
};

function tokenize(text: string): { text: string; id: number }[] {
  const tokens: { text: string; id: number }[] = [];
  let i = 0;
  while (i < text.length) {
    let matched = false;
    for (let len = Math.min(8, text.length - i); len > 0; len--) {
      const sub = text.substring(i, i + len);
      if (COMPOUND[sub] !== undefined) {
        tokens.push({ text: sub, id: COMPOUND[sub] });
        i += len;
        matched = true;
        break;
      }
      if (len <= 4 && VOCAB[sub] !== undefined) {
        tokens.push({ text: sub, id: VOCAB[sub] });
        i += len;
        matched = true;
        break;
      }
    }
    if (!matched) {
      tokens.push({ text: text[i], id: (text.charCodeAt(i) % 256) + 3 });
      i++;
    }
  }
  return tokens;
}

export default function TokenizationSection() {
  // BPE state
  const [bpeStep, setBpeStep] = useState(0);
  const autoRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const [autoPlaying, setAutoPlaying] = useState(false);

  // Realtime tokenizer state
  const [inputText, setInputText] = useState('你好，我是MiniMind。');

  const stopAuto = useCallback(() => {
    if (autoRef.current) {
      clearInterval(autoRef.current);
      autoRef.current = null;
    }
    setAutoPlaying(false);
  }, []);

  useEffect(() => {
    return () => {
      if (autoRef.current) clearInterval(autoRef.current);
    };
  }, []);

  const handleAuto = useCallback(() => {
    if (autoRef.current) {
      stopAuto();
      return;
    }
    setAutoPlaying(true);
    autoRef.current = setInterval(() => {
      setBpeStep(prev => {
        if (prev < BPE_STEPS.length - 1) return prev + 1;
        stopAuto();
        return prev;
      });
    }, 800);
  }, [stopAuto]);

  const handleReset = useCallback(() => {
    stopAuto();
    setBpeStep(0);
  }, [stopAuto]);

  const currentTokens = BPE_STEPS[bpeStep];
  const realtimeTokens = tokenize(inputText);

  return (
    <>
      <h2>1. 分词过程 (Tokenization)</h2>
      <p className="desc">
        将原始文本拆分为 token 序列，是 LLM 处理语言的第一步。MiniMind 使用 BPE（字节对编码）分词器，词表大小 vocab_size = 6400。
      </p>

      <Card title="BPE 合并动画">
        <p style={{ marginBottom: 10, fontSize: '0.9rem', color: 'var(--fg2)' }}>
          BPE（Byte Pair Encoding）从字符级开始，统计所有相邻 token 对的出现频率，将最高频的一对合并为新 token，反复迭代直到达到目标词表大小。
          下方以"学习人工智能"为例，点击按钮逐步观看合并过程：
        </p>
        <div
          style={{
            minHeight: 50, padding: 10, background: 'var(--bg)', borderRadius: 'var(--radius)',
            border: '1px solid var(--border)', marginBottom: 10,
            display: 'flex', flexWrap: 'wrap', gap: 2, alignItems: 'center',
          }}
        >
          {currentTokens.map((t, i) => (
            <span
              key={`${bpeStep}-${i}`}
              className="token-box"
              style={{ background: tokenColor(i), color: '#fff', fontSize: '1rem', padding: '6px 12px' }}
            >
              {t}
            </span>
          ))}
        </div>
        <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap' }}>
          <button className="btn" onClick={() => setBpeStep(s => Math.max(0, s - 1))}>◀ 上一步</button>
          <button className="btn primary" onClick={() => setBpeStep(s => Math.min(BPE_STEPS.length - 1, s + 1))}>下一步 ▶</button>
          <button className="btn" onClick={handleAuto}>{autoPlaying ? '⏸ 暂停' : '▶ 自动播放'}</button>
          <button className="btn" onClick={handleReset}>重置</button>
          <span style={{ fontSize: '0.8rem', color: 'var(--fg3)' }}>
            步骤 {bpeStep}/{BPE_STEPS.length - 1}
            {bpeStep > 0
              ? ` — 合并后 ${currentTokens.length} 个 token`
              : ` — ${currentTokens.length} 个字符`}
          </span>
        </div>
        {bpeStep > 0 && (
          <p style={{ marginTop: 8, fontSize: '0.85rem', color: 'var(--fg2)', fontStyle: 'italic' }}>
            {bpeStep === 1 && '第 1 步：发现"人"+"工"是最高频相邻对，合并为"人工"。这一步减少了 1 个 token。'}
            {bpeStep === 2 && '第 2 步：发现"智"+"能"是下一个最高频对，合并为"智能"。'}
            {bpeStep === 3 && '第 3 步：发现"人工"+"智能"可以继续合并，得到"人工智能"。多次合并可以形成更长的子词。'}
            {bpeStep === 4 && '第 4 步：最终"学"+"习"也被合并为"学习"，6 个字符被压缩为 2 个 token。BPE 的压缩率取决于训练语料中的词频统计。'}
          </p>
        )}
        <SourcePanel
          title="对照源码：dataset/lm_dataset.py:31-49 (PretrainDataset)"
          code={`class PretrainDataset(Dataset):
    """预训练数据集：将原始文本转为 token 序列"""
    def __init__(self, data_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer           # BPE 分词器，词表大小 6400
        self.max_length = max_length         # 每条样本最大长度（含特殊 token）
        # 使用 HuggingFace datasets 加载 JSONL 文件，每行一个 {"text": "..."} 对象
        self.samples = load_dataset('json', data_files=data_path, split='train')

    def __getitem__(self, index):
        sample = self.samples[index]
        # 调用分词器进行 BPE 编码
        # add_special_tokens=False: 不自动加 BOS/EOS，后面手动添加
        # truncation=True: 超长文本截断到 max_length-2（预留 BOS+EOS 位置）
        tokens = self.tokenizer(
            str(sample['text']),
            add_special_tokens=False,
            max_length=self.max_length - 2,
            truncation=True
        ).input_ids
        # 手动拼接: [BOS] + 正文 tokens + [EOS]
        tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]
        # 右侧填充 PAD 到固定长度 max_length，使 batch 内所有样本等长
        input_ids = tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens))
        # labels 与 input_ids 相同，但 PAD 位置标记为 -100
        # PyTorch CrossEntropyLoss 会自动忽略 -100 标签，不对 PAD 计算损失
        labels = input_ids.clone()
        labels[input_ids == self.tokenizer.pad_token_id] = -100
        return input_ids, labels`}
        />
      </Card>

      <Card title="实时分词演示">
        <p style={{ marginBottom: 8, fontSize: '0.9rem', color: 'var(--fg2)' }}>
          在下方输入文本，实时查看分词结果。本演示使用内嵌的高频 token 子集（约 120 个词元）进行最长前缀匹配模拟。
          每个彩色方块代表一个 token，悬停可查看对应的 ID。分词器会优先匹配最长的复合 token（如"人工智能"→1600），
          未知字符回退到 Unicode 字节编码。
        </p>
        <textarea
          value={inputText}
          onChange={e => setInputText(e.target.value)}
          placeholder="输入中文或英文文本，例如：你好世界 Hello World"
        />
        <div style={{ marginTop: 10 }}>
          <div className="label">Token 序列：</div>
          <div
            style={{
              minHeight: 36, padding: 8, background: 'var(--bg)',
              borderRadius: 'var(--radius)', border: '1px solid var(--border)',
            }}
          >
            {realtimeTokens.map((t, i) => (
              <span
                key={i}
                className="token-box"
                style={{ background: tokenColor(i), color: '#fff' }}
                title={`ID: ${t.id}`}
              >
                {t.text === ' ' ? '␣' : t.text === '\n' ? '↵' : t.text}
              </span>
            ))}
          </div>
        </div>
        <div style={{ marginTop: 8 }}>
          <div className="label">Token IDs：</div>
          <div style={{ fontFamily: 'monospace', fontSize: '0.85rem', color: 'var(--accent)', wordBreak: 'break-all' }}>
            [{realtimeTokens.map(t => t.id).join(', ')}]
          </div>
        </div>
      </Card>
    </>
  );
}
