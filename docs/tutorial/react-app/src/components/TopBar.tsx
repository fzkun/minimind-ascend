import { useTheme } from '../context/ThemeContext';

const TAB_LABELS = [
  '1.分词', '2.Embedding', '3.注意力', '4.RoPE',
  '5.FFN&MoE', '6.训练流程', '7.前向传播',
];

interface TopBarProps {
  activeTab: number;
  onTabChange: (index: number) => void;
}

export default function TopBar({ activeTab, onTabChange }: TopBarProps) {
  const { isDark, toggle } = useTheme();

  return (
    <div className="top-bar">
      <span className="logo">MiniMind</span>
      {TAB_LABELS.map((label, i) => (
        <button
          key={i}
          className={`tab-btn${i === activeTab ? ' active' : ''}`}
          onClick={() => onTabChange(i)}
        >
          {label}
        </button>
      ))}
      <button className="theme-toggle" onClick={toggle} title="切换主题">
        {isDark ? '☀️' : '🌙'}
      </button>
    </div>
  );
}
