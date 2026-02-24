import { useTheme } from '../context/ThemeContext';

const TAB_LABELS = [
  '0.架构总览', '1.分词', '2.词向量', '3.注意力', '4.位置编码',
  '5.前馈网络', '6.训练流程', '7.推理过程',
  '8.上线部署', '9.昇腾实战',
  '10.工具测试', '11.训练管理',
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
