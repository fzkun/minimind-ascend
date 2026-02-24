import { useTheme } from '../context/ThemeContext';

interface TabGroup {
  group: string;
  tabs: string[];
}

const TAB_GROUPS: TabGroup[] = [
  {
    group: '教学',
    tabs: ['架构总览', '分词', '词向量', '注意力', '位置编码', '前馈网络', '训练流程', '推理过程'],
  },
  {
    group: '实战',
    tabs: ['上线部署', '昇腾实战', '工具测试'],
  },
];

// 扁平化索引
const ALL_TABS = TAB_GROUPS.flatMap(g => g.tabs);

interface TopBarProps {
  activeTab: number;
  onTabChange: (index: number) => void;
}

export default function TopBar({ activeTab, onTabChange }: TopBarProps) {
  const { isDark, toggle } = useTheme();

  let globalIdx = 0;

  return (
    <div className="top-bar">
      <span className="logo">MiniMind</span>
      {TAB_GROUPS.map((group) => (
        <div key={group.group} className="tab-group">
          <span className="tab-group-label">{group.group}</span>
          <span className="tab-group-sep">|</span>
          {group.tabs.map((label) => {
            const idx = globalIdx++;
            return (
              <button
                key={idx}
                className={`tab-btn${idx === activeTab ? ' active' : ''}`}
                onClick={() => onTabChange(idx)}
              >
                {label}
              </button>
            );
          })}
        </div>
      ))}
      <button className="theme-toggle" onClick={toggle} title="切换主题">
        {isDark ? '☀️' : '🌙'}
      </button>
    </div>
  );
}

export { ALL_TABS };
