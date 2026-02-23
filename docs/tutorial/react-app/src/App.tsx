import { useState, useCallback, lazy, Suspense } from 'react';
import { ThemeProvider } from './context/ThemeContext';
import TopBar from './components/TopBar';

const sections = [
  lazy(() => import('./sections/TokenizationSection')),
  lazy(() => import('./sections/EmbeddingSection')),
  lazy(() => import('./sections/AttentionSection')),
  lazy(() => import('./sections/RoPESection')),
  lazy(() => import('./sections/FFNMoESection')),
  lazy(() => import('./sections/TrainingSection')),
  lazy(() => import('./sections/ForwardPassSection')),
];

export default function App() {
  const [activeTab, setActiveTab] = useState(0);
  const [mounted, setMounted] = useState<Set<number>>(() => new Set([0]));

  const handleTabChange = useCallback((index: number) => {
    setActiveTab(index);
    setMounted(prev => {
      if (prev.has(index)) return prev;
      const next = new Set(prev);
      next.add(index);
      return next;
    });
  }, []);

  return (
    <ThemeProvider>
      <TopBar activeTab={activeTab} onTabChange={handleTabChange} />
      <div className="container">
        {sections.map((Section, i) => {
          if (!mounted.has(i)) return null;
          return (
            <div
              key={i}
              className={`section${i === activeTab ? ' active' : ''}`}
              style={i !== activeTab ? { display: 'none' } : undefined}
            >
              <Suspense fallback={<div style={{ padding: 20, color: 'var(--fg2)' }}>加载中...</div>}>
                <Section />
              </Suspense>
            </div>
          );
        })}
      </div>
    </ThemeProvider>
  );
}
