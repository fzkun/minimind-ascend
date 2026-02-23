import { useState } from 'react';

interface SourcePanelProps {
  title: string;
  code: string;
}

export default function SourcePanel({ title, code }: SourcePanelProps) {
  const [open, setOpen] = useState(false);

  return (
    <div className="source-panel">
      <div
        className={`source-toggle${open ? ' open' : ''}`}
        onClick={() => setOpen(v => !v)}
      >
        <span className="arrow">▶</span> {title}
      </div>
      <pre className={`source-content${open ? ' open' : ''}`}>
        <code>{code}</code>
      </pre>
    </div>
  );
}
