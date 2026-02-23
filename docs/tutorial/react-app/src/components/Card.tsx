import type { ReactNode } from 'react';

interface CardProps {
  title: string;
  children: ReactNode;
}

export default function Card({ title, children }: CardProps) {
  return (
    <div className="card">
      <h3>{title}</h3>
      {children}
    </div>
  );
}
