interface TokenBoxProps {
  text: string;
  color: string;
  title?: string;
  style?: React.CSSProperties;
}

export default function TokenBox({ text, color, title, style }: TokenBoxProps) {
  return (
    <span
      className="token-box"
      style={{ background: color, color: '#fff', ...style }}
      title={title}
    >
      {text}
    </span>
  );
}
