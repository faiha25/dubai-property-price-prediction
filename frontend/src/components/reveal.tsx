import type { ReactNode, CSSProperties } from "react";

/**
 * Fade-and-rise entrance driven purely by a CSS @keyframes animation
 * (see .reveal in globals.css) — no JS timer or scroll observer
 * involved, so it can't get stuck if a tab throttles rAF/setTimeout,
 * and it degrades to fully visible under prefers-reduced-motion.
 */
export default function Reveal({
  children,
  delay = 0,
  className,
}: {
  children: ReactNode;
  delay?: number;
  className?: string;
}) {
  const style: CSSProperties = { animationDelay: `${delay}s` };

  return (
    <div className={`reveal ${className ?? ""}`} style={style}>
      {children}
    </div>
  );
}
