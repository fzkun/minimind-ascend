import { useRef, useEffect, useCallback } from 'react';

export function useCanvas(
  draw: (ctx: CanvasRenderingContext2D, width: number, height: number) => void,
  deps: unknown[],
  logicalWidth: number,
  logicalHeight: number,
) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const redraw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = logicalWidth * dpr;
    canvas.height = logicalHeight * dpr;
    canvas.style.width = logicalWidth + 'px';
    canvas.style.height = logicalHeight + 'px';
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    draw(ctx, logicalWidth, logicalHeight);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [logicalWidth, logicalHeight, ...deps]);

  useEffect(() => {
    redraw();
  }, [redraw]);

  return canvasRef;
}
