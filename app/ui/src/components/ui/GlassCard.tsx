import React from 'react';
import { cn } from '@/lib/utils';

interface GlassCardProps extends React.HTMLAttributes<HTMLDivElement> {
  children: React.ReactNode;
  variant?: 'default' | 'flat';
}

export function GlassCard({ children, className, variant = 'default', ...props }: GlassCardProps) {
  return (
    <div
      className={cn(
        "rounded-xl border transition-all duration-300",
        // Light mode: White with opacity + heavy blur
        // Dark mode: Black with opacity + heavy blur
        "bg-white/60 dark:bg-black/40 backdrop-blur-xl",
        "border-white/40 dark:border-white/10",
        "shadow-lg dark:shadow-none",
        variant === 'default' && "hover:shadow-xl hover:bg-white/70 dark:hover:bg-black/50",
        className
      )}
      {...props}
    >
      {children}
    </div>
  );
}
