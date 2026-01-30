import React from 'react';
import { cn } from '@/lib/utils';
import { Slot } from "@radix-ui/react-slot"

export interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
    asChild?: boolean;
    variant?: 'default' | 'outline' | 'ghost' | 'destructive';
    size?: 'default' | 'sm' | 'lg' | 'icon';
}

const GlassButton = React.forwardRef<HTMLButtonElement, ButtonProps>(
    ({ className, variant = "default", size = "default", asChild = false, ...props }, ref) => {
        const Comp = asChild ? Slot : "button"
        return (
            <Comp
                className={cn(
                    "inline-flex items-center justify-center whitespace-nowrap rounded-lg text-sm font-medium ring-offset-background transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50",
                    {
                        "bg-primary/90 text-primary-foreground hover:bg-primary/100 shadow-md backdrop-blur-sm": variant === 'default',
                        "border border-input bg-transparent hover:bg-accent hover:text-accent-foreground": variant === 'outline',
                        "hover:bg-accent hover:text-accent-foreground": variant === 'ghost',
                        "bg-destructive text-destructive-foreground hover:bg-destructive/90": variant === 'destructive',
                        "h-10 px-4 py-2": size === 'default',
                        "h-9 rounded-md px-3": size === 'sm',
                        "h-11 rounded-md px-8": size === 'lg',
                        "h-10 w-10": size === 'icon',
                    },
                    className
                )}
                ref={ref}
                {...props}
            />
        )
    }
)
GlassButton.displayName = "GlassButton"

export { GlassButton }
