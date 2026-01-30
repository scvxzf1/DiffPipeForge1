import React from 'react';
import { cn } from '@/lib/utils';
import { ChevronDown } from 'lucide-react';

export interface SelectOption {
    label: string;
    value: string;
}

export interface GlassSelectProps extends React.SelectHTMLAttributes<HTMLSelectElement> {
    label?: string;
    options: SelectOption[];
    error?: string;
}

const GlassSelect = React.forwardRef<HTMLSelectElement, GlassSelectProps>(
    ({ className, label, options, error, ...props }, ref) => {
        return (
            <div className="w-full space-y-2">
                {label && (
                    <label className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70 text-gray-700 dark:text-gray-300">
                        {label}
                    </label>
                )}
                <div className="relative">
                    <select
                        className={cn(
                            "flex h-10 w-full appearance-none rounded-t-lg border-b-2 border-t-0 border-x-0 px-3 py-2 text-sm border-solid",
                            "bg-white/40 dark:bg-white/5 backdrop-blur-md",
                            "border-gray-500 dark:border-white/60",
                            "focus-visible:outline-none focus-visible:border-primary",
                            "disabled:cursor-not-allowed disabled:opacity-50",
                            "transition-all duration-200",
                            className
                        )}
                        ref={ref}
                        {...props}
                    >
                        {options.map((opt) => (
                            <option key={opt.value} value={opt.value} className="bg-white dark:bg-gray-900 text-gray-900 dark:text-gray-100">
                                {opt.label}
                            </option>
                        ))}
                    </select>
                    <ChevronDown className="absolute right-3 top-2.5 h-4 w-4 opacity-50 pointer-events-none" />
                </div>
                {error && <p className="text-xs text-red-500 font-medium ml-1">{error}</p>}
            </div>
        );
    }
);
GlassSelect.displayName = "GlassSelect";

export { GlassSelect };
