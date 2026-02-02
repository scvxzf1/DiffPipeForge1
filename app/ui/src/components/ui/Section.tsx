import React from 'react';

interface SectionProps {
    title: string;
    description?: string;
    children: React.ReactNode;
    className?: string;
}

export function Section({ title, description, children, className }: SectionProps) {
    return (
        <div className={className}>
            <div className="mb-6">
                <h2 className="text-2xl font-bold tracking-tight mb-1">{title}</h2>
                {description && <p className="text-muted-foreground text-sm">{description}</p>}
            </div>
            {children}
        </div>
    );
}
