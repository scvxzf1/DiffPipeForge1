import React, { createContext, useContext, useState, useCallback } from 'react';
import { createPortal } from 'react-dom';
import { GlassCard } from './GlassCard';
import { AlertCircle, CheckCircle2 } from 'lucide-react';
import { cn } from '@/lib/utils';

type ToastType = 'error' | 'success';

interface Toast {
    id: number;
    message: string;
    type: ToastType;
}

interface GlassToastContextType {
    showToast: (message: string, type?: ToastType) => void;
}

const GlassToastContext = createContext<GlassToastContextType | undefined>(undefined);

export function useGlassToast() {
    const context = useContext(GlassToastContext);
    if (!context) {
        throw new Error('useGlassToast must be used within a GlassToastProvider');
    }
    return context;
}

// Toast Item Component to handle self-dismissal
function ToastItem({ toast, onRemove }: { toast: Toast; onRemove: (id: number) => void }) {
    React.useEffect(() => {
        const timer = setTimeout(() => {
            onRemove(toast.id);
        }, 3000);
        return () => clearTimeout(timer);
    }, [toast.id, onRemove]);

    return (
        <div className="animate-in fade-in zoom-in-95 duration-300 slide-in-from-top-2">
            <GlassCard className={cn(
                "px-4 py-2 flex items-center gap-2 backdrop-blur-md pointer-events-auto shadow-lg border",
                toast.type === 'error'
                    ? "bg-red-500/10 dark:bg-red-900/10 border-red-500/20 text-red-600 dark:text-red-400"
                    : "bg-white/20 dark:bg-white/10 border-white/20 text-gray-800 dark:text-white"
            )}>
                {toast.type === 'error' ? <AlertCircle className="w-4 h-4" /> : <CheckCircle2 className="w-4 h-4" />}
                <span className="text-sm font-medium">{toast.message}</span>
            </GlassCard>
        </div>
    );
}

export function GlassToastProvider({ children }: { children: React.ReactNode }) {
    const [toasts, setToasts] = useState<Toast[]>([]);

    const showToast = useCallback((message: string, type: ToastType = 'error') => {
        const id = Date.now() + Math.random(); // Add random to ensure uniqueness
        setToasts(prev => [...prev, { id, message, type }]);
    }, []);

    const removeToast = useCallback((id: number) => {
        setToasts(prev => prev.filter(t => t.id !== id));
    }, []);

    return (
        <GlassToastContext.Provider value={{ showToast }}>
            {children}
            {createPortal(
                <div className="fixed top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 z-[100] flex flex-col gap-2 pointer-events-none items-center">
                    {toasts.map(toast => (
                        <ToastItem key={toast.id} toast={toast} onRemove={removeToast} />
                    ))}
                </div>,
                document.body
            )}
        </GlassToastContext.Provider>
    );
}
