import React from 'react';
import { GlassCard } from './GlassCard';
import { GlassButton } from './GlassButton';

interface GlassConfirmDialogProps {
    isOpen: boolean;
    onClose: () => void;
    onConfirm: () => void;
    title: React.ReactNode;
    description: React.ReactNode;
    confirmText?: string;
    cancelText?: string;
}

export function GlassConfirmDialog({
    isOpen,
    onClose,
    onConfirm,
    title,
    description,
    confirmText = "Confirm",
    cancelText = "Cancel",
    hideConfirm = false
}: GlassConfirmDialogProps & { hideConfirm?: boolean }) {
    if (!isOpen) return null;

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm">
            <GlassCard className="w-full max-w-md p-6 m-4 animate-in fade-in zoom-in-95 duration-200">
                <div className="mb-4">
                    <h3 className="text-xl font-bold mb-2">{title}</h3>
                    <div className="text-sm text-muted-foreground">{description}</div>
                </div>
                <div className="flex justify-end gap-3">
                    <GlassButton variant="ghost" onClick={onClose}>
                        {cancelText}
                    </GlassButton>
                    {!hideConfirm && (
                        <GlassButton variant="destructive" onClick={() => {
                            onConfirm();
                            onClose();
                        }}>
                            {confirmText}
                        </GlassButton>
                    )}
                </div>
            </GlassCard>
        </div>
    );
}
