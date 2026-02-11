import { useState, useEffect, useCallback } from 'react';
import { GlassCard } from '../ui/GlassCard';
import { useTranslation } from 'react-i18next';
import { Image as ImageIcon, Loader2, CheckCircle2, RotateCcw } from 'lucide-react';
import { cn } from '@/lib/utils';
import { toast } from 'sonner';

interface ImagePreviewGridProps {
    directory: string;
    className?: string;
    limit?: number;
    title?: string;
    autoRefresh?: boolean;
    refreshInterval?: number;
    isRestorable?: boolean;
    showMasks?: boolean;
    maskDirName?: string;
    overrideMaskPath?: string;
    refreshTrigger?: number | boolean;
}

interface ImageItem {
    path: string;
    thumbnail: string;
    maskThumbnail?: string;
}

export function ImagePreviewGrid({
    directory,
    className,
    limit = 20,
    title,
    autoRefresh = false,
    refreshInterval = 5000,
    isRestorable = false,
    showMasks = false,
    maskDirName,
    overrideMaskPath,
    refreshTrigger
}: ImagePreviewGridProps) {
    const { t } = useTranslation();
    const [images, setImages] = useState<ImageItem[]>([]);
    const [selectedFiles, setSelectedFiles] = useState<Set<string>>(new Set());
    const [isRestoring, setIsRestoring] = useState(false);
    const [total, setTotal] = useState(0);
    const [currentLimit, setCurrentLimit] = useState(limit);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [lastSelectedIndex, setLastSelectedIndex] = useState<number | null>(null);

    // Reset selection and limits when directory changes
    useEffect(() => {
        setImages([]);
        setSelectedFiles(new Set());
        setLastSelectedIndex(null);
        setCurrentLimit(limit);
        setTotal(0);
        setError(null);
    }, [directory, limit]);

    const loadImages = useCallback(async (isBackground = false) => {
        if (!directory) return;

        if (!isBackground) setLoading(true);
        if (!isBackground) setError(null);

        try {
            const res = await window.ipcRenderer.invoke('list-images', { dirPath: directory, limit: currentLimit });
            if (res.success) {
                setTotal(res.total || 0);
                const items = await Promise.all(res.images.map(async (path: string) => {
                    const thumbnail = await window.ipcRenderer.invoke('get-thumbnail', path);
                    let maskThumbnail = undefined;

                    if (showMasks) {
                        try {
                            // Construct mask path: dir/masks/filename.png
                            // We need to handle path joining properly
                            // For now, let's assume we can invoke a backend helper or do simple string manipulation if guaranteed standard paths
                            // But 'get-mask-thumbnail' effectively does this logic
                            const maskRes = await window.ipcRenderer.invoke('get-mask-thumbnail', {
                                originalPath: path,
                                maskDirName,
                                overrideMaskPath
                            });
                            if (maskRes.success) {
                                maskThumbnail = maskRes.thumbnail;
                            }
                        } catch (e) {
                            // ignore missing masks
                        }
                    }

                    return { path, thumbnail, maskThumbnail };
                }));
                setImages(items);
            } else {
                if (!isBackground) setError(res.error || 'Failed to load images');
            }
        } catch (err) {
            console.error("Failed to fetch images:", err);
            if (!isBackground) setError('Error loading images');
        } finally {
            if (!isBackground) setLoading(false);
        }
    }, [directory, currentLimit]);

    // Initial load and Auto-refresh
    useEffect(() => {
        loadImages();

        let intervalId: any;
        if (autoRefresh) {
            intervalId = setInterval(() => {
                loadImages(true);
            }, refreshInterval);
        }

        return () => {
            if (intervalId) clearInterval(intervalId);
        };
    }, [loadImages, autoRefresh, refreshInterval, refreshTrigger]);

    if (!directory) return null;

    if (loading && images.length === 0) {
        return (
            <GlassCard className={cn("p-4 flex items-center justify-center min-h-[100px]", className)}>
                <Loader2 className="w-6 h-6 animate-spin text-primary" />
            </GlassCard>
        );
    }

    if (images.length === 0 && !loading) {
        return (
            <GlassCard className={cn("p-4 flex flex-col items-center justify-center min-h-[100px] text-muted-foreground", className)}>
                <ImageIcon className="w-8 h-8 mb-2 opacity-50" />
                <span className="text-xs">{error || t('common.no_images_found', 'No images found')}</span>
            </GlassCard>
        );
    }

    return (
        <GlassCard className={cn("p-4 space-y-3 relative", className)}>
            <div className="flex items-center justify-between">
                <h3 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground flex items-center gap-2">
                    <ImageIcon className="w-3.5 h-3.5" />
                    {title || t('common.preview_title', 'Directory Preview')}
                    <span className="text-[10px] bg-primary/10 text-primary px-1.5 py-0.5 rounded-full">
                        {images.length} / {total}
                    </span>
                    {isRestorable && (
                        <span className="text-[10px] text-muted-foreground/70 ml-2 font-normal">
                            {t('common.shift_click_hint')}
                        </span>
                    )}
                </h3>
            </div>

            {showMasks ? (
                /* Paired layout: 2 pairs per row, each pair = original + mask side by side */
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 select-none">
                    {images.map((item, idx) => (
                        <div
                            key={idx}
                            className="flex gap-2 rounded-lg overflow-hidden border border-white/5 bg-black/20 p-2 group"
                            title={item.path}
                        >
                            {/* Original Image */}
                            <div className="flex-1 flex flex-col gap-1">
                                <span className="text-[10px] text-muted-foreground/60 truncate px-0.5">{t('toolbox.mask.original')}</span>
                                <div
                                    className="aspect-square rounded-md overflow-hidden cursor-pointer bg-black/30"
                                    onClick={() => window.ipcRenderer.invoke('open-external', item.path)}
                                >
                                    <img
                                        src={item.thumbnail}
                                        alt={`original-${idx}`}
                                        className="w-full h-full object-cover transition-transform duration-300 group-hover:scale-105"
                                        loading="lazy"
                                    />
                                </div>
                            </div>
                            {/* Mask Image */}
                            <div className="flex-1 flex flex-col gap-1">
                                <span className="text-[10px] text-muted-foreground/60 truncate px-0.5">{t('toolbox.mask.mask_label')}</span>
                                <div className="aspect-square rounded-md overflow-hidden bg-black/30 flex items-center justify-center">
                                    {item.maskThumbnail ? (
                                        <img
                                            src={item.maskThumbnail}
                                            alt={`mask-${idx}`}
                                            className="w-full h-full object-cover transition-transform duration-300 group-hover:scale-105"
                                            loading="lazy"
                                        />
                                    ) : (
                                        <span className="text-[10px] text-muted-foreground/40 italic">{t('toolbox.mask.no_mask')}</span>
                                    )}
                                </div>
                            </div>
                        </div>
                    ))}
                </div>
            ) : (
                /* Original dense grid layout */
                <div className="grid grid-cols-5 md:grid-cols-8 lg:grid-cols-10 gap-2 select-none">
                    {images.map((item, idx) => (
                        <div
                            key={idx}
                            className={cn(
                                "aspect-square rounded-md overflow-hidden border relative group cursor-pointer transition-all duration-200",
                                selectedFiles.has(item.path)
                                    ? "border-primary ring-2 ring-primary bg-primary/20"
                                    : "bg-black/20 border-white/5"
                            )}
                            onClick={(e) => {
                                if (isRestorable) {
                                    const newSet = new Set(selectedFiles);

                                    if (e.shiftKey && lastSelectedIndex !== null) {
                                        const start = Math.min(lastSelectedIndex, idx);
                                        const end = Math.max(lastSelectedIndex, idx);
                                        for (let i = start; i <= end; i++) {
                                            newSet.add(images[i].path);
                                        }
                                        setLastSelectedIndex(idx);
                                    } else {
                                        if (newSet.has(item.path)) {
                                            newSet.delete(item.path);
                                        } else {
                                            newSet.add(item.path);
                                        }
                                        setLastSelectedIndex(idx);
                                    }

                                    setSelectedFiles(newSet);
                                } else {
                                    window.ipcRenderer.invoke('open-external', item.path);
                                }
                            }}
                            title={item.path}
                        >
                            <img
                                src={item.thumbnail}
                                alt={`preview-${idx}`}
                                className={cn(
                                    "w-full h-full object-cover transition-transform duration-300",
                                    !selectedFiles.has(item.path) && "group-hover:scale-110",
                                    selectedFiles.has(item.path) && "opacity-80 scale-95"
                                )}
                                loading="lazy"
                            />

                            {/* Hover Overlay / Selection Indicator */}
                            <div className={cn(
                                "absolute inset-0 transition-opacity flex items-center justify-center",
                                selectedFiles.has(item.path) ? "opacity-100 bg-primary/20" : "opacity-0 bg-black/40 group-hover:opacity-100"
                            )}>
                                {isRestorable ? (
                                    <div className={cn(
                                        "w-6 h-6 rounded-full border-2 flex items-center justify-center transition-all",
                                        selectedFiles.has(item.path)
                                            ? "bg-primary border-primary text-primary-foreground"
                                            : "border-white/60 hover:border-white hover:bg-white/20"
                                    )}>
                                        {selectedFiles.has(item.path) && <CheckCircle2 className="w-4 h-4" />}
                                    </div>
                                ) : (
                                    <ImageIcon className="w-4 h-4 text-white/80" />
                                )}
                            </div>
                        </div>
                    ))}
                </div>
            )}

            {images.length < total && (
                <div className="flex justify-center pt-2">
                    <button
                        onClick={() => setCurrentLimit(prev => prev + 20)}
                        disabled={loading}
                        className="text-xs text-primary hover:text-primary/80 transition-colors flex items-center gap-1.5 focus:outline-none disabled:opacity-50"
                    >
                        {loading ? (
                            <Loader2 className="w-3 h-3 animate-spin" />
                        ) : (
                            <span className="flex items-center gap-1">
                                {t('common.load_more', 'Load More')} ({total - images.length})
                            </span>
                        )}
                    </button>
                </div>
            )}

            {/* Restore Button Overlay */}
            {selectedFiles.size > 0 && isRestorable && (
                <div className="absolute bottom-4 right-4 z-20 animate-in fade-in slide-in-from-bottom-4">
                    <button
                        onClick={async () => {
                            setIsRestoring(true);
                            try {
                                const filesToRestore = Array.from(selectedFiles);
                                const result = await window.ipcRenderer.invoke('restore-files', filesToRestore);
                                if (result.success) {
                                    toast.success(t('common.restore_success', { count: result.count }));
                                    setSelectedFiles(new Set());
                                    setLastSelectedIndex(null);

                                    // Trigger immediate refresh after restore
                                    await loadImages();
                                } else {
                                    toast.error(t('common.restore_failed', { error: result.error }));
                                }
                            } catch (e) {
                                console.error(e);
                                toast.error(t('common.error'));
                            } finally {
                                setIsRestoring(false);
                            }
                        }}
                        disabled={isRestoring}
                        className="bg-primary text-primary-foreground px-4 py-2 rounded-full shadow-lg text-sm font-semibold flex items-center gap-2 hover:bg-primary/90 transition-colors"
                    >
                        {isRestoring ? <Loader2 className="w-4 h-4 animate-spin" /> : <RotateCcw className="w-4 h-4" />}
                        {t('common.restore_selected', { count: selectedFiles.size })}
                    </button>
                </div>
            )}
        </GlassCard>
    );
}
