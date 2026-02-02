import { useState, useEffect } from 'react';
import { GlassCard } from '../ui/GlassCard';
import { useTranslation } from 'react-i18next';
import { Image as ImageIcon, Loader2, FileText, ExternalLink } from 'lucide-react';
import { cn } from '@/lib/utils';

interface TaggingPreviewGridProps {
    directory: string;
    className?: string;
    limit?: number;
}

interface TaggedImageItem {
    path: string;
    thumbnail: string;
    caption?: string;
    hasCaption: boolean;
}

export function TaggingPreviewGrid({ directory, className, limit = 20 }: TaggingPreviewGridProps) {
    const { t } = useTranslation();
    const [images, setImages] = useState<TaggedImageItem[]>([]);
    const [total, setTotal] = useState(0);
    const [currentLimit, setCurrentLimit] = useState(limit);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [editingIdx, setEditingIdx] = useState<number | null>(null);
    const [editValue, setEditValue] = useState('');

    useEffect(() => {
        setCurrentLimit(limit);
    }, [limit]);

    const fetchImages = async () => {
        if (!directory) {
            setImages([]);
            setTotal(0);
            return;
        }
        setLoading(true);
        setError(null);
        try {
            const res = await window.ipcRenderer.invoke('list-images', { dirPath: directory, limit: currentLimit });
            if (res.success) {
                setTotal(res.total || 0);

                const items = await Promise.all(res.images.map(async (path: string) => {
                    const [thumbnail, captionRes] = await Promise.all([
                        window.ipcRenderer.invoke('get-thumbnail', path),
                        window.ipcRenderer.invoke('read-caption', path)
                    ]);
                    return {
                        path,
                        thumbnail,
                        caption: captionRes.content,
                        hasCaption: captionRes.exists
                    };
                }));
                setImages(items);
            } else {
                setError(res.error || 'Failed to load images');
            }
        } catch (err) {
            console.error("Failed to fetch images:", err);
            setError('Error loading images');
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        fetchImages();
    }, [directory, currentLimit]);

    const handleSaveEdit = async (idx: number) => {
        const item = images[idx];
        const res = await window.ipcRenderer.invoke('write-caption', { imagePath: item.path, content: editValue });
        if (res.success) {
            // Update local state instead of refetching everything for smoothness
            const newImages = [...images];
            newImages[idx] = { ...item, caption: editValue, hasCaption: true };
            setImages(newImages);
            setEditingIdx(null);
        } else {
            console.error("Save failed:", res.error);
        }
    };

    if (!directory) return null;

    if (loading && images.length === 0) {
        return (
            <GlassCard className={cn("p-4 flex items-center justify-center min-h-[150px]", className)}>
                <Loader2 className="w-6 h-6 animate-spin text-primary" />
            </GlassCard>
        );
    }

    if (images.length === 0 && !loading) {
        return (
            <GlassCard className={cn("p-4 flex flex-col items-center justify-center min-h-[150px] text-muted-foreground", className)}>
                <ImageIcon className="w-8 h-8 mb-2 opacity-50" />
                <span className="text-xs">{error || t('common.no_images_found', 'No images found')}</span>
            </GlassCard>
        );
    }

    return (
        <GlassCard className={cn("p-4 space-y-4", className)}>
            <div className="flex items-center justify-between">
                <h3 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground flex items-center gap-2">
                    <FileText className="w-3.5 h-3.5" />
                    {t('common.preview_title', 'Directory Preview')} (Tagging Mode)
                    <span className="text-[10px] bg-primary/10 text-primary px-1.5 py-0.5 rounded-full">
                        {images.length} / {total}
                    </span>
                </h3>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-2 gap-4">
                {images.map((item, idx) => (
                    <div
                        key={idx}
                        className="group flex flex-row h-[180px] bg-black/20 border border-white/5 rounded-xl overflow-hidden hover:border-primary/30 transition-all duration-300"
                    >
                        {/* Image Side (1/3 width) */}
                        <div
                            className="w-[180px] min-w-[180px] relative cursor-pointer overflow-hidden border-r border-white/5"
                            onClick={() => window.ipcRenderer.invoke('open-external', item.path)}
                        >
                            <img
                                src={item.thumbnail}
                                alt={`preview-${idx}`}
                                className="w-full h-full object-cover transition-transform duration-500 group-hover:scale-105"
                                loading="lazy"
                            />
                            {/* Sequence Number */}
                            <div className="absolute top-2 left-2 px-2 py-0.5 bg-black/60 text-[11px] font-bold text-white rounded backdrop-blur-md border border-white/10">
                                #{idx + 1}
                            </div>
                            {item.hasCaption && (
                                <div className="absolute top-2 right-2 px-1.5 py-0.5 bg-green-500/80 text-[10px] font-bold text-white rounded backdrop-blur-sm uppercase">
                                    {t('toolbox.tagging.status_tagged')}
                                </div>
                            )}
                        </div>

                        {/* Caption Side (2/3 width) */}
                        <div className="flex-1 p-3.5 overflow-y-auto scrollbar-hide bg-white/[0.02] flex flex-col">
                            <div className="flex items-center justify-between mb-2.5 opacity-60 shrink-0">
                                <span className="text-xs font-mono truncate max-w-[150px] font-medium" title={item.path}>
                                    {item.path.split(/[\\\/]/).pop()}
                                </span>
                                <ExternalLink className="w-3.5 h-3.5 hover:text-primary cursor-pointer transition-colors" onClick={() => window.ipcRenderer.invoke('open-external', item.path)} />
                            </div>
                            <div className="flex-1 flex flex-col justify-center">
                                {editingIdx === idx ? (
                                    <div className="h-full flex flex-col gap-2">
                                        <textarea
                                            autoFocus
                                            className="flex-1 w-full bg-black/40 border border-primary/50 rounded p-2 text-sm font-mono outline-none resize-none scrollbar-hide"
                                            value={editValue}
                                            onChange={(e) => setEditValue(e.target.value)}
                                            onKeyDown={(e) => {
                                                if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
                                                    handleSaveEdit(idx);
                                                } else if (e.key === 'Escape') {
                                                    setEditingIdx(null);
                                                }
                                            }}
                                        />
                                        <div className="flex justify-end gap-2 shrink-0">
                                            <button
                                                onClick={() => setEditingIdx(null)}
                                                className="text-[10px] px-2.5 py-1 rounded bg-white/5 hover:bg-white/10 transition-colors"
                                            >
                                                {t('common.cancel')}
                                            </button>
                                            <button
                                                onClick={() => handleSaveEdit(idx)}
                                                className="text-[10px] px-2.5 py-1 rounded bg-primary/20 text-primary hover:bg-primary/30 transition-colors font-medium"
                                            >
                                                {t('common.confirm')} (Ctrl+Enter)
                                            </button>
                                        </div>
                                    </div>
                                ) : (
                                    <div
                                        className="cursor-text select-none h-full"
                                        onDoubleClick={() => {
                                            setEditingIdx(idx);
                                            setEditValue(item.caption || '');
                                        }}
                                    >
                                        {item.hasCaption ? (
                                            <div className="text-[13px] text-foreground/90 leading-relaxed font-mono whitespace-pre-wrap">
                                                {item.caption}
                                            </div>
                                        ) : (
                                            <div className="flex items-center justify-center text-sm text-muted-foreground/40 uppercase tracking-widest font-medium italic text-center h-full min-h-[40px]">
                                                {t('toolbox.tagging.status_untagged')}
                                            </div>
                                        )}
                                    </div>
                                )}
                            </div>
                        </div>
                    </div>
                ))}
            </div>

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
        </GlassCard>
    );
}
