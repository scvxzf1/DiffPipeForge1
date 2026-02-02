import { useState, useEffect } from 'react';
import { GlassCard } from '../ui/GlassCard';
import { useTranslation } from 'react-i18next';
import { Image as ImageIcon, Loader2 } from 'lucide-react';
import { cn } from '@/lib/utils';

interface ImagePreviewGridProps {
    directory: string;
    className?: string;
    limit?: number;
}

interface ImageItem {
    path: string;
    thumbnail: string;
}

export function ImagePreviewGrid({ directory, className, limit = 20 }: ImagePreviewGridProps) {
    const { t } = useTranslation();
    const [images, setImages] = useState<ImageItem[]>([]);
    const [total, setTotal] = useState(0);
    const [currentLimit, setCurrentLimit] = useState(limit);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        setCurrentLimit(limit);
    }, [limit]);

    useEffect(() => {
        if (!directory) {
            setImages([]);
            setTotal(0);
            return;
        }

        const fetchImages = async () => {
            setLoading(true);
            setError(null);
            try {
                const res = await window.ipcRenderer.invoke('list-images', { dirPath: directory, limit: currentLimit });
                if (res.success) {
                    setTotal(res.total || 0);
                    // Fetch thumbnails for better performance
                    const items = await Promise.all(res.images.map(async (path: string) => {
                        const thumbnail = await window.ipcRenderer.invoke('get-thumbnail', path);
                        return { path, thumbnail };
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

        fetchImages();
    }, [directory, currentLimit]);

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
        <GlassCard className={cn("p-4 space-y-3", className)}>
            <div className="flex items-center justify-between">
                <h3 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground flex items-center gap-2">
                    <ImageIcon className="w-3.5 h-3.5" />
                    {t('common.preview_title', 'Directory Preview')}
                    <span className="text-[10px] bg-primary/10 text-primary px-1.5 py-0.5 rounded-full">
                        {images.length} / {total}
                    </span>
                </h3>
            </div>

            <div className="grid grid-cols-5 md:grid-cols-8 lg:grid-cols-10 gap-2">
                {images.map((item, idx) => (
                    <div
                        key={idx}
                        className="aspect-square rounded-md overflow-hidden bg-black/20 border border-white/5 relative group cursor-pointer"
                        onClick={() => window.ipcRenderer.invoke('open-external', item.path)} // Open original path
                        title={item.path}
                    >
                        <img
                            src={item.thumbnail}
                            alt={`preview-${idx}`}
                            className="w-full h-full object-cover transition-transform duration-300 group-hover:scale-110"
                            loading="lazy"
                        />
                        {/* Hover Overlay */}
                        <div className="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
                            <ImageIcon className="w-4 h-4 text-white/80" />
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
