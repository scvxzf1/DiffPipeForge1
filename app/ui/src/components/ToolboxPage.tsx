import { useState, useEffect } from 'react';
import { useTranslation } from 'react-i18next';
import { cn } from '@/lib/utils';
import { Sparkles, Layers, Filter, Hash, BarChartHorizontal } from 'lucide-react';
import { Section } from '@/components/ui/Section';
import { GeminiTagger } from './tools/GeminiTagger';
import { MaskGenerator } from './tools/MaskGenerator';
import { QualityFilter } from './tools/QualityFilter';
import { Deduplication } from './tools/Deduplication';
import { RenameTool } from './tools/RenameTool';
import { AspectRatioStats } from './tools/AspectRatioStats';
import { VideoFrameStats } from './tools/VideoFrameStats';
import { ImageConverter } from './tools/ImageConverter';
import { StyleFilter } from './tools/StyleFilter';
import { Video, RefreshCw, Scissors } from 'lucide-react';

export type ToolCategory = 'tagging' | 'mask' | 'quality' | 'dedup' | 'rename' | 'aspect' | 'video' | 'convert' | 'filter';

export function ToolboxPage() {
    const { t } = useTranslation();
    const [activeCategory, setActiveCategory] = useState<ToolCategory>('tagging');
    const [isLoaded, setIsLoaded] = useState(false);

    const categories = [
        { id: 'tagging', label: t('toolbox.categories.tagging'), icon: Sparkles },
        { id: 'mask', label: t('toolbox.categories.mask'), icon: Scissors },
        { id: 'quality', label: t('toolbox.categories.quality'), icon: Filter },
        { id: 'filter', label: t('toolbox.categories.filter'), icon: Sparkles },
        { id: 'dedup', label: t('toolbox.categories.dedup'), icon: Layers },
        { id: 'rename', label: t('toolbox.categories.rename'), icon: Hash },
        { id: 'aspect', label: t('toolbox.categories.aspect'), icon: BarChartHorizontal },
        { id: 'video', label: t('toolbox.categories.video'), icon: Video },
        { id: 'convert', label: t('toolbox.categories.convert'), icon: RefreshCw },
    ] as const;

    useEffect(() => {
        // Load initial state
        window.ipcRenderer.invoke('get-tool-settings', 'toolbox_state').then(settings => {
            if (settings && settings.activeCategory) {
                // Determine if the saved category matches one of our known categories
                // We have to cast to string for comparison or be careful with types
                const saved = settings.activeCategory as ToolCategory;
                if (categories.some(c => c.id === saved)) {
                    setActiveCategory(saved);
                }
            }
            setIsLoaded(true);
        });
    }, []);

    useEffect(() => {
        if (isLoaded) {
            window.ipcRenderer.invoke('save-tool-settings', {
                toolId: 'toolbox_state',
                settings: { activeCategory }
            });
        }
    }, [activeCategory, isLoaded]);


    return (
        <div className="flex flex-col gap-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
            <Section
                title={t('toolbox.title')}
                description={t('toolbox.description')}
            >
                <div className="flex gap-2 p-1 bg-black/20 rounded-xl w-fit mb-6 overflow-x-auto scrollbar-hide max-w-full">
                    {categories.map((cat) => (
                        <button
                            key={cat.id}
                            onClick={() => setActiveCategory(cat.id)}
                            className={cn(
                                "flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200 shrink-0",
                                activeCategory === cat.id
                                    ? "bg-primary text-primary-foreground shadow-lg"
                                    : "text-muted-foreground hover:text-foreground hover:bg-white/5"
                            )}
                        >
                            <cat.icon className="w-4 h-4" />
                            {cat.label}
                        </button>
                    ))}
                </div>

                <div className="grid grid-cols-1 gap-6">
                    {activeCategory === 'tagging' && <GeminiTagger />}
                    {activeCategory === 'mask' && <MaskGenerator />}
                    {activeCategory === 'quality' && <QualityFilter />}
                    {activeCategory === 'filter' && <StyleFilter />}
                    {activeCategory === 'dedup' && <Deduplication />}
                    {activeCategory === 'rename' && <RenameTool />}
                    {activeCategory === 'aspect' && <AspectRatioStats />}
                    {activeCategory === 'video' && <VideoFrameStats />}
                    {activeCategory === 'convert' && <ImageConverter />}
                </div>
            </Section>
        </div>
    );
}
