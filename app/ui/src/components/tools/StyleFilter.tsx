import { useState, useEffect, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import { GlassCard } from '@/components/ui/GlassCard';
import { GlassInput } from '@/components/ui/GlassInput';
import { GlassButton } from '@/components/ui/GlassButton';
import { Play, Square, Loader2, FolderOpen, RefreshCcw } from 'lucide-react';
import { useGlassToast } from '../ui/GlassToast';

interface FilterSettings {
    imageDir: string;
    keep: string;
    remove: string;
    batchSize: string;
    threads: string;
}

const DEFAULT_SETTINGS: FilterSettings = {
    imageDir: '',
    keep: 'anime, 2d illustration, drawing, flat color',
    remove: 'photorealistic, realistic, photo, 3d render, real person',
    batchSize: '32',
    threads: '8'
};

export function StyleFilter() {
    const { t } = useTranslation();
    const { showToast } = useGlassToast();
    const [settings, setSettings] = useState<FilterSettings>(DEFAULT_SETTINGS);
    const [isRunning, setIsRunning] = useState(false);
    const [logs, setLogs] = useState<string[]>([]);
    const scrollRef = useRef<HTMLDivElement>(null);

    // Persist settings
    useEffect(() => {
        const saved = localStorage.getItem('style-filter-settings');
        if (saved) {
            try {
                setSettings({ ...DEFAULT_SETTINGS, ...JSON.parse(saved) });
            } catch (e) {
                console.error("Failed to load settings:", e);
            }
        }
    }, []);

    useEffect(() => {
        localStorage.setItem('style-filter-settings', JSON.stringify(settings));
    }, [settings]);

    // Logs auto-scroll
    useEffect(() => {
        if (scrollRef.current) {
            scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
        }
    }, [logs]);

    const handleSelectDir = async () => {
        const path = await window.ipcRenderer.invoke('open-directory-dialog');
        if (path) {
            setSettings(prev => ({ ...prev, imageDir: path }));
        }
    };

    const handleStart = async () => {
        if (!settings.imageDir) {
            showToast(t('toolbox.errors.no_dir'), 'error');
            return;
        }

        setIsRunning(true);
        setLogs([]);

        // Construct arguments
        const args = [
            'tools/filter_style/filter_style.py',
            '--dir', settings.imageDir,
            '--keep', settings.keep,
            '--remove', settings.remove,
            '--batch-size', settings.batchSize,
            '--model', 'openai/clip-vit-base-patch32',
            '--threads', settings.threads
        ];

        try {
            const listener = (_event: any, data: string) => {
                setLogs(prev => [...prev, data]);
            };

            window.ipcRenderer.on('tool-output', listener);
            const result = await window.ipcRenderer.invoke('run-tool', args);
            window.ipcRenderer.removeListener('tool-output', listener);

            if (result.success) {
                showToast(t('toolbox.style_filter.finished'), 'success');
            } else {
                showToast(result.error || "Failed", 'error');
            }
        } catch (err: any) {
            showToast(err.message || "Error", 'error');
        } finally {
            setIsRunning(false);
        }
    };

    const handleStop = async () => {
        await window.ipcRenderer.invoke('kill-backend');
        setIsRunning(false);
    };

    return (
        <div className="flex flex-col gap-6">
            <div className="w-full">
                <GlassCard className="p-6">
                    <div className="flex items-center justify-between mb-6">
                        <h3 className="text-lg font-semibold flex items-center gap-2">
                            <span className="w-2 h-2 rounded-full bg-primary animate-pulse" />
                            {t('toolbox.style_filter.title')}
                        </h3>
                        <div className="flex gap-2">
                            <GlassButton
                                variant="ghost"
                                size="sm"
                                onClick={() => {
                                    if (window.confirm(t('toolbox.confirm_reset'))) {
                                        setSettings(DEFAULT_SETTINGS);
                                    }
                                }}
                                className="h-8 px-2 text-xs"
                            >
                                <RefreshCcw className="w-3 h-3 mr-1" />
                                {t('toolbox.reset_defaults')}
                            </GlassButton>
                        </div>
                    </div>

                    <div className="space-y-4">
                        <div className="space-y-2">
                            <label className="text-sm font-medium text-foreground/70 ml-1">
                                {t('toolbox.tagging.image_dir')}
                            </label>
                            <div className="flex gap-2">
                                <GlassInput
                                    value={settings.imageDir}
                                    onChange={(e) => setSettings(prev => ({ ...prev, imageDir: e.target.value }))}
                                    placeholder="C:\path\to\images"
                                />
                                <GlassButton onClick={handleSelectDir}>
                                    <FolderOpen className="w-4 h-4" />
                                </GlassButton>
                                <GlassButton
                                    variant="ghost"
                                    onClick={() => window.ipcRenderer.invoke('open-folder', settings.imageDir)}
                                    disabled={!settings.imageDir}
                                >
                                    {t('toolbox.open')}
                                </GlassButton>
                            </div>
                        </div>

                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                            <div className="space-y-2">
                                <label className="text-sm font-medium text-foreground/70 ml-1 flex flex-col">
                                    {t('toolbox.style_filter.keep')}
                                    <span className="text-[10px] text-muted-foreground font-normal">{t('toolbox.style_filter.keep_hint')}</span>
                                </label>
                                <GlassInput
                                    value={settings.keep}
                                    onChange={(e) => setSettings(prev => ({ ...prev, keep: e.target.value }))}
                                />
                            </div>
                            <div className="space-y-2">
                                <label className="text-sm font-medium text-foreground/70 ml-1 flex flex-col">
                                    {t('toolbox.style_filter.remove')}
                                    <span className="text-[10px] text-muted-foreground font-normal">{t('toolbox.style_filter.remove_hint')}</span>
                                </label>
                                <GlassInput
                                    value={settings.remove}
                                    onChange={(e) => setSettings(prev => ({ ...prev, remove: e.target.value }))}
                                />
                            </div>
                        </div>

                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                            <div className="space-y-2">
                                <label className="text-sm font-medium text-foreground/70 ml-1 flex flex-col">
                                    {t('toolbox.style_filter.batch_size')}
                                    <span className="text-[10px] text-muted-foreground font-normal">{t('toolbox.style_filter.batch_hint')}</span>
                                </label>
                                <GlassInput
                                    type="number"
                                    value={settings.batchSize}
                                    onChange={(e) => setSettings(prev => ({ ...prev, batchSize: e.target.value }))}
                                />
                            </div>
                            <div className="space-y-2">
                                <label className="text-sm font-medium text-foreground/70 ml-1 flex flex-col">
                                    {t('toolbox.style_filter.threads')}
                                    <span className="text-[10px] text-muted-foreground font-normal">{t('toolbox.style_filter.threads_hint')}</span>
                                </label>
                                <GlassInput
                                    type="number"
                                    value={settings.threads}
                                    onChange={(e) => setSettings(prev => ({ ...prev, threads: e.target.value }))}
                                />
                            </div>
                        </div>

                        <div className="pt-4 flex gap-3">
                            {!isRunning ? (
                                <GlassButton
                                    className="flex-1 h-12 text-base font-semibold"
                                    onClick={handleStart}
                                >
                                    <Play className="w-5 h-5 mr-2 fill-current" />
                                    {t('common.start')}
                                </GlassButton>
                            ) : (
                                <GlassButton
                                    variant="destructive"
                                    className="flex-1 h-12 text-base font-semibold"
                                    onClick={handleStop}
                                >
                                    <Square className="w-5 h-5 mr-2 fill-current" />
                                    {t('common.stop')}
                                </GlassButton>
                            )}
                        </div>
                    </div>
                </GlassCard>
            </div>

            <div className="w-full">
                <GlassCard className="p-6 h-[400px] flex flex-col">
                    <div className="flex items-center justify-between mb-4">
                        <h4 className="text-sm font-medium flex items-center gap-2">
                            <span className="w-1.5 h-1.5 rounded-full bg-primary/50" />
                            {t('toolbox.status_logs')}
                        </h4>
                        {isRunning && (
                            <div className="flex items-center gap-2 text-[10px] text-primary animate-pulse">
                                <Loader2 className="w-3 h-3 animate-spin" />
                                {t('common.running')}
                            </div>
                        )}
                    </div>

                    <div
                        ref={scrollRef}
                        className="flex-1 bg-black/40 rounded-xl p-4 font-mono text-[11px] overflow-y-auto scrollbar-hide border border-white/5 space-y-1"
                    >
                        {logs.length > 0 ? (
                            logs.map((log, i) => (
                                <div key={i} className="text-foreground/80 break-all border-l-2 border-primary/20 pl-2">
                                    {log}
                                </div>
                            ))
                        ) : (
                            <div className="h-full flex items-center justify-center text-muted-foreground/30 italic">
                                {t('toolbox.tagging.no_logs')}
                            </div>
                        )}
                    </div>
                </GlassCard>
            </div>
        </div>
    );
}
