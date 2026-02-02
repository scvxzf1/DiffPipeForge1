import { useState, useEffect, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import { GlassCard } from '@/components/ui/GlassCard';
import { GlassInput } from '@/components/ui/GlassInput';
import { GlassButton } from '@/components/ui/GlassButton';
import { Play, Square, Loader2, FolderOpen, RefreshCcw, Download, ExternalLink, Settings2, Terminal } from 'lucide-react';
import { cn } from '@/lib/utils';
import { useGlassToast } from '../ui/GlassToast';
import { ImagePreviewGrid } from './ImagePreviewGrid';

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
    const { t, i18n } = useTranslation();
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
        await window.ipcRenderer.invoke('stop-tool');
        setIsRunning(false);
    };

    const handleOpenDir = async () => {
        if (!settings.imageDir) {
            showToast(t('toolbox.errors.no_dir'), 'error');
            return;
        }
        await window.ipcRenderer.invoke('open-path', settings.imageDir);
    };

    return (
        <div className="space-y-6">
            <GlassCard className="p-6">
                <div className="space-y-6">
                    <div className="flex items-end gap-2">
                        <div className="flex-1">
                            <label className="text-sm font-medium mb-1.5 block flex items-center gap-2">
                                <FolderOpen className="w-4 h-4" />
                                {t('toolbox.tagging.image_dir')}
                            </label>
                            <GlassInput
                                value={settings.imageDir}
                                onChange={(e) => setSettings(prev => ({ ...prev, imageDir: e.target.value }))}
                                placeholder="C:\path\to\images"
                            />
                        </div>
                        <GlassButton onClick={handleSelectDir} variant="outline" className="mb-[1px]">
                            {t('common.browse')}
                        </GlassButton>
                    </div>

                    <div className="border border-white/5 rounded-xl overflow-hidden bg-white/5">
                        <div className="flex items-center justify-between px-4 py-2 border-b border-white/5 bg-white/5">
                            <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-muted-foreground/80">
                                <Settings2 className="w-3.5 h-3.5" />
                                {t('toolbox.style_filter.title')}
                            </div>
                            <GlassButton
                                onClick={() => {
                                    if (window.confirm(t('toolbox.confirm_reset'))) {
                                        setSettings(DEFAULT_SETTINGS);
                                    }
                                }}
                                variant="outline"
                                size="sm"
                                className="h-6 px-2 gap-1 text-[10px] border-red-500/20 text-red-400/80 hover:text-red-400 hover:bg-red-400/10 hover:border-red-400/40 transition-all"
                            >
                                <RefreshCcw className="w-3" />
                                {t('toolbox.reset_defaults')}
                            </GlassButton>
                        </div>

                        <div className="p-4 space-y-4">
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                <div className="space-y-1.5">
                                    <label className="text-xs font-medium text-muted-foreground flex flex-col">
                                        {t('toolbox.style_filter.keep')}
                                        <span className="text-[10px] opacity-70 font-normal">{t('toolbox.style_filter.keep_hint')}</span>
                                    </label>
                                    <GlassInput
                                        value={settings.keep}
                                        onChange={(e) => setSettings(prev => ({ ...prev, keep: e.target.value }))}
                                    />
                                </div>
                                <div className="space-y-1.5">
                                    <label className="text-xs font-medium text-muted-foreground flex flex-col">
                                        {t('toolbox.style_filter.remove')}
                                        <span className="text-[10px] opacity-70 font-normal">{t('toolbox.style_filter.remove_hint')}</span>
                                    </label>
                                    <GlassInput
                                        value={settings.remove}
                                        onChange={(e) => setSettings(prev => ({ ...prev, remove: e.target.value }))}
                                    />
                                </div>
                            </div>

                            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-4 border-t border-white/5">
                                <div className="space-y-1.5">
                                    <label className="text-xs font-medium text-muted-foreground flex flex-col">
                                        {t('toolbox.style_filter.batch_size')}
                                        <span className="text-[10px] opacity-70 font-normal">{t('toolbox.style_filter.batch_hint')}</span>
                                    </label>
                                    <GlassInput
                                        type="number"
                                        value={settings.batchSize}
                                        onChange={(e) => setSettings(prev => ({ ...prev, batchSize: e.target.value }))}
                                    />
                                </div>
                                <div className="space-y-1.5">
                                    <label className="text-xs font-medium text-muted-foreground flex flex-col">
                                        {t('toolbox.style_filter.threads')}
                                        <span className="text-[10px] opacity-70 font-normal">{t('toolbox.style_filter.threads_hint')}</span>
                                    </label>
                                    <GlassInput
                                        type="number"
                                        value={settings.threads}
                                        onChange={(e) => setSettings(prev => ({ ...prev, threads: e.target.value }))}
                                    />
                                </div>
                            </div>
                        </div>
                    </div>

                    <div className="flex items-center justify-between gap-4 pt-4 border-t border-white/5 mt-2">
                        <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-black/20 border border-white/5 w-fit transition-all mr-auto">
                            <div className={cn(
                                "w-1.5 h-1.5 rounded-full transition-all duration-500",
                                isRunning ? "bg-green-500 shadow-[0_0_8px_rgba(34,197,94,0.5)] animate-pulse" : "bg-blue-500/40"
                            )} />
                            <span className="text-[11px] font-medium text-muted-foreground/80 tracking-wide uppercase">
                                {isRunning ? t('common.running') : t('common.ready')}
                            </span>
                        </div>
                        <div className="flex gap-2">
                            <GlassButton
                                onClick={handleOpenDir}
                                variant="outline"
                                className="gap-2"
                                disabled={!settings.imageDir}
                            >
                                <ExternalLink className="w-4 h-4" />
                                {t('toolbox.open')}
                            </GlassButton>

                            {!isRunning ? (
                                <>
                                    <GlassButton
                                        variant="outline"
                                        className="h-10 text-sm font-semibold"
                                        onClick={async () => {
                                            if (window.confirm(t('toolbox.style_filter.download_confirm'))) {
                                                setIsRunning(true);
                                                setLogs([]);
                                                const lang = i18n.language;
                                                const source = lang === 'zh' ? 'modelscope' : 'huggingface';

                                                try {
                                                    const listener = (_event: any, data: string) => {
                                                        setLogs(prev => [...prev, data]);
                                                    };
                                                    window.ipcRenderer.on('tool-output', listener);
                                                    const result = await window.ipcRenderer.invoke('run-tool', {
                                                        scriptName: 'download_clip.py',
                                                        args: ['--source', source],
                                                        online: true
                                                    });
                                                    window.ipcRenderer.removeListener('tool-output', listener);

                                                    if (result.success) {
                                                        showToast(t('common.success'), 'success');
                                                    } else {
                                                        showToast(result.error || "Download Failed", 'error');
                                                    }
                                                } catch (err: any) {
                                                    showToast(err.message || "Error", 'error');
                                                } finally {
                                                    setIsRunning(false);
                                                }
                                            }
                                        }}
                                        disabled={isRunning}
                                    >
                                        {isRunning ? (
                                            <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                                        ) : (
                                            <Download className="w-4 h-4 mr-2" />
                                        )}
                                        {t('toolbox.style_filter.download_model')}
                                    </GlassButton>
                                    <GlassButton
                                        className="h-10 px-6 text-sm font-semibold"
                                        onClick={handleStart}
                                    >
                                        <Play className="w-4 h-4 mr-2 fill-current" />
                                        {t('common.start')}
                                    </GlassButton>
                                </>
                            ) : (
                                <GlassButton
                                    variant="destructive"
                                    className="h-10 px-6 text-sm font-semibold"
                                    onClick={handleStop}
                                >
                                    <Square className="w-4 h-4 mr-2 fill-current" />
                                    {t('common.stop')}
                                </GlassButton>
                            )}
                        </div>
                    </div>
                </div>
            </GlassCard>

            <GlassCard className="bg-black/40 border-primary/10 overflow-hidden">
                <div className="flex items-center justify-between px-4 py-3 border-b border-white/5 bg-white/5">
                    <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-primary">
                        <Terminal className="w-3.5 h-3.5" />
                        {t('toolbox.status_logs')}
                    </div>
                </div>
                <div
                    ref={scrollRef}
                    className="h-[300px] overflow-y-auto p-4 font-mono text-[11px] leading-relaxed space-y-0.5"
                >
                    {logs.length === 0 ? (
                        <div className="h-full flex items-center justify-center text-muted-foreground italic">
                            {t('toolbox.tagging.no_logs')}
                        </div>
                    ) : (
                        logs.map((log, i) => (
                            <div key={i} className="whitespace-pre-wrap break-all opacity-90 animate-in fade-in slide-in-from-left-2 duration-300">
                                {log}
                            </div>
                        ))
                    )}
                </div>
            </GlassCard>

            <ImagePreviewGrid directory={settings.imageDir} className="mt-6" />
        </div>
    );
}
