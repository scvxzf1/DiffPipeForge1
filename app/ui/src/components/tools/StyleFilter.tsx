import { useState, useEffect, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import { GlassCard } from '@/components/ui/GlassCard';
import { GlassInput } from '@/components/ui/GlassInput';
import { GlassButton } from '@/components/ui/GlassButton';
import { Play, Square, FolderOpen, RefreshCcw, Download, ExternalLink, Settings2, Terminal } from 'lucide-react';
import { cn } from '@/lib/utils';
import { useGlassToast } from '../ui/GlassToast';
import { GlassConfirmDialog } from '../ui/GlassConfirmDialog';
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
    const [imageDir, setImageDir] = useState('');
    const [settings, setSettings] = useState<FilterSettings>(DEFAULT_SETTINGS);
    const [activeTask, setActiveTask] = useState<'idle' | 'filtering' | 'downloading'>('idle');
    const [isDownloadConfirmOpen, setIsDownloadConfirmOpen] = useState(false);
    const [logs, setLogs] = useState<string[]>([]);
    const scrollRef = useRef<HTMLDivElement>(null);

    // Sync status and logs on mount
    useEffect(() => {
        const sync = async () => {
            // Load settings
            const commonSettings = await window.ipcRenderer.invoke('get-tool-settings', 'common_toolbox_settings');
            if (commonSettings.imageDir) setImageDir(commonSettings.imageDir);

            const toolSettings = await window.ipcRenderer.invoke('get-tool-settings', 'style_filter');
            if (toolSettings) {
                setSettings(prev => ({ ...prev, ...toolSettings }));
            }

            const status = await window.ipcRenderer.invoke('get-tool-status');
            if (status.isRunning) {
                if (status.scriptName === 'filter_style.py') {
                    setActiveTask('filtering');
                } else if (status.scriptName === 'download_clip.py') {
                    setActiveTask('downloading');
                }
            }

            const savedLogs = await window.ipcRenderer.invoke('get-tool-logs');
            if (savedLogs && savedLogs.length > 0) {
                setLogs(savedLogs);
            }
        };
        sync();
    }, []);

    const saveSettings = async () => {
        // Save shared settings
        await window.ipcRenderer.invoke('save-tool-settings', {
            toolId: 'common_toolbox_settings',
            settings: { imageDir }
        });

        // Save tool-specific settings
        await window.ipcRenderer.invoke('save-tool-settings', {
            toolId: 'style_filter',
            settings: settings
        });
    };

    // Auto-save on change (debounced)
    useEffect(() => {
        const timer = setTimeout(() => {
            saveSettings();
        }, 1000);
        return () => clearTimeout(timer);
    }, [imageDir, settings]);

    // Logs auto-scroll
    useEffect(() => {
        if (scrollRef.current) {
            scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
        }
    }, [logs]);

    const handleSelectDir = async () => {
        const result = await window.ipcRenderer.invoke('dialog:openFile', {
            properties: ['openDirectory']
        });
        if (!result.canceled && result.filePaths.length > 0) {
            setImageDir(result.filePaths[0]);
        }
    };

    const handleStart = async () => {
        if (!imageDir) {
            showToast(t('toolbox.errors.no_dir'), 'error');
            return;
        }

        // Check if model exists
        const modelExists = await window.ipcRenderer.invoke('check-style-model');
        if (!modelExists) {
            showToast(t('toolbox.style_filter.no_model'), 'error');
            return;
        }

        setActiveTask('filtering');
        const listener = (_event: any, data: string) => {
            setLogs(prev => [...prev, data]);
        };

        try {
            window.ipcRenderer.on('tool-output', listener);
            const result = await window.ipcRenderer.invoke('run-tool', {
                scriptName: 'filter_style/filter_style.py',
                args: [
                    '--dir', imageDir,
                    '--keep', settings.keep,
                    '--remove', settings.remove,
                    '--batch-size', settings.batchSize,
                    '--model', 'filter_style/clip-vit-base-patch32',
                    '--threads', settings.threads
                ]
            });
            window.ipcRenderer.removeListener('tool-output', listener);

            if (result.success) {
                showToast(t('toolbox.style_filter.finished'), 'success');
            } else {
                showToast(result.error || "Failed", 'error');
            }
        } catch (err: any) {
            showToast(err.message || "Error", 'error');
        } finally {
            setActiveTask('idle');
        }
    };

    const handleStop = async () => {
        await window.ipcRenderer.invoke('stop-tool');
        setActiveTask('idle');
    };

    const handleOpenDir = async () => {
        if (!imageDir) {
            showToast(t('toolbox.errors.no_dir'), 'error');
            return;
        }
        await window.ipcRenderer.invoke('open-path', imageDir);
    };

    useEffect(() => {
        const handleOutput = (_: any, data: string) => {
            setLogs(prev => [...prev.slice(-200), data]);
        };
        const handleStatus = (_: any, status: any) => {
            if (status.type === 'finished') {
                setActiveTask('idle');
                if (status.scriptName === 'filter_style.py') {
                    if (status.isSuccess) {
                        showToast(t('toolbox.style_filter.finished'), 'success');
                    } else {
                        showToast(t('toolbox.style_filter.stopped') || "Stopped", 'error');
                    }
                } else if (status.scriptName === 'download_clip.py') {
                    if (status.isSuccess) {
                        showToast(t('toolbox.style_filter.download_success'), 'success');
                    } else {
                        showToast(t('toolbox.style_filter.download_failed'), 'error');
                    }
                }
            }
        };

        const removeOutput = (window.ipcRenderer as any).on('tool-output', handleOutput);
        const removeStatus = (window.ipcRenderer as any).on('tool-status', handleStatus);

        return () => {
            removeOutput();
            removeStatus();
        };
    }, []);

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
                                value={imageDir}
                                onChange={(e) => setImageDir(e.target.value)}
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
                                activeTask !== 'idle' ? "bg-green-500 shadow-[0_0_8px_rgba(34,197,94,0.5)] animate-pulse" : "bg-blue-500/40"
                            )} />
                            <span className="text-[11px] font-medium text-muted-foreground/80 tracking-wide uppercase">
                                {activeTask === 'downloading' ? t('toolbox.style_filter.downloading') :
                                    activeTask === 'filtering' ? t('common.running') : t('common.ready')}
                            </span>
                        </div>
                        <div className="flex gap-2">
                            <GlassButton
                                onClick={handleOpenDir}
                                variant="outline"
                                className="gap-2"
                                disabled={!imageDir}
                            >
                                <ExternalLink className="w-4 h-4" />
                                {t('toolbox.open')}
                            </GlassButton>

                            {activeTask === 'downloading' ? (
                                <GlassButton
                                    variant="destructive"
                                    className="h-10 px-6 text-sm font-semibold"
                                    onClick={handleStop}
                                >
                                    <Square className="w-4 h-4 mr-2 fill-current" />
                                    {t('common.stop_download')}
                                </GlassButton>
                            ) : activeTask === 'filtering' ? (
                                <GlassButton
                                    variant="destructive"
                                    className="h-10 px-6 text-sm font-semibold"
                                    onClick={handleStop}
                                >
                                    <Square className="w-4 h-4 mr-2 fill-current" />
                                    {t('common.stop')}
                                </GlassButton>
                            ) : (
                                <>
                                    <GlassButton
                                        variant="outline"
                                        className="h-10 text-sm font-semibold"
                                        onClick={() => setIsDownloadConfirmOpen(true)}
                                        disabled={activeTask !== 'idle'}
                                    >
                                        <Download className="w-4 h-4 mr-2" />
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

            <div className="grid grid-cols-1 gap-6">
                <ImagePreviewGrid directory={imageDir} autoRefresh={true} />
                {imageDir && (
                    <ImagePreviewGrid
                        directory={`${imageDir}/style_mismatch`}
                        title={t('toolbox.style_mismatch_preview')}
                        autoRefresh={true}
                        isRestorable={true}
                    />
                )}
            </div>

            <GlassConfirmDialog
                isOpen={isDownloadConfirmOpen}
                onClose={() => setIsDownloadConfirmOpen(false)}
                onConfirm={async () => {
                    setIsDownloadConfirmOpen(false);
                    setActiveTask('downloading');
                    setLogs([]);
                    const lang = i18n.language;
                    const source = lang === 'zh' ? 'modelscope' : 'huggingface';
                    const modelId = source === 'modelscope' ? 'openai-mirror/clip-vit-base-patch32' : 'openai/clip-vit-base-patch32';

                    try {
                        const listener = (_event: any, data: string) => {
                            setLogs(prev => [...prev, data]);
                        };
                        window.ipcRenderer.on('tool-output', listener);
                        const result = await window.ipcRenderer.invoke('run-tool', {
                            scriptName: 'download_clip.py',
                            args: ['--source', source, '--model', modelId, '--output-dir', 'filter_style'],
                            online: true
                        });
                        window.ipcRenderer.removeListener('tool-output', listener);

                        if (!result.success) {
                            // Only show toast if it failed to START (status listener handles finish/stop)
                            if (result.error) showToast(result.error, 'error');
                        }
                    } catch (err: any) {
                        showToast(err.message || "Error", 'error');
                    } finally {
                        setActiveTask('idle');
                    }
                }}
                title={t('toolbox.style_filter.download_model')}
                description={t('toolbox.style_filter.download_confirm')}
                confirmText={t('common.confirm')}
                cancelText={t('common.cancel')}
            />
        </div>
    );
}
