import { useState, useEffect, useRef } from 'react';
import { GlassCard } from '../ui/GlassCard';
import { GlassButton } from '../ui/GlassButton';
import { GlassInput } from '../ui/GlassInput';
import { useTranslation } from 'react-i18next';
import { cn } from '@/lib/utils';
import { FolderOpen, Play, StopCircle, Terminal, ExternalLink, RefreshCw, Cpu, Trash2, FileType } from 'lucide-react';
import { GlassSelect } from '../ui/GlassSelect';
import { useGlassToast } from '../ui/GlassToast';
import { ImagePreviewGrid } from './ImagePreviewGrid';

export function ImageConverter() {
    const { t } = useTranslation();
    const { showToast } = useGlassToast();
    const [imageDir, setImageDir] = useState('');
    const [threads, setThreads] = useState('8');
    const [deleteSource, setDeleteSource] = useState(true);
    const [targetFormat, setTargetFormat] = useState('png');
    const [isRunning, setIsRunning] = useState(false);
    const [logs, setLogs] = useState<string[]>([]);
    const scrollRef = useRef<HTMLDivElement>(null);

    // Sync status and logs on mount
    useEffect(() => {
        const sync = async () => {
            const commonSettings = await window.ipcRenderer.invoke('get-tool-settings', 'common_toolbox_settings');
            if (commonSettings.imageDir) setImageDir(commonSettings.imageDir);

            const toolSettings = await window.ipcRenderer.invoke('get-tool-settings', 'image_converter');
            if (toolSettings) {
                if (toolSettings.threads) setThreads(toolSettings.threads);
                if (toolSettings.deleteSource !== undefined) setDeleteSource(toolSettings.deleteSource);
                if (toolSettings.targetFormat) setTargetFormat(toolSettings.targetFormat);
            }

            const status = await window.ipcRenderer.invoke('get-tool-status');
            if (status.scriptName === 'any_to_png_muilt.py') {
                setIsRunning(status.isRunning);
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
            toolId: 'image_converter',
            settings: { threads, deleteSource, targetFormat }
        });
    };

    // Auto-save on change (debounced)
    useEffect(() => {
        const timer = setTimeout(() => {
            saveSettings();
        }, 1000);
        return () => clearTimeout(timer);
    }, [imageDir, threads, deleteSource, targetFormat]);

    const handleSelectDir = async () => {
        const result = await window.ipcRenderer.invoke('dialog:openFile', {
            properties: ['openDirectory']
        });
        if (!result.canceled && result.filePaths.length > 0) {
            setImageDir(result.filePaths[0]);
        }
    };

    const runTool = async () => {
        if (!imageDir) {
            showToast(t('toolbox.errors.no_dir'), 'error');
            return;
        }

        setLogs([]);
        setIsRunning(true);
        showToast(t('toolbox.convert.started'), 'success');

        const args = [
            '--dir', imageDir,
            '--format', targetFormat,
            '--threads', threads,
        ];

        if (deleteSource) {
            args.push('--delete');
        }

        const result = await window.ipcRenderer.invoke('run-tool', {
            scriptName: 'any_to_png_muilt.py',
            args
        });

        if (!result.success) {
            showToast(result.error || 'Failed to start', 'error');
            setIsRunning(false);
        }
    };

    const stopTool = async () => {
        await window.ipcRenderer.invoke('stop-tool');
        setIsRunning(false);
    };

    const handleOpenDir = async () => {
        if (!imageDir) {
            showToast(t('toolbox.errors.no_dir'), 'error');
            return;
        }
        const result = await window.ipcRenderer.invoke('open-path', imageDir);
        if (!result.success) {
            showToast(result.error, 'error');
        }
    };

    useEffect(() => {
        const handleOutput = (_: any, data: string) => {
            setLogs(prev => [...prev.slice(-200), data]);
        };
        const handleStatus = (_: any, status: any) => {
            if (status.type === 'finished') {
                setIsRunning(false);
                if (status.scriptName === 'any_to_png_muilt.py') {
                    if (status.isSuccess) {
                        showToast(t('toolbox.convert.finished'), 'success');
                    } else {
                        showToast(t('toolbox.convert.stopped'), 'error');
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

    useEffect(() => {
        if (scrollRef.current) {
            scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
        }
    }, [logs]);

    return (
        <div className="space-y-6 animate-in fade-in slide-in-from-bottom-2 duration-300 pb-10">
            {/* Configuration Card */}
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
                                placeholder="C:/path/to/images"
                            />
                        </div>
                        <GlassButton onClick={handleSelectDir} variant="outline" className="mb-[1px]">
                            {t('common.browse')}
                        </GlassButton>
                    </div>

                    <div className="border border-white/5 rounded-xl overflow-hidden bg-white/5">
                        <div className="flex items-center justify-between px-4 py-2 border-b border-white/5 bg-white/5">
                            <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-muted-foreground/80">
                                <RefreshCw className="w-3.5 h-3.5" />
                                {t('toolbox.convert.config')}
                            </div>
                        </div>

                        <div className="p-4 grid grid-cols-1 md:grid-cols-2 gap-6">
                            <div className="space-y-2">
                                <label className="text-sm font-medium flex items-center gap-2 h-5">
                                    <FileType className="w-4 h-4 text-primary/80" />
                                    {t('toolbox.convert.target_format')}
                                </label>
                                <GlassSelect
                                    value={targetFormat}
                                    onChange={(e) => setTargetFormat(e.target.value)}
                                    options={[
                                        { label: 'PNG', value: 'png' },
                                        { label: 'JPG', value: 'jpg' },
                                        { label: 'WEBP', value: 'webp' },
                                    ]}
                                />
                            </div>

                            <div className="space-y-2">
                                <label className="text-sm font-medium flex items-center gap-2 h-5">
                                    <Cpu className="w-4 h-4 text-primary/80" />
                                    {t('toolbox.convert.threads')}
                                </label>
                                <GlassInput
                                    type="number"
                                    min="1"
                                    max="32"
                                    value={threads}
                                    onChange={(e) => setThreads(e.target.value)}
                                    placeholder="8"
                                />
                                <p className="text-[10px] text-muted-foreground/60">{t('toolbox.convert.threads_hint')}</p>
                            </div>

                            <div className="space-y-2">
                                <label className="text-sm font-medium flex items-center gap-2 h-5 mb-3">
                                    <Trash2 className="w-4 h-4 text-primary/80" />
                                    {t('toolbox.convert.delete_source')}
                                </label>
                                <div className="flex items-center gap-2">
                                    <button
                                        onClick={() => setDeleteSource(!deleteSource)}
                                        className={cn(
                                            "relative inline-flex h-6 w-11 items-center rounded-full transition-colors focus:outline-none focus:ring-2 focus:ring-primary focus:ring-offset-2 focus:ring-offset-background",
                                            deleteSource ? "bg-red-500" : "bg-white/10"
                                        )}
                                    >
                                        <span
                                            className={cn(
                                                "inline-block h-4 w-4 transform rounded-full bg-white transition-transform",
                                                deleteSource ? "translate-x-6" : "translate-x-1"
                                            )}
                                        />
                                    </button>
                                    <span className="text-xs text-muted-foreground">
                                        {deleteSource ? t('common.enabled') : t('common.disabled')}
                                    </span>
                                </div>
                                <p className="text-[10px] text-red-400/80 flex items-center gap-1">
                                    {t('toolbox.convert.delete_hint')}
                                </p>
                            </div>
                        </div>
                    </div>

                    <div className="flex items-center justify-between gap-4 pt-4 border-t border-white/5">
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
                                disabled={!imageDir}
                            >
                                <ExternalLink className="w-4 h-4" />
                                {t('toolbox.open')}
                            </GlassButton>

                            {isRunning ? (
                                <GlassButton onClick={stopTool} variant="outline" className="gap-2 text-red-400">
                                    <StopCircle className="w-4 h-4" />
                                    {t('common.stop')}
                                </GlassButton>
                            ) : (
                                <GlassButton onClick={runTool} variant="default" className="gap-2">
                                    <Play className="w-4 h-4" />
                                    {t('common.start')}
                                </GlassButton>
                            )}
                        </div>
                    </div>

                    {/* Image Preview Grid */}
                </div>
            </GlassCard>

            {/* Logs Card */}
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

            {/* Image Preview Grid */}
            <ImagePreviewGrid directory={imageDir} className="mt-6" />
        </div>
    );
}
