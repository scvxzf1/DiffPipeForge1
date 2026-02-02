import { useState, useEffect, useRef } from 'react';
import { GlassCard } from '../ui/GlassCard';
import { GlassButton } from '../ui/GlassButton';
import { GlassInput } from '../ui/GlassInput';
import { GlassSelect } from '../ui/GlassSelect';
import { useTranslation } from 'react-i18next';
import { cn } from '@/lib/utils';
import { FolderOpen, Play, StopCircle, Terminal, ExternalLink } from 'lucide-react';
import { useGlassToast } from '../ui/GlassToast';
import { ImagePreviewGrid } from './ImagePreviewGrid';

export function Deduplication() {
    const { t } = useTranslation();
    const { showToast } = useGlassToast();
    const [imageDir, setImageDir] = useState('');
    const [mode, setMode] = useState<'exact' | 'similar'>('exact');
    const [threshold, setThreshold] = useState('5');
    const [isRunning, setIsRunning] = useState(false);
    const [logs, setLogs] = useState<string[]>([]);
    const scrollRef = useRef<HTMLDivElement>(null);

    // Sync status and logs on mount
    useEffect(() => {
        const sync = async () => {
            // Load settings
            const commonSettings = await window.ipcRenderer.invoke('get-tool-settings', 'common_toolbox_settings');
            if (commonSettings.imageDir) setImageDir(commonSettings.imageDir);

            const toolSettings = await window.ipcRenderer.invoke('get-tool-settings', 'deduplication');
            if (toolSettings) {
                if (toolSettings.mode) setMode(toolSettings.mode);
                if (toolSettings.threshold) setThreshold(toolSettings.threshold);
            }

            const status = await window.ipcRenderer.invoke('get-tool-status');
            if (status.scriptName === 'image_deduplication.py') {
                setIsRunning(status.isRunning);
            } else {
                setIsRunning(false);
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
            toolId: 'deduplication',
            settings: { mode, threshold }
        });
    };

    // Auto-save on change (debounced)
    useEffect(() => {
        const timer = setTimeout(() => {
            saveSettings();
        }, 1000);
        return () => clearTimeout(timer);
    }, [imageDir, mode, threshold]);

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
        showToast(t('toolbox.deduplicate.started'), 'success');

        const args = [
            '--dir', imageDir,
            '--mode', mode
        ];

        if (mode === 'similar') {
            args.push('--threshold', threshold);
        }

        const result = await window.ipcRenderer.invoke('run-tool', {
            scriptName: 'image_deduplication.py',
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
            // Logs for debugging
            console.log('[Deduplication] Received status:', status);

            if (status.type === 'finished') {
                setIsRunning(false);

                if (status.scriptName === 'image_deduplication.py') {
                    if (status.isSuccess) {
                        showToast(t('toolbox.deduplicate.finished'), 'success');
                    } else {
                        showToast(t('toolbox.deduplicate.stopped'), 'error');
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
                                placeholder="C:/path/to/images"
                            />
                        </div>
                        <GlassButton onClick={handleSelectDir} variant="outline" className="mb-[1px]">
                            {t('common.browse')}
                        </GlassButton>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                        <div className="space-y-1.5">
                            <label className="text-sm font-medium">{t('toolbox.deduplicate.mode')}</label>
                            <GlassSelect
                                value={mode}
                                onChange={(e) => setMode(e.target.value as any)}
                                options={[
                                    { label: t('toolbox.deduplicate.exact'), value: 'exact' },
                                    { label: t('toolbox.deduplicate.similar'), value: 'similar' },
                                ]}
                            />
                        </div>

                        {mode === 'similar' && (
                            <div className="space-y-1.5 animate-in slide-in-from-top-2 duration-200">
                                <label className="text-sm font-medium">{t('toolbox.deduplicate.threshold')}</label>
                                <GlassInput
                                    type="number"
                                    value={threshold}
                                    onChange={(e) => setThreshold(e.target.value)}
                                />
                                <p className="text-[10px] text-muted-foreground opacity-70">
                                    {t('toolbox.deduplicate.threshold_hint')}
                                </p>
                            </div>
                        )}
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
