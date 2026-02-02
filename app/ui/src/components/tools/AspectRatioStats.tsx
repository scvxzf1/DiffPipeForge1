import { useState, useEffect, useRef } from 'react';
import { GlassCard } from '../ui/GlassCard';
import { GlassButton } from '../ui/GlassButton';
import { GlassInput } from '../ui/GlassInput';
import { GlassSelect } from '../ui/GlassSelect';
import { useTranslation } from 'react-i18next';
import { cn } from '@/lib/utils';
import { FolderOpen, Play, StopCircle, Terminal, ExternalLink, BarChart3, Settings2, Hash, FileType } from 'lucide-react';
import { useGlassToast } from '../ui/GlassToast';
import { ImagePreviewGrid } from './ImagePreviewGrid';

export function AspectRatioStats() {
    const { t } = useTranslation();
    const { showToast } = useGlassToast();
    const [imageDir, setImageDir] = useState('');
    const [targetType, setTargetType] = useState<'image' | 'video'>('image');
    const [mode, setMode] = useState<'preset' | 'custom'>('preset');
    const [bucketSize, setBucketSize] = useState('0.1');
    const [isRunning, setIsRunning] = useState(false);
    const [logs, setLogs] = useState<string[]>([]);
    const [results, setResults] = useState<any>(null);
    const scrollRef = useRef<HTMLDivElement>(null);

    // Sync status and logs on mount
    useEffect(() => {
        const sync = async () => {
            // Load settings
            const commonSettings = await window.ipcRenderer.invoke('get-tool-settings', 'common_toolbox_settings');
            if (commonSettings.imageDir) setImageDir(commonSettings.imageDir);

            const toolSettings = await window.ipcRenderer.invoke('get-tool-settings', 'aspect_ratio_stats');
            if (toolSettings) {
                if (toolSettings.targetType !== undefined) setTargetType(toolSettings.targetType);
                if (toolSettings.mode !== undefined) setMode(toolSettings.mode);
                if (toolSettings.bucketSize !== undefined) setBucketSize(toolSettings.bucketSize);
            }

            const status = await window.ipcRenderer.invoke('get-tool-status');
            if (status.scriptName === 'aspect_ratio_statistics.py') {
                setIsRunning(status.isRunning);
            } else {
                setIsRunning(false);
            }

            const savedLogs = await window.ipcRenderer.invoke('get-tool-logs');
            if (savedLogs && savedLogs.length > 0) {
                setLogs(savedLogs);
                // Try to parse results from existing logs if any
                parseResultsFromLogs(savedLogs);
            }
        };
        sync();
    }, []);

    const parseResultsFromLogs = (allLogs: string[]) => {
        const fullContent = allLogs.join('\n');
        const startMarker = '__ASPECT_STATS_JSON_START__';
        const endMarker = '__ASPECT_STATS_JSON_END__';

        const startIdx = fullContent.indexOf(startMarker);
        const endIdx = fullContent.indexOf(endMarker);

        if (startIdx !== -1 && endIdx !== -1 && endIdx > startIdx) {
            try {
                const jsonStr = fullContent.substring(startIdx + startMarker.length, endIdx).trim();
                const parsed = JSON.parse(jsonStr);
                setResults(parsed);
            } catch (e) {
                console.error('Failed to parse stats JSON:', e);
            }
        }
    };

    const saveSettings = async () => {
        // Save shared settings
        await window.ipcRenderer.invoke('save-tool-settings', {
            toolId: 'common_toolbox_settings',
            settings: { imageDir }
        });

        // Save tool-specific settings
        await window.ipcRenderer.invoke('save-tool-settings', {
            toolId: 'aspect_ratio_stats',
            settings: { mode, bucketSize, targetType }
        });
    };

    // Auto-save on change (debounced)
    useEffect(() => {
        const timer = setTimeout(() => {
            saveSettings();
        }, 1000);
        return () => clearTimeout(timer);
    }, [imageDir, mode, bucketSize, targetType]);

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
        setResults(null);
        setIsRunning(true);
        showToast(t('toolbox.aspect.started'), 'success');

        const args = [
            '--dir', imageDir,
            '--type', targetType,
            '--mode', mode,
        ];

        if (mode === 'custom') {
            args.push('--bucket-size', bucketSize || '0.1');
        }

        const result = await window.ipcRenderer.invoke('run-tool', {
            scriptName: 'aspect_ratio_statistics.py',
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
            setLogs(prev => {
                const newLogs = [...prev.slice(-200), data];
                // Check if this chunk or the accumulated content contains the end marker
                if (data.includes('__ASPECT_STATS_JSON_END__') || newLogs.some(l => l.includes('__ASPECT_STATS_JSON_END__'))) {
                    parseResultsFromLogs(newLogs);
                }
                return newLogs;
            });
        };
        const handleStatus = (_: any, status: any) => {
            console.log('[AspectRatioStats] Received status:', status);
            if (status.type === 'finished') {
                setIsRunning(false);
                if (status.scriptName === 'aspect_ratio_statistics.py') {
                    if (status.isSuccess) {
                        showToast(t('toolbox.aspect.finished'), 'success');
                    } else {
                        showToast(t('toolbox.aspect.stopped'), 'error');
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
            {/* 1. Configuration Card */}
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
                                <Settings2 className="w-3.5 h-3.5" />
                                {t('toolbox.quality.config')}
                            </div>
                        </div>

                        <div className="p-4 grid grid-cols-1 md:grid-cols-2 gap-6">
                            <div className="space-y-2">
                                <label className="text-sm font-medium flex items-center gap-2 h-5">
                                    <FileType className="w-4 h-4 text-primary/80" />
                                    {t('toolbox.aspect.target_type')}
                                </label>
                                <GlassSelect
                                    value={targetType}
                                    onChange={(e) => setTargetType(e.target.value as any)}
                                    options={[
                                        { label: t('toolbox.aspect.type_image'), value: 'image' },
                                        { label: t('toolbox.aspect.type_video'), value: 'video' },
                                    ]}
                                />
                            </div>
                            <div className="space-y-2">
                                <label className="text-sm font-medium flex items-center gap-2 h-5">
                                    <BarChart3 className="w-4 h-4 text-primary/80" />
                                    {t('toolbox.aspect.mode')}
                                </label>
                                <GlassSelect
                                    value={mode}
                                    onChange={(e) => setMode(e.target.value as any)}
                                    options={[
                                        { label: t('toolbox.aspect.mode_preset'), value: 'preset' },
                                        { label: t('toolbox.aspect.mode_custom'), value: 'custom' },
                                    ]}
                                />
                            </div>
                            {mode === 'custom' && (
                                <div className="space-y-2 animate-in fade-in slide-in-from-left-2 duration-200">
                                    <label className="text-sm font-medium flex items-center gap-2 h-5">
                                        <Hash className="w-4 h-4 text-primary/80" />
                                        {t('toolbox.aspect.bucket_size')}
                                    </label>
                                    <GlassInput
                                        type="number"
                                        step="0.01"
                                        value={bucketSize}
                                        onChange={(e) => setBucketSize(e.target.value)}
                                        className="h-10"
                                    />
                                    <p className="text-[10px] text-muted-foreground/60 flex items-center gap-1.5 ml-1">
                                        <div className="w-1 h-1 rounded-full bg-primary/30" />
                                        {t('toolbox.aspect.bucket_size_hint')}
                                    </p>
                                </div>
                            )}
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

            {/* 2. Visualization Analysis Card */}
            <GlassCard className="min-h-[300px] flex flex-col overflow-hidden border-primary/20 bg-primary/5">
                <div className="flex items-center justify-between px-6 py-4 border-b border-white/5 bg-white/5">
                    <div className="flex items-center gap-2 text-sm font-bold tracking-tight text-foreground">
                        <BarChart3 className="w-5 h-5 text-primary" />
                        {t('toolbox.aspect.title')} - {t('toolbox.aspect.visual_analysis')}
                    </div>
                    {results && (
                        <div className="text-[10px] font-medium bg-primary/20 text-primary px-2 py-0.5 rounded-full border border-primary/30">
                            {t('toolbox.aspect.total_count')}: {results.total}
                        </div>
                    )}
                </div>

                <div className="p-6">
                    {!results ? (
                        <div className="min-h-[200px] flex flex-col items-center justify-center text-muted-foreground gap-4 opacity-60">
                            <BarChart3 className="w-12 h-12 stroke-[1.5]" />
                            <p className="text-sm font-medium animate-pulse">
                                {isRunning ? t('toolbox.aspect.analyzing') : t('toolbox.aspect.click_to_start')}
                            </p>
                        </div>
                    ) : (
                        <div className="space-y-8 animate-in fade-in duration-500">
                            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                                <div className="p-3 bg-white/5 border border-white/5 rounded-xl space-y-1">
                                    <div className="text-[10px] font-semibold text-muted-foreground uppercase opacity-70">{t('toolbox.aspect.avg_ratio')}</div>
                                    <div className="text-lg font-bold text-primary">{results.avg.toFixed(3)}</div>
                                </div>
                                <div className="p-3 bg-white/5 border border-white/5 rounded-xl space-y-1">
                                    <div className="text-[10px] font-semibold text-muted-foreground uppercase opacity-70">{t('toolbox.aspect.min_ratio')}</div>
                                    <div className="text-lg font-bold text-foreground/80">{results.min.toFixed(3)}</div>
                                </div>
                                <div className="p-3 bg-white/5 border border-white/5 rounded-xl space-y-1">
                                    <div className="text-[10px] font-semibold text-muted-foreground uppercase opacity-70">{t('toolbox.aspect.max_ratio')}</div>
                                    <div className="text-lg font-bold text-foreground/80">{results.max.toFixed(3)}</div>
                                </div>
                            </div>

                            <div className="space-y-6">
                                {results.distribution.map((item: any, idx: number) => (
                                    <div key={idx} className="space-y-2 group">
                                        <div className="flex items-center justify-between text-xs transition-colors group-hover:text-primary">
                                            <span className="font-semibold text-foreground/90">{item.bucket}</span>
                                            <span className="font-mono text-muted-foreground">
                                                {item.count} <span className="opacity-40 mx-1">|</span> {item.percentage.toFixed(1)}%
                                            </span>
                                        </div>
                                        <div className="h-2 w-full bg-white/5 rounded-full overflow-hidden border border-white/5">
                                            <div
                                                className="h-full bg-gradient-to-r from-primary/60 to-primary shadow-[0_0_10px_rgba(var(--primary-rgb),0.3)] transition-all duration-1000 ease-out rounded-full"
                                                style={{ width: `${item.percentage}%` }}
                                            />
                                        </div>
                                    </div>
                                ))}
                            </div>

                            {results.errors > 0 && (
                                <div className="p-3 bg-red-500/5 border border-red-500/10 rounded-xl flex items-center gap-3 text-red-400/80">
                                    <div className="w-1.5 h-1.5 rounded-full bg-red-500 flex-shrink-0" />
                                    <span className="text-[10px] font-medium">{t('toolbox.aspect.error_warning', { count: results.errors })}</span>
                                </div>
                            )}
                        </div>
                    )}
                </div>
            </GlassCard>

            {/* 3. Status Logs Card */}
            <GlassCard className="bg-black/40 border-primary/10 overflow-hidden">
                <div className="flex items-center justify-between px-4 py-3 border-b border-white/5 bg-white/5">
                    <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-primary">
                        <Terminal className="w-3.5 h-3.5" />
                        {t('toolbox.status_logs')}
                    </div>
                </div>
                <div
                    ref={scrollRef}
                    className="h-[250px] overflow-y-auto p-4 font-mono text-[11px] leading-relaxed space-y-0.5"
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
