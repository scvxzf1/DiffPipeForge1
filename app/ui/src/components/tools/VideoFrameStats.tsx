import { useState, useEffect, useRef } from 'react';
import { GlassCard } from '../ui/GlassCard';
import { GlassButton } from '../ui/GlassButton';
import { GlassInput } from '../ui/GlassInput';
import { useTranslation } from 'react-i18next';
import { cn } from '@/lib/utils';
import { FolderOpen, Play, StopCircle, Terminal, ExternalLink, BarChart3, Settings2, Hash, Video, Clock, Layers } from 'lucide-react';
import { useGlassToast } from '../ui/GlassToast';
import { ImagePreviewGrid } from './ImagePreviewGrid';

export function VideoFrameStats() {
    const { t } = useTranslation();
    const { showToast } = useGlassToast();
    const [videoDir, setVideoDir] = useState('');
    const [mode, setMode] = useState<'scan' | 'reduce'>('scan');
    const [targetFps, setTargetFps] = useState('15');
    const [isRunning, setIsRunning] = useState(false);
    const [logs, setLogs] = useState<string[]>([]);
    const [results, setResults] = useState<any>(null);
    const scrollRef = useRef<HTMLDivElement>(null);

    // Sync status and logs on mount
    useEffect(() => {
        const sync = async () => {
            // Load settings
            const commonSettings = await window.ipcRenderer.invoke('get-tool-settings', 'common_toolbox_settings');
            if (commonSettings.imageDir) setVideoDir(commonSettings.imageDir);

            const toolSettings = await window.ipcRenderer.invoke('get-tool-settings', 'video_frame_stats');
            if (toolSettings) {
                if (toolSettings.mode !== undefined) setMode(toolSettings.mode);
                if (toolSettings.targetFps !== undefined) setTargetFps(toolSettings.targetFps);
            }

            const status = await window.ipcRenderer.invoke('get-tool-status');
            if (status.scriptName === 'video_frame_processing.py') {
                setIsRunning(status.isRunning);
            } else {
                setIsRunning(false);
            }

            const savedLogs = await window.ipcRenderer.invoke('get-tool-logs');
            if (savedLogs && savedLogs.length > 0) {
                setLogs(savedLogs);
                parseResultsFromLogs(savedLogs);
            }
        };
        sync();
    }, []);

    const parseResultsFromLogs = (allLogs: string[]) => {
        const fullContent = allLogs.join('\n');
        const startMarker = '__VIDEO_STATS_JSON_START__';
        const endMarker = '__VIDEO_STATS_JSON_END__';

        const startIdx = fullContent.indexOf(startMarker);
        const endIdx = fullContent.indexOf(endMarker);

        if (startIdx !== -1 && endIdx !== -1 && endIdx > startIdx) {
            try {
                const jsonStr = fullContent.substring(startIdx + startMarker.length, endIdx).trim();
                const parsed = JSON.parse(jsonStr);
                setResults(parsed);
            } catch (e) {
                console.error('Failed to parse video stats JSON:', e);
            }
        }
    };

    const saveSettings = async () => {
        // Save shared settings (using imageDir key for compatibility)
        await window.ipcRenderer.invoke('save-tool-settings', {
            toolId: 'common_toolbox_settings',
            settings: { imageDir: videoDir }
        });

        // Save tool-specific settings
        await window.ipcRenderer.invoke('save-tool-settings', {
            toolId: 'video_frame_stats',
            settings: { mode, targetFps }
        });
    };

    // Auto-save on change (debounced)
    useEffect(() => {
        const timer = setTimeout(() => {
            saveSettings();
        }, 1000);
        return () => clearTimeout(timer);
    }, [videoDir, mode, targetFps]);

    const handleSelectDir = async () => {
        const result = await window.ipcRenderer.invoke('dialog:openFile', {
            properties: ['openDirectory']
        });
        if (!result.canceled && result.filePaths.length > 0) {
            setVideoDir(result.filePaths[0]);
        }
    };

    const runTool = async () => {
        if (!videoDir) {
            showToast(t('toolbox.errors.no_video_dir'), 'error');
            return;
        }

        setLogs([]);
        setResults(null);
        setIsRunning(true);
        showToast(t('toolbox.video.started'), 'success');

        const args = ['--path', videoDir];
        if (mode === 'reduce') {
            args.push('--reduce', '--fps', targetFps || '15');
        }

        const result = await window.ipcRenderer.invoke('run-tool', {
            scriptName: 'video_frame_processing.py',
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
        if (!videoDir) {
            showToast(t('toolbox.errors.no_video_dir'), 'error');
            return;
        }
        const result = await window.ipcRenderer.invoke('open-path', videoDir);
        if (!result.success) {
            showToast(result.error, 'error');
        }
    };

    useEffect(() => {
        const handleOutput = (_: any, data: string) => {
            setLogs(prev => {
                const newLogs = [...prev.slice(-200), data];
                if (data.includes('__VIDEO_STATS_JSON_END__') || newLogs.some(l => l.includes('__VIDEO_STATS_JSON_END__'))) {
                    parseResultsFromLogs(newLogs);
                }
                return newLogs;
            });
        };
        const handleStatus = (_: any, status: any) => {
            if (status.type === 'finished') {
                setIsRunning(false);
                if (status.scriptName === 'video_frame_processing.py') {
                    if (status.isSuccess) {
                        showToast(t('toolbox.video.finished'), 'success');
                    } else {
                        showToast(t('toolbox.video.stopped'), 'error');
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

    const formatDuration = (seconds: number) => {
        const h = Math.floor(seconds / 3600);
        const m = Math.floor((seconds % 3600) / 60);
        const s = Math.floor(seconds % 60);
        return [h, m, s].map(v => v < 10 ? '0' + v : v).filter((v, i) => v !== '00' || i > 0).join(':');
    };

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
                                value={videoDir}
                                onChange={(e) => setVideoDir(e.target.value)}
                                placeholder="C:/path/to/videos"
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
                            <div className="space-y-4">
                                <label className="text-sm font-medium flex items-center gap-2 h-5">
                                    <Video className="w-4 h-4 text-primary/80" />
                                    {t('toolbox.aspect.mode')}
                                </label>
                                <div className="flex gap-2">
                                    <button
                                        onClick={() => setMode('scan')}
                                        className={cn(
                                            "flex-1 py-2 px-4 rounded-lg text-xs font-medium transition-all border",
                                            mode === 'scan'
                                                ? "bg-primary/20 border-primary text-primary"
                                                : "bg-white/5 border-white/5 text-muted-foreground hover:bg-white/10"
                                        )}
                                    >
                                        {t('toolbox.video.scan_mode')}
                                    </button>
                                    <button
                                        onClick={() => setMode('reduce')}
                                        className={cn(
                                            "flex-1 py-2 px-4 rounded-lg text-xs font-medium transition-all border",
                                            mode === 'reduce'
                                                ? "bg-primary/20 border-primary text-primary"
                                                : "bg-white/5 border-white/5 text-muted-foreground hover:bg-white/10"
                                        )}
                                    >
                                        {t('toolbox.video.convert_mode')}
                                    </button>
                                </div>
                            </div>

                            {mode === 'reduce' && (
                                <div className="space-y-4 animate-in fade-in slide-in-from-left-2 duration-200">
                                    <label className="text-sm font-medium flex items-center gap-2 h-5">
                                        <Hash className="w-4 h-4 text-primary/80" />
                                        {t('toolbox.video.target_fps')}
                                    </label>
                                    <GlassInput
                                        type="number"
                                        value={targetFps}
                                        onChange={(e) => setTargetFps(e.target.value)}
                                        placeholder="15"
                                        className="h-10"
                                    />
                                    <p className="text-[10px] text-muted-foreground/60 flex items-center gap-1.5 ml-1">
                                        <div className="w-1 h-1 rounded-full bg-primary/30" />
                                        {t('toolbox.video.fps_hint')}
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
                                disabled={!videoDir}
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
                </div>
            </GlassCard>

            {/* 2. Visualization Analysis Card */}
            <GlassCard className="min-h-[300px] flex flex-col overflow-hidden border-primary/20 bg-primary/5">
                <div className="flex items-center justify-between px-6 py-4 border-b border-white/5 bg-white/5">
                    <div className="flex items-center gap-2 text-sm font-bold tracking-tight text-foreground">
                        <BarChart3 className="w-5 h-5 text-primary" />
                        {t('toolbox.video.title')} - {t('toolbox.video.visual_analysis')}
                    </div>
                    {results && (
                        <div className="text-[10px] font-medium bg-primary/20 text-primary px-2 py-0.5 rounded-full border border-primary/30">
                            {t('toolbox.video.video_count')}: {results.total_videos}
                        </div>
                    )}
                </div>

                <div className="p-6">
                    {!results ? (
                        <div className="min-h-[200px] flex flex-col items-center justify-center text-muted-foreground gap-4 opacity-60">
                            <Video className="w-12 h-12 stroke-[1.5]" />
                            <p className="text-sm font-medium animate-pulse">
                                {isRunning ? t('toolbox.aspect.analyzing') : t('toolbox.aspect.click_to_start')}
                            </p>
                        </div>
                    ) : (
                        <div className="space-y-8 animate-in fade-in duration-500">
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                <div className="p-4 bg-white/5 border border-white/5 rounded-xl flex items-center gap-4">
                                    <div className="w-10 h-10 rounded-full bg-primary/10 flex items-center justify-center text-primary">
                                        <Layers className="w-5 h-5" />
                                    </div>
                                    <div className="space-y-0.5">
                                        <div className="text-[10px] font-semibold text-muted-foreground uppercase opacity-70">{t('toolbox.video.total_frames')}</div>
                                        <div className="text-lg font-bold text-foreground">{results.total_frames.toLocaleString()}</div>
                                    </div>
                                </div>
                                <div className="p-4 bg-white/5 border border-white/5 rounded-xl flex items-center gap-4">
                                    <div className="w-10 h-10 rounded-full bg-blue-500/10 flex items-center justify-center text-blue-400">
                                        <Clock className="w-5 h-5" />
                                    </div>
                                    <div className="space-y-0.5">
                                        <div className="text-[10px] font-semibold text-muted-foreground uppercase opacity-70">{t('toolbox.video.total_duration')}</div>
                                        <div className="text-lg font-bold text-foreground">{formatDuration(results.total_duration)}</div>
                                    </div>
                                </div>
                            </div>

                            <div className="space-y-4">
                                <div className="text-xs font-bold text-muted-foreground uppercase tracking-widest px-1">{t('toolbox.video.frame_dist')}</div>
                                <div className="space-y-5">
                                    {results.frame_distribution.map((item: any, idx: number) => (
                                        <div key={idx} className="space-y-2 group">
                                            <div className="flex items-center justify-between text-[11px] transition-colors group-hover:text-primary">
                                                <span className="font-semibold text-foreground/80">{item.range}</span>
                                                <span className="font-mono text-muted-foreground">
                                                    {item.count} <span className="opacity-40 mx-1">|</span> {item.percentage.toFixed(1)}%
                                                </span>
                                            </div>
                                            <div className="h-1.5 w-full bg-white/5 rounded-full overflow-hidden">
                                                <div
                                                    className="h-full bg-primary/60 group-hover:bg-primary transition-all duration-1000 ease-out"
                                                    style={{ width: `${item.percentage}%` }}
                                                />
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            </div>
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
                            <div key={i} className="whitespace-pre-wrap break-all opacity-80 animate-in fade-in slide-in-from-left-1 duration-200">
                                {log}
                            </div>
                        ))
                    )}
                </div>
            </GlassCard>

            {/* Image Preview Grid */}
            <ImagePreviewGrid directory={videoDir} className="mt-6" />
        </div>
    );
}
