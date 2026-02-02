import { useState, useEffect, useRef } from 'react';
import { GlassCard } from '../ui/GlassCard';
import { GlassButton } from '../ui/GlassButton';
import { GlassInput } from '../ui/GlassInput';
import { GlassSelect } from '../ui/GlassSelect';
import { useTranslation } from 'react-i18next';
import { cn } from '@/lib/utils';
import { FolderOpen, Play, StopCircle, Terminal, Key, Type, Zap, Globe, Cpu, ExternalLink } from 'lucide-react';
import { useGlassToast } from '../ui/GlassToast';
import { TaggingPreviewGrid } from './TaggingPreviewGrid';

export function GeminiTagger() {
    const { t } = useTranslation();
    const { showToast } = useGlassToast();
    const [imageDir, setImageDir] = useState('');
    const [apiKeys, setApiKeys] = useState('');
    const [prompt, setPrompt] = useState(
        "请作为专业动漫图像打标器，基于输入的动漫风格图像，生成简明扼要的中文自然语言打标文件，需覆盖动漫风格类型、角色特征、场景元素、艺术细节、额外关键信息，所有打标文件用中文逗号分隔，确保打标文件精准贴合图像内容，不遗漏核心特征，用一段自然语言来描述，而非标签，不要分段。"
    );
    const [apiType, setApiType] = useState<'gemini' | 'openai'>('gemini');
    const [baseUrl, setBaseUrl] = useState('');
    const [modelName, setModelName] = useState('gemini-2.5-flash-preview-09-2025');
    const [concurrency, setConcurrency] = useState('40');
    const [isRunning, setIsRunning] = useState(false);
    const [logs, setLogs] = useState<string[]>([]);
    const scrollRef = useRef<HTMLDivElement>(null);

    // Load saved settings and sync status
    useEffect(() => {
        const load = async () => {
            const commonSettings = await window.ipcRenderer.invoke('get-tool-settings', 'common_toolbox_settings');
            const settings = await window.ipcRenderer.invoke('get-tool-settings', 'gemini_tagger');

            if (commonSettings.imageDir) setImageDir(commonSettings.imageDir);
            else if (settings.imageDir) setImageDir(settings.imageDir); // Fallback to local

            if (settings.apiKeys) setApiKeys(settings.apiKeys);
            if (settings.prompt) setPrompt(settings.prompt);
            if (settings.concurrency) setConcurrency(settings.concurrency);
            if (settings.apiType) setApiType(settings.apiType);
            if (settings.baseUrl) setBaseUrl(settings.baseUrl);
            if (settings.modelName) setModelName(settings.modelName);

            // Sync running status and logs
            const status = await window.ipcRenderer.invoke('get-tool-status');
            if (status.scriptName === 'gemini_concurrent_tagging.py') {
                setIsRunning(status.isRunning);
            } else {
                setIsRunning(false);
            }

            const savedLogs = await window.ipcRenderer.invoke('get-tool-logs');
            if (savedLogs && savedLogs.length > 0) {
                setLogs(savedLogs);
            }
        };
        load();
    }, []);

    const saveSettings = async () => {
        await window.ipcRenderer.invoke('save-tool-settings', {
            toolId: 'gemini_tagger',
            settings: {
                apiKeys,
                prompt,
                concurrency,
                apiType,
                baseUrl,
                modelName
            }
        });
        await window.ipcRenderer.invoke('save-tool-settings', {
            toolId: 'common_toolbox_settings',
            settings: { imageDir }
        });
    };

    // Auto-save on change (debounced)
    useEffect(() => {
        const timer = setTimeout(() => {
            if (imageDir || apiKeys || prompt) {
                saveSettings();
            }
        }, 1000);
        return () => clearTimeout(timer);
    }, [imageDir, apiKeys, prompt, concurrency, apiType, baseUrl, modelName]);

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
        if (!apiKeys) {
            showToast(t('toolbox.errors.no_keys'), 'error');
            return;
        }

        await saveSettings();
        setLogs([]);
        setIsRunning(true);

        showToast(t('toolbox.tagging.started'), 'success');

        const result = await window.ipcRenderer.invoke('run-tool', {
            scriptName: 'gemini_concurrent_tagging.py',
            args: [
                '--dir', imageDir,
                '--api_keys', apiKeys.split('\n').filter(k => k.trim()).join(','),
                '--api_type', apiType,
                '--base_url', baseUrl,
                '--model', modelName,
                '--prompt', prompt,
                '--threads', concurrency
            ]
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
            // Log status for debugging
            console.log('[GeminiTagger] Received status:', status);

            if (status.type === 'finished') {
                // Always stop running state if *any* tool finishes, 
                // as we only support one active tool at a time in the toolbox.
                setIsRunning(false);

                if (status.scriptName === 'gemini_concurrent_tagging.py') {
                    if (status.isSuccess) {
                        showToast(t('toolbox.tagging.finished'), 'success');
                    } else {
                        showToast(t('toolbox.tagging.stopped'), 'error');
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
                    {/* Dir Row */}
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

                    {/* Main Sections Row */}
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                        <div>
                            <label className="text-sm font-medium mb-1.5 block flex items-center gap-2">
                                <Key className="w-4 h-4" />
                                {t('toolbox.tagging.api_keys')}
                            </label>
                            <textarea
                                value={apiKeys}
                                onChange={(e) => setApiKeys(e.target.value)}
                                className="w-full h-40 bg-black/10 dark:bg-white/5 border border-white/10 rounded-xl p-3 text-sm focus:outline-none focus:ring-2 focus:ring-primary/50 transition-all resize-none"
                                placeholder="Key 1&#10;Key 2..."
                            />
                            <p className="text-[10px] text-muted-foreground mt-1">
                                {t('toolbox.tagging.api_keys_hint')}
                            </p>
                        </div>

                        <div>
                            <label className="text-sm font-medium mb-1.5 block flex items-center gap-2">
                                <Type className="w-4 h-4" />
                                {t('toolbox.tagging.prompt')}
                            </label>
                            <textarea
                                value={prompt}
                                onChange={(e) => setPrompt(e.target.value)}
                                className="w-full h-40 bg-black/10 dark:bg-white/5 border border-white/10 rounded-xl p-3 text-sm focus:outline-none focus:ring-2 focus:ring-primary/50 transition-all resize-none"
                            />
                        </div>
                    </div>

                    {/* API Configuration Row */}
                    <div className="grid grid-cols-1 md:grid-cols-4 gap-4 p-4 bg-white/5 rounded-xl border border-white/5 items-start">
                        <div className="space-y-1.5">
                            <label className="text-xs font-medium text-muted-foreground flex items-center gap-2">
                                <Globe className="w-3.5 h-3.5" />
                                {t('toolbox.tagging.api_type')}
                            </label>
                            <GlassSelect
                                value={apiType}
                                onChange={(e) => setApiType(e.target.value as any)}
                                options={[
                                    { label: t('toolbox.tagging.api_type_gemini'), value: 'gemini' },
                                    { label: t('toolbox.tagging.api_type_openai'), value: 'openai' },
                                ]}
                            />
                        </div>

                        {apiType === 'openai' ? (
                            <div className="space-y-1.5">
                                <label className="text-xs font-medium text-muted-foreground flex items-center gap-2">
                                    <Globe className="w-3.5 h-3.5" />
                                    {t('toolbox.tagging.base_url')}
                                </label>
                                <GlassInput
                                    value={baseUrl}
                                    onChange={(e) => setBaseUrl(e.target.value)}
                                    placeholder="https://api.openai.com/v1"
                                />
                            </div>
                        ) : (
                            <div className="hidden md:block" /> // Placeholder to maintain alignment
                        )}

                        <div className="space-y-1.5">
                            <label className="text-xs font-medium text-muted-foreground flex items-center gap-2">
                                <Cpu className="w-3.5 h-3.5" />
                                {t('toolbox.tagging.model_name')}
                            </label>
                            <GlassInput
                                value={modelName}
                                onChange={(e) => setModelName(e.target.value)}
                                placeholder={apiType === 'gemini' ? "gemini-2.5-flash-preview-09-2025" : "gpt-4o"}
                            />
                        </div>

                        <div className="space-y-1.5">
                            <label className="text-xs font-medium text-muted-foreground flex items-center gap-2">
                                <Zap className="w-3.5 h-3.5" />
                                {t('toolbox.tagging.concurrency')}
                            </label>
                            <GlassInput
                                type="number"
                                value={concurrency}
                                onChange={(e) => setConcurrency(e.target.value)}
                                placeholder="40"
                            />
                            <p className="text-[10px] text-muted-foreground opacity-70">
                                {t('toolbox.tagging.concurrency_hint')}
                            </p>
                        </div>
                    </div>

                    {/* Actions Row */}
                    <div className="flex items-center justify-between gap-4 pt-2">
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
            <TaggingPreviewGrid directory={imageDir} className="mt-6" />
        </div>
    );
}
