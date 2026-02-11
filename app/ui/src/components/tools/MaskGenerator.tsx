
import { useState, useEffect, useRef } from 'react';
import { GlassCard } from '../ui/GlassCard';
import { GlassButton } from '../ui/GlassButton';
import { GlassInput } from '../ui/GlassInput';
import { GlassSelect } from '../ui/GlassSelect';
import { useTranslation } from 'react-i18next';
import { cn } from '@/lib/utils';
import { FolderOpen, Play, StopCircle, Terminal, Download, CheckCircle, AlertCircle, Scissors, ExternalLink, ChevronDown, ChevronUp, User, Shirt, RotateCcw } from 'lucide-react';
import { useGlassToast } from '../ui/GlassToast';
import { ImagePreviewGrid } from './ImagePreviewGrid';
import { GlassConfirmDialog } from '../ui/GlassConfirmDialog';
import { HelpIcon } from '../ui/HelpIcon';

// Body part label definitions
const BODY_LABELS = ['face', 'hair', 'left_arm', 'right_arm', 'left_leg', 'right_leg'] as const;
const CLOTHING_LABELS = ['hat', 'sunglass', 'upper_clothes', 'skirt', 'pants', 'dress', 'belt', 'shoe', 'bag', 'scarf'] as const;
const ALL_LABELS = [...BODY_LABELS, ...CLOTHING_LABELS];

// Detail method options
const DETAIL_METHODS = ['VITMatte', 'GuidedFilter', 'PyMatting', 'None'] as const;
const DEVICE_OPTIONS = [{ value: 'cuda', label: 'GPU' }, { value: 'cpu', label: 'CPU' }] as const;

export function MaskGenerator() {
    const { t, i18n } = useTranslation();
    const { showToast } = useGlassToast();
    const [imageDir, setImageDir] = useState('');
    const [maskDir, setMaskDir] = useState('');
    const [isRunning, setIsRunning] = useState(false);
    const [logs, setLogs] = useState<string[]>([]);
    const [modelExists, setModelExists] = useState<boolean | null>(null);
    const [isDownloading, setIsDownloading] = useState(false);
    const [isConfirmDialogOpen, setIsConfirmDialogOpen] = useState(false);

    // Label toggles
    const [selectedLabels, setSelectedLabels] = useState<Set<string>>(new Set());

    // Advanced settings (collapsed by default)
    const [showAdvanced, setShowAdvanced] = useState(false);
    const [detailMethod, setDetailMethod] = useState<string>('VITMatte');
    const [detailErode, setDetailErode] = useState(12);
    const [detailDilate, setDetailDilate] = useState(6);
    const [blackPoint, setBlackPoint] = useState(0.15);
    const [whitePoint, setWhitePoint] = useState(0.99);
    const [processDetail, setProcessDetail] = useState(true);
    const [device, setDevice] = useState<string>('cuda');
    const [maxMegapixels, setMaxMegapixels] = useState(2.0);
    const [outputWhiteLevel, setOutputWhiteLevel] = useState(1.0);
    const [outputBlackLevel, setOutputBlackLevel] = useState(0.0);
    const [isLoaded, setIsLoaded] = useState(false);
    const [refreshTrigger, setRefreshTrigger] = useState(0);

    const handleResetDefaults = () => {
        setDetailMethod('VITMatte');
        setDetailErode(12);
        setDetailDilate(6);
        setBlackPoint(0.15);
        setWhitePoint(0.99);
        setProcessDetail(true);
        setDevice('cuda');
        setMaxMegapixels(2.0);
        setOutputWhiteLevel(1.0);
        setOutputBlackLevel(0.0);
        showToast(t('common.config_loaded'), 'success');
    };

    const MODEL_PATH = "segformer/segformer_b2_clothes";

    const scrollRef = useRef<HTMLDivElement>(null);

    // Sync status and logs on mount
    useEffect(() => {
        // Disabled auto-check on mount to prevent issues in dev/linux
        // checkModelStatus(); 
        const sync = async () => {
            const commonSettings = await window.ipcRenderer.invoke('get-tool-settings', 'common_toolbox_settings');
            if (commonSettings.imageDir) {
                setImageDir(commonSettings.imageDir);
            }

            // Load mask generator specific settings
            const maskSettings = await window.ipcRenderer.invoke('get-tool-settings', 'mask_generator_state');
            if (maskSettings) {
                if (maskSettings.selectedLabels) setSelectedLabels(new Set(maskSettings.selectedLabels));
                if (maskSettings.showAdvanced !== undefined) setShowAdvanced(maskSettings.showAdvanced);
                if (maskSettings.detailMethod) setDetailMethod(maskSettings.detailMethod);
                if (maskSettings.detailErode !== undefined) setDetailErode(maskSettings.detailErode);
                if (maskSettings.detailDilate !== undefined) setDetailDilate(maskSettings.detailDilate);
                if (maskSettings.blackPoint !== undefined) setBlackPoint(maskSettings.blackPoint);
                if (maskSettings.whitePoint !== undefined) setWhitePoint(maskSettings.whitePoint);
                if (maskSettings.processDetail !== undefined) setProcessDetail(maskSettings.processDetail);
                if (maskSettings.device) setDevice(maskSettings.device);
                if (maskSettings.maxMegapixels !== undefined) setMaxMegapixels(maskSettings.maxMegapixels);
                if (maskSettings.outputWhiteLevel !== undefined) setOutputWhiteLevel(maskSettings.outputWhiteLevel);
                if (maskSettings.outputBlackLevel !== undefined) setOutputBlackLevel(maskSettings.outputBlackLevel);
            }
            setIsLoaded(true);

            const status = await window.ipcRenderer.invoke('get-tool-status');
            if (status.scriptName === 'mask_generate.py') {
                setIsRunning(status.isRunning);
                if (status.isRunning) {
                    const savedLogs = await window.ipcRenderer.invoke('get-tool-logs');
                    if (savedLogs && savedLogs.length > 0) {
                        setLogs(savedLogs);
                        const hasDownloadStart = savedLogs.some((l: string) => l.includes('Downloading'));
                        const hasDownloadEnd = savedLogs.some((l: string) => l.includes('DOWNLOAD_SUCCESS') || l.includes('DOWNLOAD_FAILED'));
                        if (hasDownloadStart && !hasDownloadEnd) {
                            setIsDownloading(true);
                        }
                    }
                }
            } else {
                const savedLogs = await window.ipcRenderer.invoke('get-tool-logs');
                if (savedLogs && savedLogs.length > 0) {
                    setLogs(savedLogs);
                    const hasSuccess = savedLogs.some((l: string) => l.includes('DOWNLOAD_SUCCESS'));
                    if (hasSuccess) {
                        setModelExists(true);
                    }
                }
            }
        };
        sync();
    }, []);

    const performModelCheck = async (): Promise<boolean> => {
        try {
            const result = await window.ipcRenderer.invoke('run-python-script-capture', {
                scriptPath: 'tools/mask_generate.py',
                args: ['--mode', 'check', '--model_path', MODEL_PATH]
            });

            const exists = result && result.stdout.includes('MODEL_EXISTS: True');
            setModelExists(exists);
            return exists;
        } catch (error) {
            console.error("Failed to check model status:", error);
            setModelExists(false);
            return false;
        }
    };

    // Alias for compatibility if needed, but we used performModelCheck internally now
    const checkModelStatus = performModelCheck;

    // Persistence save hook
    useEffect(() => {
        if (isLoaded) {
            window.ipcRenderer.invoke('save-tool-settings', {
                toolId: 'mask_generator_state',
                settings: {
                    selectedLabels: Array.from(selectedLabels),
                    showAdvanced,
                    detailMethod,
                    detailErode,
                    detailDilate,
                    blackPoint,
                    whitePoint,
                    processDetail,
                    device,
                    maxMegapixels,
                    outputWhiteLevel,
                    outputBlackLevel
                }
            });
        }
    }, [isLoaded, selectedLabels, showAdvanced, detailMethod, detailErode, detailDilate, blackPoint, whitePoint, processDetail, device, maxMegapixels, outputWhiteLevel, outputBlackLevel]);

    // Streaming updates hook
    useEffect(() => {
        const handleOutput = (_event: any, message: string) => {
            if (message.includes('Saved mask:')) {
                setRefreshTrigger(prev => prev + 1);
            }
        };
        window.ipcRenderer.on('tool-output', handleOutput);
        return () => {
            window.ipcRenderer.removeListener('tool-output', handleOutput);
        };
    }, []);

    const handleDownloadModel = async () => {
        setIsConfirmDialogOpen(true);
    };

    const confirmDownload = async () => {
        setIsConfirmDialogOpen(false);
        setIsDownloading(true);
        setIsRunning(true);

        const source = i18n.language.startsWith('en') ? 'huggingface' : 'modelscope';
        const sourceName = source === 'huggingface' ? 'Hugging Face' : 'ModelScope';

        setLogs(prev => [...prev, `[Download] Starting Segformer & VITMatte model download from ${sourceName}...`]);

        try {
            await window.ipcRenderer.invoke('run-tool', {
                scriptName: 'mask_generate.py',
                args: ['--mode', 'download', '--model_path', MODEL_PATH, '--source', source, '--model_type', 'all'],
                online: true
            });
        } catch (error) {
            console.error("Download failed:", error);
            setIsDownloading(false);
            setIsRunning(false);
            showToast(t('toolbox.mask.download_failed'), 'error');
        }
    };

    const handleSelectDir = async () => {
        const result = await window.ipcRenderer.invoke('dialog:openFile', {
            properties: ['openDirectory']
        });
        if (!result.canceled && result.filePaths.length > 0) {
            setImageDir(result.filePaths[0]);
            await window.ipcRenderer.invoke('save-tool-settings', {
                toolId: 'common_toolbox_settings',
                settings: { imageDir: result.filePaths[0] }
            });
        }
    };

    const handleOpenDir = async () => {
        if (!imageDir) return;
        await window.ipcRenderer.invoke('open-path', imageDir);
    };

    const handleOpenMaskDir = async () => {
        const target = maskDir || (imageDir ? `${imageDir}_masks` : '');
        if (!target) return;
        await window.ipcRenderer.invoke('open-path', target);
    };

    const handleSelectMaskDir = async () => {
        const result = await window.ipcRenderer.invoke('dialog:openFile', {
            properties: ['openDirectory']
        });
        if (!result.canceled && result.filePaths.length > 0) {
            setMaskDir(result.filePaths[0]);
        }
    };

    // Label toggle helpers
    const toggleLabel = (label: string) => {
        setSelectedLabels(prev => {
            const next = new Set(prev);
            if (next.has(label)) next.delete(label);
            else next.add(label);
            return next;
        });
    };

    const presetSelectAll = () => setSelectedLabels(new Set(ALL_LABELS));
    const presetClearAll = () => setSelectedLabels(new Set());
    const presetBody = () => setSelectedLabels(new Set(BODY_LABELS));
    const presetClothing = () => setSelectedLabels(new Set(CLOTHING_LABELS));

    const runTool = async () => {
        if (!imageDir) {
            showToast(t('toolbox.errors.no_dir'), 'error');
            return;
        }

        // On-demand model check
        if (modelExists !== true) {
            showToast(t('common.checking_model'), 'info');
            const exists = await performModelCheck();
            if (!exists) {
                // If model missing, show download dialog
                setIsConfirmDialogOpen(true);
                return;
            }
        }


        if (selectedLabels.size === 0) {
            showToast(t('toolbox.mask.no_labels'), 'error');
            return;
        }

        // Validate mask dir
        const effectiveMaskDir = maskDir || `${imageDir}_masks`;
        const normalizedImageDir = imageDir.replace(/\\/g, '/').replace(/\/$/, '').toLowerCase();
        const normalizedMaskDir = effectiveMaskDir.replace(/\\/g, '/').replace(/\/$/, '').toLowerCase();

        if (normalizedMaskDir.startsWith(normalizedImageDir + '/')) {
            showToast(t('toolbox.mask.mask_dir_inside_source'), 'error');
            return;
        }

        setLogs([]);
        setIsRunning(true);
        showToast(t('toolbox.mask.started'), 'success');

        const labelsStr = Array.from(selectedLabels).join(',');

        const args = [
            '--mode', 'run',
            '--input_dir', imageDir,
            '--model_path', MODEL_PATH,
            '--output_dir', effectiveMaskDir,
            '--labels', labelsStr,
            '--detail_method', detailMethod,
            '--detail_erode', String(detailErode),
            '--detail_dilate', String(detailDilate),
            '--black_point', String(blackPoint),
            '--white_point', String(whitePoint),
            '--process_detail', processDetail ? 'true' : 'false',
            '--device', device,
            '--max_megapixels', String(maxMegapixels),
            '--output_white_level', String(outputWhiteLevel),
            '--output_black_level', String(outputBlackLevel),
        ];

        const result = await window.ipcRenderer.invoke('run-tool', {
            scriptName: 'mask_generate.py',
            args: args,
        });

        if (!result.success) {
            showToast(result.error || 'Failed to start', 'error');
            setIsRunning(false);
        }
    };

    const stopTool = async () => {
        await window.ipcRenderer.invoke('stop-tool');
        setIsRunning(false);
        setIsDownloading(false);
        showToast(t('toolbox.mask.stopped'), 'success');
    };

    // Auto scroll logs
    useEffect(() => {
        if (scrollRef.current) {
            scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
        }
    }, [logs]);

    // Listen for logs and status
    useEffect(() => {
        const handleLog = (_event: any, log: string) => {
            setLogs(prev => [...prev, log]);

            if (log.includes('DOWNLOAD_SUCCESS')) {
                setIsDownloading(false);
                setIsRunning(false);
                setModelExists(true);
                showToast(t('toolbox.mask.download_success'), 'success');
                checkModelStatus();
            }
            if (log.includes('DOWNLOAD_FAILED')) {
                setIsDownloading(false);
                setIsRunning(false);
                showToast(t('toolbox.mask.download_failed'), 'error');
            }
            if (log.includes('Processing finished')) {
                setIsRunning(false);
                showToast(t('toolbox.mask.finished'), 'success');
            }
        };

        const handleStatus = (_event: any, status: any) => {
            setIsRunning(status.isRunning);
            if (!status.isRunning) {
                setIsDownloading(false);
            }
        };

        const removeLogListener = (window.ipcRenderer as any).on('tool-output', handleLog);
        const removeStatusListener = (window.ipcRenderer as any).on('tool-status', handleStatus);

        return () => {
            removeLogListener();
            removeStatusListener();
        };
    }, []);

    const renderLabelToggle = (label: string) => (
        <button
            key={label}
            onClick={() => toggleLabel(label)}
            disabled={isRunning}
            className={cn(
                "px-3 py-1.5 rounded-lg text-xs font-medium border transition-all duration-200 select-none",
                selectedLabels.has(label)
                    ? "bg-primary/20 text-primary border-primary/40 shadow-[0_0_8px_rgba(var(--primary-rgb),0.2)]"
                    : "bg-white/5 text-muted-foreground border-white/10 hover:bg-white/10 hover:border-white/20",
                isRunning && "opacity-50 cursor-not-allowed"
            )}
        >
            {t(`toolbox.mask.labels.${label}`)}
        </button>
    );

    return (
        <div className="space-y-6">
            <GlassCard className="p-6">
                <div className="space-y-6">
                    {/* Image Directory Selection */}
                    <div className="flex items-end gap-2">
                        <div className="flex-1">
                            <label className="text-sm font-medium mb-1.5 block flex items-center gap-2">
                                <FolderOpen className="w-4 h-4" />
                                {t('toolbox.tagging.image_dir')}
                            </label>
                            <GlassInput
                                value={imageDir}
                                onChange={(e) => setImageDir(e.target.value)}
                                placeholder={t('dataset.input_path_placeholder')}
                            />
                        </div>
                        <GlassButton onClick={handleSelectDir} variant="outline" className="mb-[1px]">
                            {t('common.browse')}
                        </GlassButton>
                    </div>

                    {/* Settings Group */}
                    <div className="border border-white/5 rounded-xl overflow-hidden bg-white/5">
                        <div className="flex items-center justify-between px-4 py-2 border-b border-white/5 bg-white/5">
                            <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-muted-foreground/80">
                                <Scissors className="w-3.5 h-3.5" />
                                {t('toolbox.mask.title')}
                            </div>

                            {/* Model Status Badge - Only show if known */}
                            {modelExists !== null && (
                                <div className={cn(
                                    "px-2 py-0.5 rounded-full text-[10px] font-medium flex items-center gap-1.5 border transition-colors",
                                    modelExists
                                        ? "bg-green-500/10 text-green-400 border-green-500/20"
                                        : "bg-red-500/10 text-red-400 border-red-500/20"
                                )}>
                                    {modelExists ? (
                                        <>
                                            <CheckCircle className="w-3 h-3" />
                                            Segformer Ready
                                        </>
                                    ) : (
                                        <>
                                            <AlertCircle className="w-3 h-3" />
                                            {t('toolbox.mask.model_missing')}
                                        </>
                                    )}
                                </div>
                            )}
                        </div>

                        <div className="p-4 space-y-4">
                            {/* Model Download Warning (if missing) */}
                            {modelExists === false && (
                                <div className="p-3 rounded-lg bg-yellow-500/10 border border-yellow-500/20 text-yellow-200/90 text-xs flex items-center justify-between gap-4">
                                    <span>{t('toolbox.mask.model_missing_hint')}</span>
                                    <GlassButton
                                        size="sm"
                                        variant={isDownloading ? "outline" : "default"}
                                        onClick={isDownloading ? stopTool : handleDownloadModel}
                                        disabled={isRunning && !isDownloading}
                                        className={cn(
                                            "h-7 px-3 text-xs transition-all shrink-0",
                                            isDownloading
                                                ? "text-red-400 border-red-400/30 hover:bg-red-400/10 hover:border-red-400/50"
                                                : "bg-yellow-500/20 hover:bg-yellow-500/30 text-yellow-200 border-yellow-500/30"
                                        )}
                                    >
                                        {isDownloading ? (
                                            <>
                                                <StopCircle className="w-3 h-3 mr-1.5 animate-pulse" />
                                                {t('common.stop')}
                                            </>
                                        ) : (
                                            <>
                                                <Download className="w-3 h-3 mr-1.5" />
                                                {t('toolbox.mask.download_model')}
                                            </>
                                        )}
                                    </GlassButton>
                                </div>
                            )}

                            {/* Body Part Selection */}
                            <div className="space-y-3">
                                {/* Preset Buttons */}
                                <div className="flex items-center gap-2">
                                    <span className="text-sm font-medium text-muted-foreground mr-1">{t('toolbox.mask.presets')}:</span>
                                    <button onClick={presetBody} disabled={isRunning}
                                        className="text-xs px-2.5 py-1 rounded-full border border-blue-400/30 text-blue-400 hover:bg-blue-400/10 transition-colors disabled:opacity-50">
                                        <User className="w-2.5 h-2.5 inline mr-1" />{t('toolbox.mask.preset_body')}
                                    </button>
                                    <button onClick={presetClothing} disabled={isRunning}
                                        className="text-xs px-2.5 py-1 rounded-full border border-purple-400/30 text-purple-400 hover:bg-purple-400/10 transition-colors disabled:opacity-50">
                                        <Shirt className="w-2.5 h-2.5 inline mr-1" />{t('toolbox.mask.preset_clothing')}
                                    </button>
                                    <button onClick={presetSelectAll} disabled={isRunning}
                                        className="text-xs px-2.5 py-1 rounded-full border border-green-400/30 text-green-400 hover:bg-green-400/10 transition-colors disabled:opacity-50">
                                        {t('toolbox.mask.preset_all')}
                                    </button>
                                    <button onClick={presetClearAll} disabled={isRunning}
                                        className="text-xs px-2.5 py-1 rounded-full border border-white/20 text-muted-foreground hover:bg-white/5 transition-colors disabled:opacity-50">
                                        {t('toolbox.mask.preset_clear')}
                                    </button>
                                </div>

                                {/* Body group */}
                                <div>
                                    <div className="text-xs text-muted-foreground/60 uppercase tracking-wider mb-1.5 flex items-center gap-1.5">
                                        <User className="w-3.5 h-3.5" /> {t('toolbox.mask.group_body')}
                                    </div>
                                    <div className="flex flex-wrap gap-1.5">
                                        {BODY_LABELS.map(renderLabelToggle)}
                                    </div>
                                </div>

                                {/* Clothing group */}
                                <div>
                                    <div className="text-xs text-muted-foreground/60 uppercase tracking-wider mb-1.5 flex items-center gap-1.5">
                                        <Shirt className="w-3.5 h-3.5" /> {t('toolbox.mask.group_clothing')}
                                    </div>
                                    <div className="flex flex-wrap gap-1.5">
                                        {CLOTHING_LABELS.map(renderLabelToggle)}
                                    </div>
                                </div>
                            </div>

                            {/* Mask Output Dir */}
                            <div className="space-y-1.5">
                                <label className="text-xs font-medium text-muted-foreground">
                                    {t('toolbox.mask.mask_output_dir')} <span className="opacity-50">({t('common.optional')})</span>
                                </label>
                                <div className="flex items-center gap-2">
                                    <GlassInput
                                        value={maskDir}
                                        onChange={(e) => setMaskDir(e.target.value)}
                                        placeholder={`${t('common.default')}: ${imageDir ? imageDir + '_masks' : 'source_folder_masks'}`}
                                        className="font-mono text-xs flex-1"
                                        disabled={isRunning}
                                    />
                                    <GlassButton onClick={handleSelectMaskDir} variant="outline" size="sm" disabled={isRunning}>
                                        {t('common.browse')}
                                    </GlassButton>
                                </div>
                            </div>

                            {/* Advanced Settings (Collapsible) */}
                            <div className="border border-white/5 rounded-lg overflow-hidden">
                                <button
                                    onClick={() => setShowAdvanced(!showAdvanced)}
                                    className="w-full flex items-center justify-between px-3 py-2 text-xs font-medium text-muted-foreground hover:bg-white/5 transition-colors"
                                >
                                    <span>{t('toolbox.mask.advanced_settings')}</span>
                                    {showAdvanced ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
                                </button>

                                {showAdvanced && (
                                    <div className="p-3 border-t border-white/5 space-y-3 bg-black/30">
                                        {/* Reset Header */}
                                        <div className="flex justify-end">
                                            <button
                                                onClick={handleResetDefaults}
                                                disabled={isRunning}
                                                className="flex items-center gap-1.5 px-2 py-1 rounded-md bg-white/5 hover:bg-white/10 text-[10px] text-muted-foreground transition-all border border-white/5 hover:border-white/10"
                                            >
                                                <RotateCcw className="w-3 h-3" />
                                                {t('toolbox.mask.reset_defaults')}
                                            </button>
                                        </div>

                                        {/* Detail Method */}
                                        <div className="grid grid-cols-2 gap-3">
                                            <div>
                                                <label className="text-xs font-medium text-muted-foreground mb-1 flex items-center gap-1.5">
                                                    {t('toolbox.mask.detail_method')}
                                                    <HelpIcon text={t('help.mask_detail_method')} />
                                                </label>
                                                <GlassSelect
                                                    value={detailMethod}
                                                    onChange={(e) => setDetailMethod(e.target.value)}
                                                    disabled={isRunning}
                                                    options={DETAIL_METHODS.map(m => ({ label: m, value: m }))}
                                                    className="h-8 text-xs"
                                                />
                                            </div>
                                            <div>
                                                <label className="text-xs font-medium text-muted-foreground mb-1 flex items-center gap-1.5">
                                                    {t('toolbox.mask.device')}
                                                    <HelpIcon text={t('help.mask_device')} />
                                                </label>
                                                <GlassSelect
                                                    value={device}
                                                    onChange={(e) => setDevice(e.target.value)}
                                                    disabled={isRunning}
                                                    options={[...DEVICE_OPTIONS]}
                                                    className="h-8 text-xs"
                                                />
                                            </div>
                                        </div>

                                        {/* Process Detail Toggle */}
                                        <div className="flex items-center gap-2">
                                            <button
                                                onClick={() => setProcessDetail(!processDetail)}
                                                disabled={isRunning}
                                                className={cn(
                                                    "w-8 h-4 rounded-full transition-all relative",
                                                    processDetail ? "bg-primary/60" : "bg-white/10"
                                                )}
                                            >
                                                <div className={cn(
                                                    "absolute top-0.5 w-3 h-3 rounded-full bg-white transition-all shadow",
                                                    processDetail ? "left-[18px]" : "left-0.5"
                                                )} />
                                            </button>
                                            <span className="text-xs text-muted-foreground">{t('toolbox.mask.process_detail')}</span>
                                        </div>

                                        {/* Erode / Dilate */}
                                        <div className="grid grid-cols-2 gap-3">
                                            <div>
                                                <label className="text-xs font-medium text-muted-foreground mb-1 flex items-center gap-1.5">
                                                    {t('toolbox.mask.detail_erode')}
                                                    <HelpIcon text={t('help.mask_detail_erode')} />
                                                </label>
                                                <input type="range" min={1} max={255} value={detailErode} onChange={(e) => setDetailErode(Number(e.target.value))} disabled={isRunning || !processDetail}
                                                    className="w-full h-1 accent-primary" />
                                                <div className="flex justify-between items-center mt-0.5">
                                                    <span className="text-[11px] text-muted-foreground">{detailErode}</span>
                                                </div>
                                            </div>
                                            <div>
                                                <label className="text-xs font-medium text-muted-foreground mb-1 flex items-center gap-1.5">
                                                    {t('toolbox.mask.detail_dilate')}
                                                    <HelpIcon text={t('help.mask_detail_dilate')} />
                                                </label>
                                                <input type="range" min={1} max={255} value={detailDilate} onChange={(e) => setDetailDilate(Number(e.target.value))} disabled={isRunning || !processDetail}
                                                    className="w-full h-1 accent-primary" />
                                                <div className="flex justify-between items-center mt-0.5">
                                                    <span className="text-[11px] text-muted-foreground">{detailDilate}</span>
                                                </div>
                                            </div>
                                        </div>

                                        {/* Black/White Points */}
                                        <div className="grid grid-cols-2 gap-3">
                                            <div>
                                                <label className="text-xs font-medium text-muted-foreground mb-1 flex items-center gap-1.5">
                                                    {t('toolbox.mask.black_point')}
                                                    <HelpIcon text={t('help.mask_black_point')} />
                                                </label>
                                                <input type="range" min={0.01} max={0.98} step={0.01} value={blackPoint}
                                                    onChange={(e) => setBlackPoint(Number(e.target.value))} disabled={isRunning || !processDetail}
                                                    className="w-full h-1 accent-primary" />
                                                <div className="flex justify-between items-center mt-0.5">
                                                    <span className="text-[11px] text-muted-foreground">{blackPoint.toFixed(2)}</span>
                                                </div>
                                            </div>
                                            <div>
                                                <label className="text-xs font-medium text-muted-foreground mb-1 flex items-center gap-1.5">
                                                    {t('toolbox.mask.white_point')}
                                                    <HelpIcon text={t('help.mask_white_point')} />
                                                </label>
                                                <input type="range" min={0.02} max={0.99} step={0.01} value={whitePoint}
                                                    onChange={(e) => setWhitePoint(Number(e.target.value))} disabled={isRunning || !processDetail}
                                                    className="w-full h-1 accent-primary" />
                                                <div className="flex justify-between items-center mt-0.5">
                                                    <span className="text-[11px] text-muted-foreground">{whitePoint.toFixed(2)}</span>
                                                </div>
                                            </div>
                                        </div>

                                        {/* Output Levels */}
                                        <div className="grid grid-cols-2 gap-3">
                                            <div>
                                                <label className="text-xs font-medium text-muted-foreground mb-1 flex items-center gap-1.5">
                                                    {t('toolbox.mask.output_black_level')}
                                                    <HelpIcon text={t('help.mask_output_black_level')} />
                                                </label>
                                                <input type="range" min={0.0} max={1.0} step={0.01} value={outputBlackLevel}
                                                    onChange={(e) => setOutputBlackLevel(Number(e.target.value))} disabled={isRunning}
                                                    className="w-full h-1 accent-primary" />
                                                <div className="flex justify-between items-center mt-0.5">
                                                    <span className="text-[11px] text-muted-foreground">{outputBlackLevel.toFixed(2)}</span>
                                                </div>
                                            </div>
                                            <div>
                                                <label className="text-xs font-medium text-muted-foreground mb-1 flex items-center gap-1.5">
                                                    {t('toolbox.mask.output_white_level')}
                                                    <HelpIcon text={t('help.mask_output_white_level')} />
                                                </label>
                                                <input type="range" min={0.0} max={1.0} step={0.01} value={outputWhiteLevel}
                                                    onChange={(e) => setOutputWhiteLevel(Number(e.target.value))} disabled={isRunning}
                                                    className="w-full h-1 accent-primary" />
                                                <div className="flex justify-between items-center mt-0.5">
                                                    <span className="text-[11px] text-muted-foreground">{outputWhiteLevel.toFixed(2)}</span>
                                                </div>
                                            </div>
                                        </div>

                                        {/* Max Megapixels (VITMatte only) */}
                                        {detailMethod.startsWith('VITMatte') && (
                                            <div>
                                                <label className="text-xs font-medium text-muted-foreground mb-1 flex items-center gap-1.5">
                                                    {t('toolbox.mask.max_megapixels')}
                                                    <HelpIcon text={t('help.mask_max_megapixels')} />
                                                </label>
                                                <div className="flex items-center gap-3">
                                                    <input type="range" min={0.5} max={8.0} step={0.1} value={maxMegapixels}
                                                        onChange={(e) => setMaxMegapixels(Number(e.target.value))} disabled={isRunning}
                                                        className="flex-1 h-1 accent-primary" />
                                                    <span className="text-[11px] text-muted-foreground w-12 text-right">{maxMegapixels.toFixed(1)} MP</span>
                                                </div>
                                            </div>
                                        )}
                                    </div>
                                )}
                            </div>
                        </div>
                    </div>

                    {/* Action Bar */}
                    <div className="flex items-center justify-between gap-4 pt-4 border-t border-white/5 mt-2">
                        <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-black/20 border border-white/5 w-fit transition-all mr-auto">
                            <div className={cn(
                                "w-1.5 h-1.5 rounded-full transition-all duration-500",
                                isRunning ? "bg-green-500 shadow-[0_0_8px_rgba(34,197,94,0.5)] animate-pulse" : "bg-blue-500/40"
                            )} />
                            <span className="text-[11px] font-medium text-muted-foreground/80 tracking-wide uppercase">
                                {isDownloading ? t('toolbox.mask.downloading') : (isRunning ? t('common.running') : t('common.ready'))}
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

                            <GlassButton
                                onClick={handleDownloadModel}
                                variant="outline"
                                className="gap-2"
                                disabled={isRunning || isDownloading}
                            >
                                <Download className="w-4 h-4" />
                                {t('toolbox.mask.download_model')}
                            </GlassButton>

                            <GlassButton
                                onClick={handleOpenMaskDir}
                                variant="outline"
                                className="gap-2"
                                disabled={!imageDir && !maskDir}
                            >
                                <FolderOpen className="w-4 h-4" />
                                {t('toolbox.mask.open_masks')}
                            </GlassButton>

                            {isRunning && !isDownloading ? (
                                <GlassButton onClick={stopTool} variant="outline" className="gap-2 text-red-400 hover:text-red-300 hover:bg-red-400/10 hover:border-red-400/30">
                                    <StopCircle className="w-4 h-4" />
                                    {t('common.stop')}
                                </GlassButton>
                            ) : (
                                <GlassButton
                                    onClick={runTool}
                                    variant="default"
                                    className="gap-2 bg-gradient-to-r from-green-500 to-emerald-600 hover:from-green-400 hover:to-emerald-500 border-0"
                                    disabled={modelExists === false || isDownloading || isRunning || selectedLabels.size === 0}
                                >
                                    <Play className="w-4 h-4 fill-current" />
                                    {t('common.start')}
                                </GlassButton>
                            )}
                        </div>
                    </div>
                </div>
            </GlassCard>

            {/* Logs */}
            <GlassCard className="bg-black/40 border-primary/10 overflow-hidden">
                <div className="flex items-center justify-between px-4 py-3 border-b border-white/5 bg-white/5">
                    <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-primary">
                        <Terminal className="w-3.5 h-3.5" />
                        {t('toolbox.status_logs')}
                    </div>
                    {logs.length > 0 && (
                        <button
                            onClick={() => setLogs([])}
                            className="text-[10px] text-muted-foreground hover:text-white transition-colors"
                        >
                            {t('common.clear')}
                        </button>
                    )}
                </div>
                <div
                    ref={scrollRef}
                    className="h-[300px] overflow-y-auto p-4 font-mono text-[11px] leading-relaxed space-y-0.5"
                >
                    {logs.length === 0 ? (
                        <div className="h-full flex items-center justify-center text-muted-foreground italic opacity-50">
                            {t('toolbox.tagging.no_logs')}
                        </div>
                    ) : (
                        logs.map((log, i) => (
                            <div key={i} className="whitespace-pre-wrap break-all opacity-90 animate-in fade-in slide-in-from-left-2 duration-300">
                                <span className={cn(
                                    log.includes('Error') || log.includes('Failed') ? 'text-red-400' :
                                        log.includes('Success') || log.includes('Finished') ? 'text-green-400' :
                                            'text-white/100'
                                )}>
                                    {log}
                                </span>
                            </div>
                        ))
                    )}
                </div>
            </GlassCard>

            {/* Preview Grid */}
            <div className="grid grid-cols-1 gap-6">
                <ImagePreviewGrid
                    directory={imageDir}
                    showMasks={true}
                    overrideMaskPath={maskDir || (imageDir ? `${imageDir}_masks` : undefined)}
                    refreshTrigger={refreshTrigger}
                />
            </div>

            <GlassConfirmDialog
                isOpen={isConfirmDialogOpen}
                onClose={() => setIsConfirmDialogOpen(false)}
                onConfirm={confirmDownload}
                title={t('toolbox.mask.download_model')}
                description={t('toolbox.mask.download_confirm')}
                confirmText={t('common.confirm')}
                cancelText={t('common.cancel')}
            />
        </div>
    );
}
