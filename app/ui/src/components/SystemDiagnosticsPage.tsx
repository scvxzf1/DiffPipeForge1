import { useState, useEffect } from 'react';
import { useTranslation } from 'react-i18next';
import { Shield, CheckCircle2, XCircle, Loader2, FolderOpen, FileText, HardDrive, RefreshCw, ChevronDown, ChevronRight, Copy, Check, X } from 'lucide-react';
import { GlassCard } from './ui/GlassCard';
import { GlassButton } from './ui/GlassButton';

interface FingerprintResult {
    totalFiles: number;
    totalSize: number;
    totalSizeFormatted: string;
    sha256: string;
    diffFiles?: { type: 'missing' | 'changed' | 'added'; path: string }[];
}

interface OfficialFingerprint {
    sha256: string;
    totalFiles: number;
    version: string;
    generatedAt: string;
}

export function SystemDiagnosticsPage() {
    const { t } = useTranslation();
    const [isCalculating, setIsCalculating] = useState(false);
    const [fingerprint, setFingerprint] = useState<FingerprintResult | null>(null);
    const [official, setOfficial] = useState<OfficialFingerprint | null>(null);
    const [pythonPath, setPythonPath] = useState<string>('');
    const [showDiffFiles, setShowDiffFiles] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [calculatedAt, setCalculatedAt] = useState<string | null>(null);
    const [showHashModal, setShowHashModal] = useState<{ type: 'local' | 'official', hash: string } | null>(null);
    const [copied, setCopied] = useState(false);

    const isMatch = fingerprint && official && fingerprint.sha256 === official.sha256;

    // Load cached fingerprint and official on mount
    useEffect(() => {
        const loadCached = async () => {
            try {
                // @ts-ignore
                const [cached, officialResult, pythonInfo] = await Promise.all([
                    window.ipcRenderer.invoke('get-fingerprint-cache'),
                    window.ipcRenderer.invoke('get-official-fingerprint'),
                    window.ipcRenderer.invoke('get-python-status')
                ]);

                if (cached) {
                    setFingerprint({
                        totalFiles: cached.totalFiles,
                        totalSize: cached.totalSize,
                        totalSizeFormatted: cached.totalSizeFormatted,
                        sha256: cached.sha256
                    });
                    setCalculatedAt(cached.calculatedAt);
                }
                setOfficial(officialResult);
                setPythonPath(pythonInfo?.path || '');
            } catch (e) {
                console.error('Failed to load cached fingerprint:', e);
            }
        };
        loadCached();
    }, []);

    const handleCalculate = async () => {
        setIsCalculating(true);
        setError(null);
        setFingerprint(null);

        try {
            // @ts-ignore
            const [fpResult, officialResult, pythonInfo] = await Promise.all([
                window.ipcRenderer.invoke('calculate-python-fingerprint'),
                window.ipcRenderer.invoke('get-official-fingerprint'),
                window.ipcRenderer.invoke('get-python-status')
            ]);

            if (fpResult.error) {
                setError(fpResult.error);
            } else {
                setFingerprint(fpResult);
                setCalculatedAt(new Date().toISOString());

                // Save to cache
                // @ts-ignore
                await window.ipcRenderer.invoke('save-fingerprint-cache', {
                    sha256: fpResult.sha256,
                    totalFiles: fpResult.totalFiles,
                    totalSize: fpResult.totalSize,
                    totalSizeFormatted: fpResult.totalSizeFormatted
                });
            }

            setOfficial(officialResult);
            setPythonPath(pythonInfo?.path || '');
        } catch (e: any) {
            setError(e.message || 'Unknown error');
        } finally {
            setIsCalculating(false);
        }
    };

    const handleCopy = (text: string) => {
        navigator.clipboard.writeText(text);
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
    };

    return (
        <div className="space-y-6">
            {/* Header */}
            <div className="mb-6">
                <h2 className="text-2xl font-bold tracking-tight mb-1">{t('diagnostics.title') || '系统诊断'}</h2>
                <p className="text-muted-foreground text-sm">{t('diagnostics.description') || '检查 Python 环境完整性和系统状态'}</p>
            </div>

            {/* Environment Status Card */}
            <GlassCard className="p-6">
                <div className="flex items-center justify-between mb-6">
                    <div className="flex items-center gap-3">
                        <div className="w-12 h-12 rounded-2xl bg-gradient-to-br from-blue-500/20 to-purple-500/20 flex items-center justify-center">
                            <Shield className="w-6 h-6 text-blue-500" />
                        </div>
                        <div>
                            <h3 className="text-lg font-bold">{t('diagnostics.env_status') || '环境状态'}</h3>
                            <p className="text-sm text-muted-foreground">{t('diagnostics.python_path') || 'Python 路径'}</p>
                        </div>
                    </div>
                    <GlassButton
                        onClick={handleCalculate}
                        disabled={isCalculating}
                        className="gap-2"
                    >
                        {isCalculating ? (
                            <>
                                <Loader2 className="w-4 h-4 animate-spin" />
                                {t('diagnostics.calculating') || '计算中...'}
                            </>
                        ) : (
                            <>
                                <RefreshCw className="w-4 h-4" />
                                {t('diagnostics.calculate') || '计算指纹'}
                            </>
                        )}
                    </GlassButton>
                </div>

                {/* Python Path */}
                {pythonPath && (
                    <div className="flex items-center gap-2 px-4 py-3 bg-black/5 dark:bg-white/5 rounded-xl mb-4">
                        <FolderOpen className="w-4 h-4 text-muted-foreground" />
                        <code className="text-xs text-muted-foreground truncate">{pythonPath}</code>
                    </div>
                )}

                {/* Error Display */}
                {error && (
                    <div className="flex items-center gap-2 px-4 py-3 bg-red-500/10 border border-red-500/20 rounded-xl mb-4">
                        <XCircle className="w-4 h-4 text-red-500" />
                        <span className="text-sm text-red-500">{error}</span>
                    </div>
                )}

                {/* Stats Grid */}
                {fingerprint && (
                    <>
                        <div className="grid grid-cols-3 gap-4 mb-6">
                            <div className="p-4 bg-black/5 dark:bg-white/5 rounded-xl text-center">
                                <FileText className="w-5 h-5 mx-auto mb-2 text-blue-500" />
                                <div className="h-8 flex items-center justify-center mb-1">
                                    <div className="text-2xl font-bold">{fingerprint.totalFiles.toLocaleString()}</div>
                                </div>
                                <div className="text-xs text-muted-foreground">{t('diagnostics.total_files') || '文件总数'}</div>
                            </div>
                            <div className="p-4 bg-black/5 dark:bg-white/5 rounded-xl text-center">
                                <HardDrive className="w-5 h-5 mx-auto mb-2 text-purple-500" />
                                <div className="h-8 flex items-center justify-center mb-1">
                                    <div className="text-2xl font-bold">{fingerprint.totalSizeFormatted}</div>
                                </div>
                                <div className="text-xs text-muted-foreground">{t('diagnostics.total_size') || '总大小'}</div>
                            </div>
                            <div
                                className="p-4 bg-black/5 dark:bg-white/5 rounded-xl text-center cursor-pointer hover:bg-black/10 dark:hover:bg-white/10 transition-all group"
                                onClick={() => setShowHashModal({ type: 'local', hash: fingerprint.sha256 })}
                                title={t('diagnostics.click_to_view') || "点击查看详情"}
                            >
                                <Shield className="w-5 h-5 mx-auto mb-2 text-green-500 group-hover:scale-110 transition-transform" />
                                <div className="h-8 flex items-center justify-center mb-1 overflow-hidden">
                                    <div className="text-[10px] font-mono leading-tight break-all line-clamp-2">
                                        {fingerprint.sha256}
                                    </div>
                                </div>
                                <div className="text-xs text-muted-foreground">{t('diagnostics.fingerprint') || '环境指纹'}</div>
                            </div>
                        </div>

                        {/* Calculation Time */}
                        {calculatedAt && (
                            <div className="text-xs text-muted-foreground text-center mb-4">
                                {t('diagnostics.last_calculated', { date: new Date(calculatedAt).toLocaleString() }) || `上次计算: ${new Date(calculatedAt).toLocaleString()}`}
                            </div>
                        )}
                    </>
                )}

                {/* Verification Result */}
                {fingerprint && (
                    <div
                        className={`p-4 rounded-xl border-2 cursor-pointer transition-all hover:brightness-110 ${official
                            ? isMatch
                                ? 'bg-green-500/10 border-green-500/30'
                                : 'bg-red-500/10 border-red-500/30'
                            : 'bg-yellow-500/10 border-yellow-500/30'
                            }`}
                        onClick={() => official && setShowHashModal({
                            type: isMatch ? 'local' : (fingerprint.sha256 ? 'local' : 'official'),
                            hash: isMatch ? fingerprint.sha256 : (fingerprint.sha256 || official.sha256)
                        })}
                    >
                        <div className="flex items-center gap-3">
                            {official ? (
                                isMatch ? (
                                    <CheckCircle2 className="w-6 h-6 text-green-500" />
                                ) : (
                                    <XCircle className="w-6 h-6 text-red-500" />
                                )
                            ) : (
                                <Shield className="w-6 h-6 text-yellow-500" />
                            )}
                            <div className="flex-1">
                                <div className="font-bold">
                                    {official
                                        ? isMatch
                                            ? (t('diagnostics.match') || '✓ 完整性校验通过')
                                            : (t('diagnostics.mismatch') || '✗ 完整性校验失败')
                                        : (t('diagnostics.no_official') || '未找到官方指纹文件')
                                    }
                                </div>
                                {official && !isMatch && (
                                    <div className="text-xs text-muted-foreground mt-1">
                                        <span className="text-red-400">{t('diagnostics.local') || '本地'}: </span>
                                        <code className="font-mono">{fingerprint.sha256.substring(0, 24)}...</code>
                                        <br />
                                        <span className="text-green-400">{t('diagnostics.official') || '官方'}: </span>
                                        <code className="font-mono">{official.sha256.substring(0, 24)}...</code>
                                    </div>
                                )}
                            </div>
                        </div>

                        {/* Diff Files List */}
                        {fingerprint.diffFiles && fingerprint.diffFiles.length > 0 && (
                            <div className="mt-4 pt-4 border-t border-white/10">
                                <button
                                    onClick={() => setShowDiffFiles(!showDiffFiles)}
                                    className="flex items-center gap-2 text-sm font-medium text-muted-foreground hover:text-foreground transition-colors"
                                >
                                    {showDiffFiles ? <ChevronDown className="w-4 h-4" /> : <ChevronRight className="w-4 h-4" />}
                                    {t('diagnostics.diff_files') || '差异文件'} ({fingerprint.diffFiles.length})
                                </button>
                                {showDiffFiles && (
                                    <div className="mt-2 max-h-64 overflow-y-auto space-y-1">
                                        {fingerprint.diffFiles.slice(0, 50).map((file, idx) => (
                                            <div key={idx} className="flex items-center gap-2 text-xs font-mono py-1">
                                                <span className={`px-1.5 py-0.5 rounded text-[10px] uppercase font-bold ${file.type === 'missing' ? 'bg-red-500/20 text-red-400' :
                                                    file.type === 'changed' ? 'bg-yellow-500/20 text-yellow-400' :
                                                        'bg-blue-500/20 text-blue-400'
                                                    }`}>
                                                    {t(`diagnostics.${file.type}`) || file.type}
                                                </span>
                                                <span className="truncate text-muted-foreground">{file.path}</span>
                                            </div>
                                        ))}
                                        {fingerprint.diffFiles.length > 50 && (
                                            <div className="text-xs text-muted-foreground italic">
                                                ... {t('diagnostics.more_diff_files', { count: fingerprint.diffFiles.length - 50 }) || `还有 ${fingerprint.diffFiles.length - 50} 个差异文件`}
                                            </div>
                                        )}
                                    </div>
                                )}
                            </div>
                        )}
                    </div>
                )}
            </GlassCard>

            {/* Version Info */}
            <GlassCard className="p-6">
                <h3 className="text-lg font-bold mb-4">{t('diagnostics.version_info') || '版本信息'}</h3>
                <div className="grid grid-cols-2 gap-4">
                    <div className="p-3 bg-black/5 dark:bg-white/5 rounded-xl">
                        <div className="text-xs text-muted-foreground mb-1">{t('diagnostics.app_version') || '应用版本'}</div>
                        <div className="font-bold">v1.0.0</div>
                    </div>
                    {official && (
                        <div
                            className="p-3 bg-black/5 dark:bg-white/5 rounded-xl cursor-pointer hover:bg-black/10 dark:hover:bg-white/10 transition-all"
                            onClick={() => setShowHashModal({ type: 'official', hash: official.sha256 })}
                            title={t('diagnostics.click_to_view') || "点击查看详情"}
                        >
                            <div className="text-xs text-muted-foreground mb-1">{t('diagnostics.official_fingerprint') || '官方指纹'}</div>
                            <div className="text-xs font-mono font-bold truncate">
                                {official.sha256.substring(0, 16)}...
                            </div>
                        </div>
                    )}
                </div>
            </GlassCard>

            {/* Hash Detail Modal */}
            {showHashModal && (
                <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/40 backdrop-blur-sm animate-in fade-in duration-200">
                    <GlassCard className="w-full max-w-lg p-6 relative overflow-hidden shadow-2xl border-white/20">
                        <button
                            onClick={() => setShowHashModal(null)}
                            className="absolute top-4 right-4 p-1 rounded-full hover:bg-white/10 transition-colors text-muted-foreground hover:text-foreground"
                        >
                            <X className="w-5 h-5" />
                        </button>

                        <div className="flex items-center gap-3 mb-6">
                            <div className={`w-10 h-10 rounded-xl flex items-center justify-center ${showHashModal.type === 'local' ? 'bg-blue-500/20 text-blue-400' : 'bg-purple-500/20 text-purple-400'
                                }`}>
                                <Shield className="w-5 h-5" />
                            </div>
                            <div>
                                <h4 className="font-bold text-lg">
                                    {showHashModal.type === 'local' ? (t('diagnostics.local_hash') || '本地指纹详情') : (t('diagnostics.official_hash') || '官方指纹详情')}
                                </h4>
                                <p className="text-xs text-muted-foreground font-mono">{t('diagnostics.hash_type') || 'SHA256 Fingerprint'}</p>
                            </div>
                        </div>

                        <div className="bg-black/20 rounded-xl p-4 mb-6 border border-white/5 group relative">
                            <div className="text-sm font-mono break-all leading-relaxed pr-8">
                                {showHashModal.hash}
                            </div>
                            <button
                                onClick={() => handleCopy(showHashModal.hash)}
                                className="absolute top-4 right-4 p-1.5 rounded-lg bg-white/5 hover:bg-white/10 transition-all opacity-0 group-hover:opacity-100"
                                title="Copy to clipboard"
                            >
                                {copied ? <Check className="w-4 h-4 text-green-400" /> : <Copy className="w-4 h-4 text-muted-foreground" />}
                            </button>
                        </div>

                        <div className="flex justify-end">
                            <GlassButton onClick={() => setShowHashModal(null)} variant="default" className="px-6">
                                {t('common.close') || '关闭'}
                            </GlassButton>
                        </div>
                    </GlassCard>
                </div>
            )}
        </div>
    );
}
