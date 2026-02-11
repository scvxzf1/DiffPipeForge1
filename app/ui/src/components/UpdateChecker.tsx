import { useEffect, useState } from 'react';
import { RefreshCw, Download, RotateCcw } from 'lucide-react';
import { GlassButton } from './ui/GlassButton';
import { useGlassToast } from './ui/GlassToast';
import { cn } from '@/lib/utils';
import { useTranslation } from 'react-i18next';

export function UpdateChecker() {
    const { t } = useTranslation();
    const { showToast } = useGlassToast();
    const [status, setStatus] = useState<'idle' | 'checking' | 'available' | 'downloading' | 'downloaded' | 'error'>('idle');
    const [progress, setProgress] = useState(0);

    useEffect(() => {
        // @ts-ignore
        const removeListener = window.ipcRenderer.on('update-status', (event: any, data: any) => {
            console.log('[UpdateChecker] Status update:', data);

            if (data.status === 'checking') {
                setStatus('checking');
            } else if (data.status === 'available') {
                setStatus('available');
                showToast(t('update.available', { version: data.info.version }) || `New version available: ${data.info.version}`, 'success');
            } else if (data.status === 'not-available') {
                setStatus('idle');
                showToast(t('update.not_available') || 'You are on the latest version', 'success');
            } else if (data.status === 'error') {
                setStatus('error');
                showToast(t('update.error') || `Update error: ${data.error}`, 'error');
            } else if (data.status === 'downloading') {
                setStatus('downloading');
                setProgress(data.progress.percent);
            } else if (data.status === 'downloaded') {
                setStatus('downloaded');
                showToast(t('update.downloaded') || 'Update downloaded. Restart to install.', 'success');
            }
        });

        return () => {
            // @ts-ignore
            if (typeof removeListener === 'function') removeListener();
        };
    }, []);

    const checkForUpdates = async () => {
        setStatus('checking');
        try {
            // @ts-ignore
            const result = await window.ipcRenderer.invoke('check-for-updates');
            if (result && result.status === 'dev') {
                setStatus('idle');
                showToast("Development mode: Cannot check for updates", 'warning');
            }
        } catch (e) {
            console.error(e);
            setStatus('error'); // Ensure we exit checking state on error
        }
    };

    const quitAndInstall = async () => {
        // @ts-ignore
        await window.ipcRenderer.invoke('quit-and-install');
    };

    if (status === 'downloading') {
        return (
            <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-black/5 dark:bg-white/5 border border-primary/20">
                <Download className="w-3.5 h-3.5 text-primary animate-bounce" />
                <div className="flex flex-col w-full min-w-[80px]">
                    <div className="flex justify-between text-[10px] items-center mb-1">
                        <span className="text-primary font-bold">{t('update.downloading') || 'Downloading'}</span>
                        <span className="text-muted-foreground">{Math.round(progress)}%</span>
                    </div>
                    <div className="h-1 w-full bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                        <div
                            className="h-full bg-primary transition-all duration-300 ease-out"
                            style={{ width: `${progress}%` }}
                        />
                    </div>
                </div>
            </div>
        );
    }

    if (status === 'downloaded') {
        return (
            <GlassButton
                onClick={quitAndInstall}
                size="sm"
                className="w-full flex items-center justify-center gap-2 bg-green-500/10 hover:bg-green-500/20 text-green-600 dark:text-green-400 border-green-500/30 animate-pulse"
            >
                <RotateCcw className="w-3.5 h-3.5" />
                <span className="text-xs font-bold">{t('update.restart') || 'Restart to Update'}</span>
            </GlassButton>
        );
    }

    return (
        <GlassButton
            onClick={checkForUpdates}
            disabled={status === 'checking'}
            variant="outline"
            className={cn(
                "gap-2 px-4 border-primary/20 hover:border-primary/50 hover:bg-primary/5 text-muted-foreground hover:text-primary transition-all duration-300 backdrop-blur-sm",
                status === 'checking' && "opacity-70 cursor-wait"
            )}
            title={t('update.check_tooltip') || "Check for updates"}
        >
            <RefreshCw className={cn(
                "w-4 h-4",
                status === 'checking' && "animate-spin text-primary"
            )} />
            <span className="font-medium">{status === 'checking' ? (t('update.checking') || 'Checking...') : (t('update.check') || 'Check for Updates')}</span>
        </GlassButton>
    );
}
