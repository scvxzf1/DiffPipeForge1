import { useState, useEffect } from 'react';
import AppLayout from './components/Layout';
import { ProjectSelectionPage } from './components/ProjectSelectionPage';
import { GlassToastProvider } from './components/ui/GlassToast';
import { useTranslation } from 'react-i18next';

export default function App() {
    const [currentProject, setCurrentProject] = useState<string | null>(null);
    const { i18n } = useTranslation();

    useEffect(() => {
        document.documentElement.classList.add('dark');
        
        const savedLang = window.ipcRenderer?.invoke('get-language');
        savedLang.then((lang: string) => {
            if (lang && lang !== i18n.language) {
                i18n.changeLanguage(lang);
            }
        });
    }, [i18n]);

    const handleProjectSelect = (path: string) => {
        console.log("Selected project:", path);
        setCurrentProject(path);
    };

    const handleBackToHome = () => {
        // @ts-ignore
        window.ipcRenderer.invoke('set-session-folder', null);
        setCurrentProject(null);
    };

    return (
        <GlassToastProvider>
            {currentProject ? (
                <AppLayout
                    onBackToHome={handleBackToHome}
                    projectPath={currentProject}
                    onProjectRenamed={(newPath) => setCurrentProject(newPath)}
                />
            ) : (
                <ProjectSelectionPage onSelect={handleProjectSelect} />
            )}
        </GlassToastProvider>
    );
}
