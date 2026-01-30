import i18n from 'i18next';
import { initReactI18next } from 'react-i18next';

import en from './locales/en.json';
import zh from './locales/zh.json';

const saveLanguageToSettings = (lang: string) => {
  if (window.ipcRenderer) {
    window.ipcRenderer.invoke('set-language', lang);
  }
};

i18n
    .use(initReactI18next)
    .init({
        resources: {
            en: { translation: en },
            zh: { translation: zh },
        },
        lng: 'zh', 
        fallbackLng: 'en',
        interpolation: {
            escapeValue: false, // React already escapes values
        },
    });

i18n.on('languageChanged', (lng) => {
    saveLanguageToSettings(lng);
});

export { saveLanguageToSettings };
export default i18n;
