/// <reference types="vite/client" />

declare module 'rollup-plugin-javascript-obfuscator';

declare namespace JSX {
    interface IntrinsicElements {
        'theme-button': any;
    }
}
declare global {
    interface Window {
        ipcRenderer: {
            send(channel: string, ...args: any[]): void;
            invoke(channel: string, ...args: any[]): Promise<any>;
            on(channel: string, func: (...args: any[]) => void): () => void;
            off(channel: string, func: (...args: any[]) => void): void;
            removeAllListeners(channel: string): void;
        };
    }
}
