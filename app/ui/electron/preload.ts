import { ipcRenderer, contextBridge } from 'electron'

// --------- Expose some API to the Renderer process ---------
contextBridge.exposeInMainWorld('ipcRenderer', {
  on(channel: string, listener: (...args: any[]) => void) {
    const subscription = (_event: any, ...args: any[]) => listener(_event, ...args)
    ipcRenderer.on(channel, subscription)
    return () => {
      ipcRenderer.removeListener(channel, subscription)
    }
  },
  off(channel: string, listener: (...args: any[]) => void) {
    // Legacy support - meaningless with contextBridge but kept for potential external compatibility
    ipcRenderer.removeListener(channel, listener)
  },
  send(...args: Parameters<typeof ipcRenderer.send>) {
    const [channel, ...omit] = args
    return ipcRenderer.send(channel, ...omit)
  },
  invoke(...args: Parameters<typeof ipcRenderer.invoke>) {
    const [channel, ...omit] = args
    return ipcRenderer.invoke(channel, ...omit)
  },
  removeAllListeners(channel: string) {
    return ipcRenderer.removeAllListeners(channel)
  },

  // You can expose other APTs you need here.
  // ...
})
