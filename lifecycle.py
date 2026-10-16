"""Keep the app tied to its launcher and provide an explicit exit action."""
import os
import subprocess
import threading


def terminate_tree(process_id):
    if os.name == "nt":
        subprocess.run(["taskkill.exe", "/PID", str(process_id), "/T", "/F"],
                       capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)
    elif process_id == os.getpid():
        os._exit(0)


def watch_launcher():
    parent = os.environ.get("ANIMEFLOW_PARENT_PID")
    if not parent or os.name != "nt":
        return

    def monitor():
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.OpenProcess(0x00100000, False, int(parent))
        if handle:
            kernel.WaitForSingleObject(handle, 0xFFFFFFFF)
            kernel.CloseHandle(handle)
        terminate_tree(os.getpid())

    threading.Thread(target=monitor, daemon=True).start()


def request_exit():
    timer = threading.Timer(1.0, terminate_tree, args=(os.getpid(),))
    timer.daemon = True
    timer.start()
    return "正在退出应用，后台任务将一并停止。"
