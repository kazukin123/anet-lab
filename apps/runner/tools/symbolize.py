"""module+offset のクラッシュスタックを PDB で関数名 + 行番号へ解決する。

    python symbolize.py <exe> <image_base_hex> <rva_hex> [<rva_hex> ...]

exe と同じディレクトリの PDB を探す。exe / PDB は当該ビルドのものであること
(リビルドするとオフセットが無効になる)。
"""
import ctypes as C
import ctypes.wintypes as W
import os
import sys

dbghelp = C.WinDLL('dbghelp.dll')
k32 = C.WinDLL('kernel32.dll')
MAXN = 2000


class SYMBOL_INFO(C.Structure):
    _fields_ = [
        ('SizeOfStruct', W.ULONG), ('TypeIndex', W.ULONG),
        ('Reserved', C.c_ulonglong * 2), ('Index', W.ULONG), ('Size', W.ULONG),
        ('ModBase', C.c_ulonglong), ('Flags', W.ULONG), ('Value', C.c_ulonglong),
        ('Address', C.c_ulonglong), ('Register', W.ULONG), ('Scope', W.ULONG),
        ('Tag', W.ULONG), ('NameLen', W.ULONG), ('MaxNameLen', W.ULONG),
        ('Name', C.c_char * (MAXN + 1))]


class IMAGEHLP_LINE64(C.Structure):
    _fields_ = [('SizeOfStruct', W.DWORD), ('Key', C.c_void_p),
                ('LineNumber', W.DWORD), ('FileName', C.c_char_p),
                ('Address', C.c_ulonglong)]


dbghelp.SymInitialize.argtypes = [W.HANDLE, C.c_char_p, W.BOOL]
dbghelp.SymLoadModuleEx.restype = C.c_ulonglong
dbghelp.SymLoadModuleEx.argtypes = [W.HANDLE, W.HANDLE, C.c_char_p, C.c_char_p,
                                    C.c_ulonglong, W.DWORD, C.c_void_p, W.DWORD]
dbghelp.SymFromAddr.argtypes = [W.HANDLE, C.c_ulonglong,
                                C.POINTER(C.c_ulonglong), C.POINTER(SYMBOL_INFO)]
dbghelp.SymGetLineFromAddr64.argtypes = [W.HANDLE, C.c_ulonglong,
                                         C.POINTER(W.DWORD), C.POINTER(IMAGEHLP_LINE64)]

exe = os.path.abspath(sys.argv[1])
base = int(sys.argv[2], 16)
rvas = [int(a, 16) for a in sys.argv[3:]]

h = k32.GetCurrentProcess()
# DEFERRED_LOADS は付けない。付けると SymFromAddr が黙って空を返す。
dbghelp.SymSetOptions(0x2 | 0x10 | 0x80000)   # UNDNAME | LOAD_LINES | NO_PROMPTS
if not dbghelp.SymInitialize(h, os.path.dirname(exe).encode(), False):
    raise SystemExit('SymInitialize failed err=%d' % k32.GetLastError())
if dbghelp.SymLoadModuleEx(h, None, exe.encode(), None, base, 0, None, 0) == 0:
    raise SystemExit('SymLoadModuleEx failed err=%d' % k32.GetLastError())

for i, rva in enumerate(rvas):
    addr = base + rva
    sym = SYMBOL_INFO()
    sym.SizeOfStruct = 88
    sym.MaxNameLen = MAXN
    disp = C.c_ulonglong(0)
    name = '<no symbol>'
    if dbghelp.SymFromAddr(h, addr, C.byref(disp), C.byref(sym)):
        name = sym.Name.decode('utf-8', 'replace')
        if disp.value:
            name += '+0x%x' % disp.value
    loc = ''
    ln = IMAGEHLP_LINE64()
    ln.SizeOfStruct = C.sizeof(IMAGEHLP_LINE64)
    d32 = W.DWORD(0)
    if dbghelp.SymGetLineFromAddr64(h, addr, C.byref(d32), C.byref(ln)):
        loc = '\n            %s:%d' % (ln.FileName.decode('utf-8', 'replace'), ln.LineNumber)
    print('[%2d] 0x%-8x %s%s' % (i, rva, name, loc))
