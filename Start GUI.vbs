' Primary Windows GUI launcher. No command prompt is shown.
' First-time Qt installation is handled here, without changing ASR models.
Option Explicit

Dim shell, fso, root, pythonw, python, quote, env
Dim command, result, failure, uvCommand

Set shell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")
root = fso.GetParentFolderName(WScript.ScriptFullName)
pythonw = fso.BuildPath(root, ".venv\Scripts\pythonw.exe")
python = fso.BuildPath(root, ".venv\Scripts\python.exe")
quote = Chr(34)

If Not fso.FileExists(pythonw) Or Not fso.FileExists(python) Then
    MsgBox "Python environment missing. Double-click Setup.bat once, then open Start GUI.vbs.", vbExclamation, "Video Translate"
    WScript.Quit 1
End If

shell.CurrentDirectory = root
Set env = shell.Environment("PROCESS")
env("PYTHONPATH") = fso.BuildPath(root, "src")
env("PYTHONUTF8") = "1"

' Verify the actual Qt widgets dependency (not just the namespace import).
command = quote & pythonw & quote & " -c " & quote & "from PySide6.QtWidgets import QApplication" & quote
On Error Resume Next
Err.Clear
result = shell.Run(command, 0, True)
failure = Err.Number
Err.Clear
On Error GoTo 0
If failure <> 0 Then
    MsgBox "Unable to check the GUI Python environment. Re-run Setup.bat.", vbExclamation, "Video Translate"
    WScript.Quit 1
End If

If result <> 0 Then
    MsgBox "First launch: PySide6 GUI components need installation. Click OK to install them using uv. No translation models will be reinstalled.", vbInformation, "Video Translate"
    uvCommand = "uv pip install --python " & quote & python & quote & " " & quote & "PySide6>=6.8,<7" & quote
    On Error Resume Next
    Err.Clear
    result = shell.Run(uvCommand, 0, True)
    failure = Err.Number
    Err.Clear
    On Error GoTo 0
    If failure <> 0 Or result <> 0 Then
        MsgBox "Could not install PySide6. Check that uv is installed and internet is available. Then run Start GUI.vbs again.", vbCritical, "Video Translate"
        WScript.Quit 1
    End If
    result = shell.Run(command, 0, True)
    If result <> 0 Then
        MsgBox "Qt installation did not complete correctly. Check the Python environment.", vbCritical, "Video Translate"
        WScript.Quit 1
    End If
End If

' The backend uses pythonw.exe; translation logs remain in the GUI Console tab.
shell.Run quote & pythonw & quote & " -m auto_translate.gui", 0, False
