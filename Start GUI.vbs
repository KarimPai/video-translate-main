' Silent Windows launcher. Start GUI.bat remains available for setup/diagnostics.
Option Explicit
Dim shell, fso, root, python, quoted, checkCommand, result, env
Set shell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")
root = fso.GetParentFolderName(WScript.ScriptFullName)
python = fso.BuildPath(root, ".venv\Scripts\pythonw.exe")
If Not fso.FileExists(python) Then
    MsgBox "Python environment is missing. Run Setup.bat, then Start GUI.bat once.", vbExclamation, "Video Translate"
    WScript.Quit 1
End If

shell.CurrentDirectory = root
Set env = shell.Environment("PROCESS")
env("PYTHONPATH") = fso.BuildPath(root, "src")
quoted = Chr(34) & python & Chr(34)
checkCommand = quoted & " -c " & Chr(34) & "import PySide6" & Chr(34)
result = shell.Run(checkCommand, 0, True)
If result <> 0 Then
    MsgBox "PySide6 is missing. Run Start GUI.bat to install it, then use Start GUI.vbs.", vbExclamation, "Video Translate"
    WScript.Quit result
End If

shell.Run quoted & " -m auto_translate.gui", 0, False
