' Primary silent Windows launcher for Video Translate.
' Errors display a Windows dialog and are logged under logs\gui-startup.log.
Option Explicit
Dim shell, fso, root, pythonw, python, quote, env
Dim checkCommand, installCommand, launchCommand, result, failure, detail, smoke
smoke = False
Set shell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")

If WScript.Arguments.Count > 0 Then
    If LCase(WScript.Arguments(0)) = "--launcher-self-test" Then
        WScript.Echo "GUI VBS launcher syntax OK"
        WScript.Quit 0
    End If
    If LCase(WScript.Arguments(0)) = "--smoke-test" Then smoke = True
End If

root = fso.GetParentFolderName(WScript.ScriptFullName)
pythonw = fso.BuildPath(root, ".venv\Scripts\pythonw.exe")
python = fso.BuildPath(root, ".venv\Scripts\python.exe")
quote = Chr(34)

If Not fso.FileExists(pythonw) Or Not fso.FileExists(python) Then
    MsgBox "Python environment missing in " & root & vbCrLf & "Run Setup.bat first.", vbExclamation, "Video Translate"
    WScript.Quit 1
End If

shell.CurrentDirectory = root
Set env = shell.Environment("PROCESS")
env("PYTHONPATH") = fso.BuildPath(root, "src")
env("PYTHONUTF8") = "1"
env("VIDEO_TRANSLATE_PROJECT_DIR") = root

' Qt import preflight: a working interpreter alone does not imply PySide6 is installed.
checkCommand = quote & pythonw & quote & " -c " & quote & "from PySide6.QtWidgets import QApplication" & quote
On Error Resume Next
Err.Clear
result = shell.Run(checkCommand, 0, True)
failure = Err.Number
detail = Err.Description
Err.Clear
On Error GoTo 0
If failure <> 0 Then
    MsgBox "Unable to check the GUI runtime: " & detail, vbCritical, "Video Translate"
    WScript.Quit 1
End If

If result <> 0 Then
    result = MsgBox("PySide6 is not installed or cannot import." & vbCrLf & _
       "Install GUI dependencies now? Existing CUDA and model packages will not be reinstalled.", _
       vbYesNo + vbQuestion, "Video Translate")
    If result <> vbYes Then WScript.Quit 1
    installCommand = "uv pip install --python " & quote & python & quote & _
                     " " & quote & "PySide6>=6.8,<7" & quote
    On Error Resume Next
    Err.Clear
    result = shell.Run(installCommand, 0, True)
    failure = Err.Number
    detail = Err.Description
    Err.Clear
    On Error GoTo 0
    If failure <> 0 Or result <> 0 Then
        MsgBox "PySide6 installation failed. Check uv and internet access." & vbCrLf & _
               detail, vbCritical, "Video Translate"
        WScript.Quit 1
    End If
    result = shell.Run(checkCommand, 0, True)
    If result <> 0 Then
        MsgBox "PySide6 still cannot import. Check the Python environment.", vbCritical, "Video Translate"
        WScript.Quit 1
    End If
End If

' Wait without a visible console; the GUI bootstrap reports startup exceptions.
launchCommand = quote & pythonw & quote & " -m auto_translate.gui_bootstrap"
If smoke Then launchCommand = launchCommand & " --smoke-test"
On Error Resume Next
Err.Clear
result = shell.Run(launchCommand, 0, True)
failure = Err.Number
detail = Err.Description
Err.Clear
On Error GoTo 0
If failure <> 0 Then
    MsgBox "Failed to launch the GUI: " & detail, vbCritical, "Video Translate"
    WScript.Quit 1
End If
If result <> 0 Then
    MsgBox "The application exited with code " & result & "." & vbCrLf & _
           "See " & fso.BuildPath(root, "logs\gui-startup.log"), vbExclamation, "Video Translate"
End If
WScript.Quit result
