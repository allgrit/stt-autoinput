Set WshShell = CreateObject("WScript.Shell")
WshShell.CurrentDirectory = CreateObject("Scripting.FileSystemObject").GetParentFolderName(WScript.ScriptFullName)
WshShell.Run """C:\Program Files\Git\bin\bash.exe"" --login -c ""cd '/c/Users/allgrit/Documents/codex/STL-autoinput' && '/c/Users/allgrit/AppData/Local/Programs/Python/Python311/python.exe' -u -c \""exec(open('dictate_realtime.py', encoding='utf-8').read())\"" """, 1, False
