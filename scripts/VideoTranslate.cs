using System;
using System.Diagnostics;
using System.IO;
using System.Windows.Forms;

internal static class VideoTranslate
{
    private static string ProjectRoot(string start)
    {
        for (var dir = new DirectoryInfo(start); dir != null; dir = dir.Parent)
            if (File.Exists(Path.Combine(dir.FullName, "src", "auto_translate", "gui_bootstrap.py"))
                && File.Exists(Path.Combine(dir.FullName, "pyproject.toml")))
                return dir.FullName;
        return null;
    }

    [STAThread]
    private static int Main(string[] args)
    {
        bool smoke = args.Length == 1 && args[0] == "--smoke-test";
        try
        {
            var root = ProjectRoot(AppDomain.CurrentDomain.BaseDirectory);
            if (smoke && root == null) root = ProjectRoot(Environment.CurrentDirectory);
            if (root == null) throw new IOException("Place VideoTranslate.exe inside the BuzzASR project folder.");
            var python = Path.Combine(root, ".venv", "Scripts", "pythonw.exe");
            if (!File.Exists(python)) throw new IOException("Run Setup.bat in " + root + " first.");
            var launch = new ProcessStartInfo(python, "-m auto_translate.gui_bootstrap" + (smoke ? " --smoke-test" : ""));
            launch.UseShellExecute = false;
            launch.CreateNoWindow = true;
            launch.WorkingDirectory = root;
            launch.EnvironmentVariables["PYTHONPATH"] = Path.Combine(root, "src");
            launch.EnvironmentVariables["PYTHONUTF8"] = "1";
            launch.EnvironmentVariables["VIDEO_TRANSLATE_PROJECT_DIR"] = root;
            using (var child = Process.Start(launch))
            {
                child.WaitForExit();
                return child.ExitCode;
            }
        }
        catch (Exception error)
        {
            if (!smoke) MessageBox.Show(error.Message, "Video Translate", MessageBoxButtons.OK, MessageBoxIcon.Error);
            return 1;
        }
    }
}
