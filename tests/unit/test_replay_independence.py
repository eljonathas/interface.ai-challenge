import subprocess
import sys


def test_replay_path_never_imports_a_model_sdk() -> None:
    script = (
        "import sys, builtins\n"
        "real_import = builtins.__import__\n"
        "def guarded(name, *args, **kwargs):\n"
        "    if name == 'openai' or name.startswith('openai.'):\n"
        "        raise ImportError('model SDK imported on the replay path')\n"
        "    return real_import(name, *args, **kwargs)\n"
        "builtins.__import__ = guarded\n"
        "import interface_ai_challenge.bootstrap\n"
        "import interface_ai_challenge.cli\n"
        "import interface_ai_challenge.application.replay.engine\n"
        "assert 'openai' not in sys.modules\n"
    )
    completed = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr
