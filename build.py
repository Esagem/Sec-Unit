"""
Build script to create a PyInstaller binary for the Sec-Unit pipeline.

Usage:
    python build.py

Output:
    dist/sec-unit  (single-file executable)
"""

import PyInstaller.__main__

PyInstaller.__main__.run(
    [
        "main.py",
        "--name=sec-unit",
        "--onefile",
        "--collect-submodules=task1",
        "--collect-submodules=task2",
        "--collect-submodules=task3",
        "--hidden-import=transformers",
        "--hidden-import=torch",
        "--hidden-import=accelerate",
        "--hidden-import=pypdf",
        "--hidden-import=yaml",
        "--hidden-import=pandas",
    ]
)
