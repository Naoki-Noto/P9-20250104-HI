# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 20:19:59 2024

@author: noyor
"""
import subprocess

# 実行したいPythonファイルのリスト
scripts = [
           'TrAdaBoostR2.py',
           'TrAdaBoostR2_top.py',
           'TrAdaBoostR2_bottom.py',
           ]

# 連続してPythonファイルを実行
for script in scripts:
    print(f"Running {script}...")
    subprocess.run(['python', script])
    print(f"{script} completed.\n")
