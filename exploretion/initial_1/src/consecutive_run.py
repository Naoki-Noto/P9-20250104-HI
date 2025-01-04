# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 20:19:59 2024

@author: noyor
"""
import subprocess

# 実行したいPythonファイルのリスト
scripts = [
           'XGBRegression_Q.py',
           'LinearRegression_standard_Q.py',
           'RFRegression_Q.py',
           ]

# 連続してPythonファイルを実行
for script in scripts:
    print(f"Running {script}...")
    subprocess.run(['python', script])
    print(f"{script} completed.\n")
