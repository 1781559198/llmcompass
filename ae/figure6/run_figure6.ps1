# 删除当前目录下的 csv 和 pdf 文件
Remove-Item -ErrorAction SilentlyContinue *.csv
Remove-Item -ErrorAction SilentlyContinue *.pdf

# 切换到上两级目录
Set-Location ..\..

# 运行 Python 模块
python -m ae.figure6.test_cost_model

# 切回原目录（可选）
Set-Location ae\figure6