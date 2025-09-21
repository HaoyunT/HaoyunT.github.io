@echo off
REM 自动提交并推送 HaoyunT.github.io 网站更新

REM 切换到本地仓库目录
cd /d F:\tex\HaoyunT.github.io

REM 添加所有更改
git add .

REM 提交更改，使用当前日期和时间作为提交信息
set datetime=%date% %time%
git commit -m "Auto update %datetime%"

REM 推送到 GitHub main 分支
git push origin main

echo 网站已更新完成！
pause
