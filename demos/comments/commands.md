### 1. windows下清理清理brian2的cython缓存
```
del /f /s /q "%userprofile%\.cython\*.*"
```

### 2. 启动tensorboard
windows：
```
tensorboard --logdir=logs
```
mac：
```
python3 /Library/Frameworks/Python.framework/Versions/3.5/lib/python3.5/site-packages/tensorboard/main.py 
--logdir ~/PycrmProjects/DL-NC/notebook/logs`
```

### 3. mac下打开和关闭隐藏文件
```
defaults write com.apple.finder AppleShowAllFiles -bool true
```