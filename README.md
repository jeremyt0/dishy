# dishy
Leaf segmentor automator for Finlay's phd 

### Pyinstaller

Normal

```
pyinstaller main.py
```

For UPX is not available

```
pyinstaller main.py --key 123456 -n test -F -w --upx-dir C:\Users\Jeee\Documents\Projects\github\dishy\tools
```

## Problems with application after "pyinstalling"

- One reason could be certain packages

kmeans1d: Not copying lib files properly

Fix: 

1. Find kmeans1d lib in venv directory.

2. Copy directory contents to pyinstaller dist 
