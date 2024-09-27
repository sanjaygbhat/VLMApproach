import sys
print(sys.path)

import config
print(dir(config))

try:
    from config import Config
    print("Config class imported successfully")
    print(dir(Config))
except ImportError as e:
    print(f"Error importing Config: {e}")

print("Content of config.py:")
with open('config.py', 'r') as f:
    print(f.read())