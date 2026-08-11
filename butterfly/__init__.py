from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("butterfly-ai-local")
except PackageNotFoundError:
    __version__ = "dev"
