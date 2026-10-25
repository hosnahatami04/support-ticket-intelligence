from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("ticketintel")
except PackageNotFoundError:  # running from a source tree without install metadata
    __version__ = "0.0.0"
