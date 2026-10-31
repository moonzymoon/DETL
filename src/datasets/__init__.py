from . import metropt3  # noqa: F401
# tep 加载器在 TE 数据就位后启用
try:
    from . import tep  # noqa: F401
except ImportError:
    pass
