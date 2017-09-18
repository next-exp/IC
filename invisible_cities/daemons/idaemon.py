from importlib import import_module
import traceback


def summon_daemon(daemon_name):
    """Take a daemon name and return a new instance of the daemon"""
    module_name = 'invisible_cities.daemons.' + daemon_name
    daemon_class  = getattr(import_module(module_name), daemon_name.capitalize())
    return daemon_class()

