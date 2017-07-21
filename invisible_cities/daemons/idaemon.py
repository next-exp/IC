from importlib import import_module
import traceback


def invoke_daemon(daemon_name):
    """Takes a daemon name and returns an instance of the daemon"""
    try:
        module_name = 'invisible_cities.daemons.' + daemon_name
        daemon_class  = getattr(import_module(module_name),
                                          daemon_name.capitalize())
    except ModuleNotFoundError:
        traceback.print_exc()
    else:

        return daemon_class()
