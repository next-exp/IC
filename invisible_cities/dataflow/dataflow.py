# TODO: test implicit pipes in fork
# TODO: test count_filter
# TODO: test spy_count
# TODO: test polymorphic result of pipe
# TODO: Add test for failure to close sideways in branch
# TODO: test string_to_pick and its usage

import builtins
import functools
import itertools as it
import copy

from collections import namedtuple
from functools   import wraps
from asyncio     import Future
from contextlib  import contextmanager
from argparse    import Namespace
from operator    import itemgetter


@contextmanager
def closing(target):
    try:     yield
    finally: target.close()

def coroutine(generator_function):
    @wraps(generator_function)
    def proxy(*args, **kwds):
        coroutine = generator_function(*args, **kwds)
        next(coroutine)
        return coroutine
    return proxy

def coroutine_send(generator_function):
    @wraps(generator_function)
    def proxy(*args, **kwds):
        coroutine = generator_function(*args, **kwds)
        next(coroutine)
        return coroutine.send
    return proxy


NoneType = type(None)

def   _exactly_one(spec): return not isinstance(spec, (tuple, list, NoneType))
def _more_than_one(spec): return     isinstance(spec, (tuple, list)          )


# TODO: improve ValueError message
def map(op=None, *, args=None, out=None, item=None):
    if item is not None:
        if args is not None or out is not None:
            raise ValueError("dataflow.map: use of `item` parameter excludes both `args` and `out`")
        assert args is None and out is None
        args = out = item

    if args is None and out is None:
        def map_loop(target):
            with closing(target):
                while True:
                    target.send(op((yield)))
    else:
        if _exactly_one(args):
            args = args,

        merged_output = _exactly_one(out)
        if merged_output:
            out = out,

        def map_loop(target):
            with closing(target):
                while True:
                    data   = yield
                    values = (data[arg] for arg in args)
                    trans  = op(*values)
                    if merged_output:
                        trans = trans,
                    for name, value in zip(out, trans):
                        data[name] = value
                    target.send(data)

    return coroutine(map_loop)


def filter(predicate, *, args=None):
    if args is None:
        def filter_loop(target):
            with closing(target):
                while True:
                    val = yield
                    if predicate(val):
                        target.send(val)
    else:
        if _exactly_one(args):
            args = args,

        def filter_loop(target):
            with closing(target):
                while True:
                    data = yield
                    values = (data[arg] for arg in args)
                    if predicate(*values):
                        target.send(data)

    return coroutine(filter_loop)

FutureFilter = namedtuple('FutureFilter', 'future filter')
PassedFailed = namedtuple('PassedFailed', 'n_passed n_failed')

def count_filter(predicate, *, args=None):
    future = Future()
    n_passed = 0
    n_failed = 0
    if args is None:
        def filter_loop(target):
            nonlocal n_passed, n_failed
            try:
                with closing(target):
                    while True:
                        val = yield
                        passed = predicate(val)
                        if passed:
                            n_passed += 1
                            target.send(val)
                        else:
                            n_failed += 1
            finally:
                future.set_result(PassedFailed(n_passed, n_failed))
    else:
        if _exactly_one(args):
            args = args,

        def filter_loop(target):
            nonlocal n_passed, n_failed
            try:
                with closing(target):
                    while True:
                        data = yield
                        values = (data[arg] for arg in args)
                        passed = predicate(*values)
                        if passed:
                            n_passed += 1
                            target.send(data)
                        else:
                            n_failed += 1
            finally:
                future.set_result(PassedFailed(n_passed, n_failed))
    return FutureFilter(future=future, filter=coroutine(filter_loop))


def spy(op):
    @coroutine
    def spy_loop(target):
        with closing(target):
            while True:
                val = yield
                op(val)
                target.send(val)
    return spy_loop

def branch(*pieces):
    sideways = pipe(*pieces)
    @coroutine
    def branch_loop(downstream):
        with closing(sideways), closing(downstream):
            while True:
                val = yield
                sideways  .send(val)
                downstream.send(val)
    return branch_loop


@coroutine
def fork(*targets):
    targets = implicit_pipes(targets)
    try:
        while True:
            value = (yield)
            for t in targets:
                t.send(value)
    finally:
        for t in targets:
            t.close()



FutureSink = namedtuple('FutureSink', 'future sink')

def RESULT(generator_function):
    @wraps(generator_function)
    def proxy(*args, **kwds):
        future = Future()
        coroutine = generator_function(future, *args, **kwds)
        next(coroutine)
        return FutureSink(future, coroutine)
    return proxy

def sink(effect, *, args=None):
    if args is None:
        def sink_loop():
            while True:
                effect((yield))
    else:
        if _exactly_one(args):
            args = args,
        def sink_loop():
            while True:
                data   = yield
                values = (data[arg] for arg in args)
                effect(*values)
    return coroutine(sink_loop)()

def reduce(update, initial):
    @RESULT
    def reduce_loop(future):
        accumulator = copy.copy(initial)
        try:
            while True:
                accumulator = update(accumulator, (yield))
        finally:
            future.set_result(accumulator)
    return reduce_loop

@RESULT
def count(future):
    count = 0
    try:
        while True:
            yield
            count += 1
    finally:
        future.set_result(count)


FutureSpy = namedtuple("FutureSpy", "future spy")

def spy_count():
    pair = count()
    return FutureSpy(future = pair.future, spy = branch(pair.sink))


def stop_when(predicate):
    @sink
    def stop_when_loop(item):
        if predicate(item):
            raise StopPipeline
    return stop_when_loop


class StopPipeline(Exception): pass

def push(source, pipe, result=()):
    for item in source:
        try:
            pipe.send(item)
        except StopPipeline:
            break
    pipe.close()
    if isinstance(result, dict):
        return Namespace(**{k: v.result() for k, v in result.items()})
    if isinstance(result, Future):
        return result.result()
    return tuple(f.result() for f in result)


def pipe(*pieces):

    pieces = tuple(builtins.map(string_to_pick, pieces))

    def apply(arg, fn):
        return fn(arg)

    if hasattr(pieces[-1], 'close'):
        return functools.reduce(apply, reversed(pieces))
    else:
        def pipe_awaiting_sink(downstream):
            return pipe(*pieces, downstream)
        return pipe_awaiting_sink


def string_to_pick(component):
    if isinstance(component, str):
        return map(itemgetter(component))
    return component


def slice(*args, close_all=False):
    spec = builtins.slice(*args)
    start, stop, step = spec.start, spec.stop, spec.step
    if start is not None and start <  0: raise ValueError('slice requires start >= 0')
    if stop  is not None and stop  <  0: raise ValueError('slice requires stop >= 0')
    if step  is not None and step  <= 0: raise ValueError('slice requires step > 0')

    if start is None: start = 0
    if step  is None: step  = 1
    if stop  is None: stopper = it.count()
    else            : stopper = range((stop - start + step - 1) // step)
    @coroutine
    def slice_loop(target):
        with closing(target):
            for _ in range(start)             : yield
            for _ in stopper:
                target.send((yield))
                for _ in range(step - 1)      : yield
            if close_all: raise StopPipeline
            while True:
                yield
    return slice_loop


def implicit_pipes(seq):
    return tuple(builtins.map(if_tuple_make_pipe, seq))


def if_tuple_make_pipe(thing):
    return pipe(*thing) if type(thing) is tuple else thing


# TODO:
# + sum
# + dispatch
# + merge
# + eliminate finally-boilerplate from RESULT (with contextlib.contextmanager?)
# + graph structure DSL (mostly done: pipe, fork, branch (dispatch))
# + network visualization


######################################################################

if __name__ == '__main__':

    show   = sink(print)
    count_2_fut, count2 = count(); every2 = filter(lambda n:not n%2)(count2)
    count_5_fut, count5 = count(); every5 = filter(lambda n:not n%5)(count5)
    count_7_fut, count7 = count(); every7 = filter(lambda n:not n%7)(count7)
    square = map(lambda n:n*n)

    graph = fork(
        stop_when(lambda n:n>10),
        show,
        square(show),
        every2,
        every5,
        every7,
    )

    print(push(pipe   = graph,
               source = range(200),
               result = (count_2_fut, count_5_fut, count_7_fut)))
