# TODO: test implicit pipes in fork

import builtins
import itertools as it
from collections import namedtuple
from functools   import wraps
from asyncio     import Future
from contextlib  import contextmanager

import functools

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

def map(op):
    @coroutine
    def map_loop(target):
        with closing(target):
            while True:
                target.send(op((yield)))
    return map_loop

def filter(predicate):
    @coroutine
    def filter_loop(target):
        with closing(target):
            while True:
                val = yield
                if predicate(val):
                    target.send(val)
    return filter_loop

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
    def bbb(downstream):
        with closing(downstream):
            while True:
                val = yield
                sideways  .send(val)
                downstream.send(val)
    return bbb


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

def sink(effect):
    @coroutine
    def sink_loop():
        while True:
            effect((yield))
    return sink_loop()

def reduce(update, initial):
    @RESULT
    def reduce_loop(future):
        accumulator = initial
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


def stop_when(predicate):
    @sink
    def stop_when_loop(item):
        if predicate(item):
            raise StopPipeline
    return stop_when_loop


class StopPipeline(Exception): pass

def push(source, pipe, futures=()):
    for item in source:
        try:
            pipe.send(item)
        except StopPipeline:
            break
    pipe.close()
    return tuple(f.result() for f in futures)

def pipe(*pieces):

    def apply(arg, fn):
        return fn(arg)

    if hasattr(pieces[-1], 'close'):
        return functools.reduce(apply, reversed(pieces))
    else:
        def pipe_awaiting_sink(downstream):
            return pipe(*pieces, downstream)
        return pipe_awaiting_sink


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

    print(push(pipe    = graph,
               source  = range(200),
               futures = (count_2_fut, count_5_fut, count_7_fut)))
