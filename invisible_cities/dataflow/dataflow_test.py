import dataflow as df

from pytest import raises
from pytest import mark
parametrize = mark.parametrize


from hypothesis            import given
from hypothesis.strategies import tuples
from hypothesis.strategies import integers
from hypothesis.strategies import none
from hypothesis.strategies import one_of


def test_simplest_pipeline():

    # The simplest possible pipeline has one source directly connected
    # to one sink.

    # We avoid using a lazy source so that we can compare the result
    # with the input
    the_source = list(range(20))

    # In this example the sink will simply collect the data it
    # receives, into a list.
    result = []
    the_sink = df.sink(result.append)

    # Use 'push' to feed the source into the pipe.
    df.push(source=the_source, pipe=the_sink)

    assert result == the_source


def test_fork():

    # Dataflows can be split with 'fork'

    the_source = list(range(10, 20))

    left  = [];  left_sink = df.sink( left.append)
    right = []; right_sink = df.sink(right.append)

    df.push(source = the_source,
            pipe   = df.fork( left_sink,
                             right_sink))

    assert left == right == the_source


def test_map():

    # The pipelines start to become interesting when the data are
    # transformed in some way. 'map' transforms every item passing
    # through the pipe by applying the supplied operation.

    def the_operation(n): return n*n
    square = df.map(the_operation)

    the_source = list(range(1,11))

    result = []
    the_sink = df.sink(result.append)

    df.push(source = the_source,
            pipe   = square(the_sink))

    assert result == list(map(the_operation, the_source))


def test_pipe():

    # The basic syntax requires any element of a pipeline to be passed
    # as argument to the one that precedes it. This looks strange to
    # the human reader, especially when using parametrized
    # components. 'pipe' allows construction of pipes from a sequence
    # of components.

    # Using 'pipe', 'test_map' could have been written like this:

    def the_operation(n): return n*n
    square = df.map(the_operation)

    the_source = list(range(1,11))

    result = []
    the_sink = df.sink(result.append)

    df.push(source = the_source,
            pipe   = df.pipe(square, the_sink))

    assert result == list(map(the_operation, the_source))


def test_longer_pipeline():

    # Pipelines can have arbitrary lengths

    the_source = list(range(1,11))

    result = []
    the_sink = df.sink(result.append)

    df.push(source = the_source,
            pipe   = df.pipe(df.map(lambda n:n+1),
                             df.map(lambda n:n*2),
                             df.map(lambda n:n-3),
                             df.map(lambda n:n/4),
                             the_sink))

    assert result == [ (((n+1)*2)-3)/4 for n in the_source ]


def test_filter():

    # 'filter' can be used to eliminate data

    def the_predicate(n): return n % 2
    odd = df.filter(the_predicate)

    the_source = list(range(20, 30))

    result = []
    the_sink = df.sink(result.append)

    df.push(source = the_source,
            pipe   = df.pipe(odd, the_sink))

    assert result == list(filter(the_predicate, the_source))


def test_flatmap():

    # Flatmap accepts a function which returns an iterable.
    # While using map with such a function would result in entire
    # iterables being sent downstream as separate elements,
    # flatmap concatenates the contents of these iterables producing one
    # long stream of the individual elements contained in those iterables.
    # Equivalent to
    # for incoming_item in upstream:
    #     for outgoing_item in make_sequence(incoming_item):
    #         send_downstream(outgoing_item)

    the_operation = range
    ranges = df.flatmap(the_operation)

    the_source = list(range(1,11))

    result = []
    the_sink = df.sink(result.append)

    df.push(source = the_source,
            pipe   = ranges(the_sink))

    expected = []
    for s in the_source:
        for item in the_operation(s):
            expected.append(item)

    assert result == expected


def test_flatmap_args_out():

    # A more complex case, taking and producing multiple values

    def repeat_numbers(a, b):
        return [(b, str(b)*i) for i in range(1, a+1)]

    the_operation = df.flatmap( repeat_numbers
                              , args = list("ab")
                              , out  = list("cd")
                              )

    the_source = [ dict(a=1, b=4)
                 , dict(a=2, b=5)
                 , dict(a=3, b=6)
                 ]

    result_a = []
    result_b = []
    def collect(a, b):
        result_a.append(a)
        result_b.append(b)

    the_sink = df.sink(collect, args=list("cd"))

    df.push(source = the_source,
            pipe   = the_operation(the_sink))

    expected_a = [ 4 ,  5 ,   5 ,  6 ,   6 ,    6 ]
    expected_b = ["4", "5", "55", "6", "66", "666"]
    assert result_a == expected_a
    assert result_b == expected_b


def test_count():

    # 'count' is an example of a sink which only produces a result
    # once the stream of data flowing into the pipeline has been
    # closed. Such results are retrieved from futures which are
    # created at the time a 'count' instance is created: a namedtuple
    # containing the sink and its corresponding future is returned.

    count = df.count()

    the_source = list(range(30,40))

    df.push(source = the_source,
            pipe   = count.sink)

    assert count.future.result() == len(the_source)


def test_push_futures():

    # 'push' provides a higher-level interface to using such futures:
    # it optionally accepts a tuple of futures, and returns a tuple of
    # their results

    count_all = df.count()
    count_odd = df.count()

    the_source = list(range(100))

    result = df.push(source = the_source,
                     pipe   = df.fork(                                 count_all.sink,
                                      df.pipe(df.filter(lambda n:n%2), count_odd.sink)),
                     result = (count_odd.future, count_all.future))

    all_count = len(the_source)
    odd_count = all_count // 2
    assert result == (odd_count, all_count)


def test_reduce():

    # 'reduce' provides a high-level way of creating future-sinks such
    # as 'count'

    # Make a component just like df.sum
    from operator import add
    total = df.reduce(add, initial=0)

    # Create two instances of it, which will be applied to different
    # (forked) sub-streams in the network
    total_all = total()
    total_odd = total()

    N = 15
    the_source = list(range(N))

    result = df.push(source = the_source,
                     pipe   = df.fork(                                 total_all.sink,
                                      df.pipe(df.filter(lambda n:n%2), total_odd.sink)),
                     result = (total_all.future, total_odd.future))

    sum_all, sum_odd = sum(the_source), (N // 2) ** 2
    assert result == (sum_all, sum_odd)


@mark.xfail
def test_sum():
    raise NotImplementedError


def test_stop_when():

    # 'stop_when' can be used to stop all branches of the network
    # immediately.

    countfuture, count = df.count()

    limit, step = 10, 2

    import itertools

    result = df.push(source = itertools.count(start=0, step=step),
                     pipe   = df.fork(df.stop_when(lambda n:n==limit),
                                      count),
                     result = (countfuture,))

    assert result == (limit // step,)


def test_stateful_stop_when():

    @df.coroutine_send
    def n_items_seen(n):
        yield # Will stop here on construction
        for _ in range(n):
            yield False
        yield True

    countfuture, count = df.count()

    import itertools
    limit, step = 10, 2

    result = df.push(source = itertools.count(start=0, step=step),
                     pipe   = df.fork(df.stop_when(n_items_seen(limit)),
                                      count),
                     result = (countfuture,))

    assert result == (limit,)


def test_spy():

    # 'spy' performs an operation on the data streaming through the
    # pipeline, without changing what is seen downstream. An obvious
    # use of this would be to insert a 'spy(print)' at some point in
    # the pipeline to observe the data flow through that point.

    the_source = list(range(50, 60))

    result = []; the_sink = df.sink(result.append)
    spied  = []; the_spy  = df.spy ( spied.append)

    df.push(source = the_source,
            pipe   = df.pipe(the_spy, the_sink))

    assert spied == result == the_source


def test_branch():

    # 'branch', like 'spy', allows you to insert operations on a copy
    # of the stream at any point in a network. In contrast to 'spy'
    # (which accepts a single plain operation), 'branch' accepts an
    # arbitrary number of pipeline components, which it combines into
    # a pipeline. It provides a more convenient way of constructing
    # some graphs that would otherwise be constructed with 'fork'.

    # Some pipeline components
    c1 = []; C1 = df.sink(c1.append)
    c2 = []; C2 = df.sink(c2.append)
    e1 = []; E1 = df.sink(e1.append)
    e2 = []; E2 = df.sink(e2.append)

    A = df.map(lambda n:n+1)
    B = df.map(lambda n:n*2)
    D = df.map(lambda n:n*3)

    # Two eqivalent networks, one constructed with 'fork' the other
    # with 'branch'.
    graph1 = df.pipe(A, df.fork(df.pipe(B,C1),
                                df.pipe(D,E1)))

    graph2 = df.pipe(A, df.branch(B,C2), D,E2)

    # Feed the same data into the two networks.
    the_source = list(range(10, 50, 4))
    df.push(source=the_source, pipe=graph1)
    df.push(source=the_source, pipe=graph2)

    # Confirm that both networks produce the same results.
    assert c1 == c2
    assert e1 == e2


def test_chain_pipes():

    # Pipelines must end in sinks. If the last component of a pipe is
    # not a sink, the pipe may be used as a component in a bigger
    # pipeline, but it will be impossible to feed any data into it
    # until it is connected to some other component which ends in a
    # sink.

    # Some basic pipeline components
    s1 = []; sink1 = df.sink(s1.append)
    s2 = []; sink2 = df.sink(s2.append)

    A = df.map(lambda n:n+1)
    B = df.map(lambda n:n*2)
    C = df.map(lambda n:n-3)

    # Two different ways of creating equivalent networks: one of them
    # groups the basic components into sub-pipes
    graph1 = df.pipe(        A, B,          C, sink1)
    graph2 = df.pipe(df.pipe(A, B), df.pipe(C, sink2))

    # Feed the same data into the two networks
    the_source = list(range(40))

    df.push(source=the_source, pipe=graph1)
    df.push(source=the_source, pipe=graph2)

    # Confirm that both networks produce the same results.
    assert s1 == s2


def test_reuse_unterminated_pipes():

    # Open-ended pipes must be connected to a sink before they can
    # receive any input. Open-ended pipes are reusable components: any
    # such pipe can be used in different points in the same or
    # different networks. They are completely independent.

    def add(n):
        return df.map(lambda x:x+n)

    A,B,C,D,E,X,Y,Z = 1,2,3,4,5,6,7,8

    component = df.pipe(add(X),
                        add(Y),
                        add(Z))

    s1 = []; sink1 = df.sink(s1.append)
    s2 = []; sink2 = df.sink(s2.append)

    # copmonent is being reused twice in this network
    graph = df.pipe(add(A),
                    df.branch(add(B), component, add(C), sink1),
                    add(D), component, add(E), sink2)

    the_source = list(range(10,20))
    df.push(source=the_source, pipe=graph)

    assert s1 == [ n + A + B + X + Y + Z + C for n in the_source ]
    assert s2 == [ n + A + D + X + Y + Z + E for n in the_source ]


def test_reuse_terminated_pipes():

    # Sink-terminated pipes are also reusable, but do note that if
    # such components are reused in the same graph, the sink at the
    # end of the component will receive inputs from more than one
    # branch: they share the sink; the branches are joined.

    def add(n):
        return df.map(lambda x:x+n)

    A,B,C,X,Y,Z = 1,2,3,4,5,6

    collected_by_sinks = []; sink1 = df.sink(collected_by_sinks.append)

    component = df.pipe(add(X),
                        add(Y),
                        add(Z),
                        sink1)

    graph = df.pipe(add(A),
                    df.branch(add(B), component),
                              add(C), component)

    the_source = list(range(10,20))
    df.push(source=the_source, pipe=graph)

    route1 = [ n + A + B + X + Y + Z for n in the_source ]
    route2 = [ n + A + C + X + Y + Z for n in the_source ]

    def intercalate(a,b):
        return [ x for pair in zip(a,b) for x in pair ]

    assert collected_by_sinks == intercalate(route1, route2)


small_ints         = integers(min_value=0, max_value=15)
small_ints_nonzero = integers(min_value=1, max_value=15)
slice_arg          = one_of(none(), small_ints)
slice_arg_nonzero  = one_of(none(), small_ints_nonzero)

@given(one_of(tuples(small_ints),
              tuples(small_ints, small_ints),
              tuples(slice_arg,  slice_arg, slice_arg_nonzero)))
def test_slice_downstream(spec):
    the_source = list('abcdefghij')
    result = []
    the_sink = df.sink(result.append)

    df.push(source = the_source,
            pipe   = df.pipe(df.slice(*spec, close_all=False), the_sink))


    specslice = slice(*spec)
    assert result == the_source[specslice]
    assert result == the_source[specslice.start : specslice.stop : specslice.step]


def test_slice_null_close_all():
    spec       = (0,)

    the_source = list('abcdefghij')
    result     = []
    the_sink   = df.sink(result.append)

    df.push(source = the_source,
            pipe   = df.pipe(df.slice(*spec, close_all=True), the_sink))

    specslice = slice(*spec)
    assert result == the_source[specslice]
    assert result == the_source[specslice.start : specslice.stop : specslice.step]


@parametrize('args',
             ((      -1,),
              (None, -1),
              (-1, None),
              (None, None, -1),
              (None, None,  0),
             ))
def test_slice_raises_ValueError(args):
    with raises(ValueError):
        df.slice(*args)


@mark.xfail
def test_pipes_must_end_in_a_sink():
    raise NotImplementedError
