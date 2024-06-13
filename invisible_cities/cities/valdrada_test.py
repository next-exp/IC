import os
import tables as tb

from .  valdrada           import valdrada
from .. core.testing_utils import assert_tables_equality


def test_valdrada_contains_all_tables(trigger_config):
    conf, PATH_OUT = trigger_config
    valdrada(**conf)
    with tb.open_file(PATH_OUT) as h5out:
        assert "Run"             in h5out.root
        assert "Run/events"      in h5out.root
        assert "Run/runInfo"     in h5out.root
        assert "Trigger"         in h5out.root
        assert "Trigger/events"  in h5out.root
        assert "Trigger/trigger" in h5out.root
        assert "Trigger/DST"     in h5out.root


def test_valdrada_exact_result_multipeak(ICDATADIR, trigger_config):
    true_out          = os.path.join(ICDATADIR, "exact_result_multipeak_valdrada.h5")
    conf, PATH_OUT   = trigger_config
    valdrada(**conf)

    tables = ("Trigger/events", "Trigger/trigger", "Trigger/DST",
              "Run/events"      , "Run/runInfo")

    with tb.open_file(true_out)  as true_output_file:
        with tb.open_file(PATH_OUT) as      output_file:
            for table in tables:
                assert hasattr(output_file.root, table)
                got      = getattr(     output_file.root, table)
                expected = getattr(true_output_file.root, table)
                assert_tables_equality(got, expected)


def test_valdrada_exact_result(ICDATADIR, trigger_config):
    true_out          = os.path.join(ICDATADIR, "exact_result_valdrada.h5")
    conf, PATH_OUT    = trigger_config
    conf['trigger_config']['multipeak'] = None
    valdrada(**conf)

    tables = ("Trigger/events", "Trigger/trigger", "Trigger/DST",
              "Run/events"      , "Run/runInfo")

    with tb.open_file(true_out)  as true_output_file:
        with tb.open_file(PATH_OUT) as      output_file:
            for table in tables:
                assert hasattr(output_file.root, table)
                got      = getattr(     output_file.root, table)
                expected = getattr(true_output_file.root, table)
                assert_tables_equality(got, expected)
