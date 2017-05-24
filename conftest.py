import os

from hypothesis import settings
from hypothesis import Verbosity

# In addition to the 'default' profile, we provide
settings.register_profile("hard"     , settings(max_examples = 1000))
settings.register_profile("dev"      , settings(max_examples =   10))
settings.register_profile("noshrink" , settings(max_examples =   10,
                                                max_shrinks  =    0))
settings.register_profile("debug"    , settings(max_examples =   10,
                                                verbosity=Verbosity.verbose))
settings.load_profile(os.getenv(u'HYPOTHESIS_PROFILE', 'dev'))
