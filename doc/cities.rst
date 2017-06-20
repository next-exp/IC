IC cities
============

IC cities are scripts that read data from "persistent storage" (e.g, from disk), then (eventually) filter it and finally create some new data that
is written in PS.

As an example, consider the basic input data to IC. These are the so called
raw waveforms or RWF. A RWF is a vector representing the output of next
FE electronics. For example, NEXT PMTs output waveforms of typically 1.3 ms
sampled in bins of 25 ns. Each sample is an element of the PMT-RWF vector,
and there are 12 such vectors in the NEW detector (corresponding to 12 PMTs).
In turn, the SiPM electronics samples each 1 mus, thus the corresponding
SiPMRWF are a factor 40 smaller than the PMTRWF, but in exchange there
are near 2000 SiPMs in NEW.

The first step to work with the PMTRWF is to perform a process called
deconvolution. The PMT RWFs show a negative swing due to the HPF
introduced by the decoupling capacitor (needed to decouple the PMT
signal from the HV). Such a negative swing can be corrected offline
by an appropriate *deconvolution algorithm*.

This could be done as follows:

- Read RWF data from file into some (transient) representation. Notice that
persistent and transient representation of the data are in general different.

- Filter RWF data (e.g, eliminate malformed events if they exist).

- Transform RWF data into CWF data (corrected waveforms) by applying  a
 deconvolution algorithm.

- Write CWF to file.

One can then write a city that performs all the above tasks. In IC such city
is called ISIDORA. Notice that ISIDORA is a *concrete* city, with a well
defined task. However, some of the chores of ISIDORA are not exclusive to it.
For example, one could decide, after deconvolution of the PMTs, to compute
the calibrated sum of the CWFs and write the output of such operation. Or
perhaps one would like to search for peaks (e.g, the regions where the
waveform raises due to the presence of a physical signal), and write those
peaks. For example, the city of IRENE does the following tasks.

- Read RWF data from file.

- Deconvolute to produce CWF

- Compute calibrated sum of PMTs.

- Compute zero-suppression in SiPMs.

- Finds PMAPS, e.g, structures in which a peak in the PMTs is matched to
a signal in the SiPMs.

- Write PMAPs.

The point is that IRENE (a concrete city), needs to do the same
tasks than ISIDORA en then some more. The second difference is that the
data written by IRENE is different than the data written by ISIDORA.

How can one avoid duplication of code? One solution is that both ISIDORA
and IRENE call functions to perform specific tasks. For example one can
define a deconvolve_rwf (RWF --> CWD) in the IC API, and both ISIDORA
and IRENE can call it. One can then imagine a city as a simple algorithm
that puts together a set of functions, present in the API to achieve its
goal.

A second possibility is to introduce an extra layer between the API and
the concrete cities. The rationale for that would be the following.
Consider again the deconvoution algorithm. Suppose that, after some time,
we decide to change to a new deconvolution algorithm with better
performance. If IRENE and ISIDORA are calling directly the API function
then we need to change that call in two cities. Since the number of cities
that use deconvolution may be large (it is the first operation that
we perform), this means changing the call in a large number of cities.

Instead, both ISIDORA and IRENE can extend a base city (which we could
call DeconvolutionCity) which defines a deconvolution function, now
inherited by both cities (and by as many cities as needed).
the method  deconvolve_rwf in DeconvolutionCity calls in turn the
"free function" defined in the API. But if we want to change the function
there is now only one substitution to make (in an obvious place).

Thus, the current structure of the cities is:

-- Base cities (City, DeconvolutionCity, PmapCity...) which hold functions
needed for one or more specific cities in order to perform their tasks. Those base cities defined the interface to the concrete cities.

--The base cities, "choose" the API functions to be called for
an specific task.   
