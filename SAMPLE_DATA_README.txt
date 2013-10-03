------------------------------------------------------------------
-  SAMPLE DATA READ ME                                           -
------------------------------------------------------------------
Reference:
 Hamilton LS, Sohl-Dickstein J, Huth AG, Carels VM, Bao S (2013). Optogenetic
        Activation of an Inhibitory Network Enhances Functional Connectivity in
        Auditory Cortex.  Neuron (in press).

Sample data here are from recordings taken from adult mouse
auditory cortex with a NeuroNexus A4x4 polytrode. The approximate
depths of rows 1, 2, 3, and 4 are 200, 300, 400, and 500 um below
the pial surface, respectively.

Sound stimuli: 23 conditions, 1 noise burst (stimulus #1) and 22
    tone pip stimuli (4 - 75kHz in 0.2 oct steps).  Intensities
    are collapsed but include levels of 50, 60, and 70 dB.

Spike channels: 16 channels from a 4 x 4 polytrode configuration.
    Sites are reordered so that row 1 in the spike matrix is the
    top left recording site, and numbers continue from left to 
    right and top to bottom as follows:

        1   2   3   4
        5   6   7   8
        9  10  11  12
       13  14  15  16

Bin size was set to 5 ms (bin_size = 0.05).
