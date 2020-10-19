#!/bin/bash

astrocv-simsky -s 512x512 --threads 1 \
     | \
astrocv-process -s 512x512 --threads 1 \
    --processing sub_bgd equalize --processing-params kernel=16 \
    --detect sv:CONTRAST --detect-params minCertainty=3.0 maxSize=10 nMaxObjects=6  \
    --detect-marker ellipse --draw-frames \
    --track hungalman --track-params m_maxSkippedFrames=3 --track-min-trace 10
