# Background

Remote collaboration suffers from the loss of nonverbal cues that are naturally available in face-to-face settings. Among these cues, **gaze** is particularly critical: it supports joint attention and enables efficient *deixis* — the act of referring to a shared object (e.g., "this one", "here"). Without shared gaze, interlocutors must compensate with more verbose language, slowing down reference resolution and increasing grounding cost (Clark & Brennan, 1991).

Prior work has established that sharing accurate gaze information improves collaborative deixis — reducing referential expressions, increasing accuracy, and lowering task load (D'Angelo & Gergle, 2016, 2018; Velichkovsky, 1995). However, these studies uniformly rely on dedicated infrared (IR) eye trackers (e.g., Tobii Pro), which achieve angular errors of roughly 0.2°–1.2°.

Webcam-based eye tracking, by contrast, relies on CNN/Transformer regression from face images and produces substantially noisier estimates — typically 2.5°–5° error under benchmark conditions (MPIIFaceGaze), and worse under unconstrained settings. This noise gap raises an open question: **does noisy webcam gaze still support collaborative deixis, and if so, under what conditions?**

Existing noise studies (e.g., Pavlovych & Stuerzlinger on spatial jitter × latency; work on sense of agency under jittery delay) have been conducted in single-user pointing tasks, not collaborative reference tasks. The intersection of webcam-level gaze noise × two-person deixis / grounding remains unexamined.

This project builds the eye tracking infrastructure necessary to run that experiment. The system supports both **webcam-based** and **IR-based** (Tobii EyeX) eye tracking, outputting gaze coordinates over OSC in a unified format so that the two conditions are directly comparable. By running the same collaborative task under both conditions, noise level becomes a controlled independent variable, allowing the study to isolate how much gaze accuracy matters for deixis and grounding.

# Purpose of this project
* To start experiment of CoGaze.
* CoGaze evaluate whether webcam eye tracking helps remote collaboration, in term of its feature: noisy.
* Research Questions:
    * RQ1
        * To what extent do webcam gaze noise and presentation methods improve deixis, such as accuracy, speed, and redundancy of referential expressions in reference resolution? 
        * How do these differences affect task difficulty (number of distractors, similarity)?
    * RQ2 Not to be written here, to avoid data exploit. next time :)
    * RQ3 Not to be written here, to avoid data exploit. next time :)

# What this project DO
* Face fixed eye tracking system
    * head should not move while doing experiment
* OSC data transmission
    * format: uncertain yet, to be determined based on experiment condition
        * should provide 
            * standardized gaze coordinates (top-left: (0,0) bottom-right(1,1))
            * certainty of gaze(from 0 to 1)
* Usable Interface
    * can do calibrate easily
    * can detect abnormality of gaze
    * should be executable without security vulnerability
    * several options: run with IR, run without IR
    * camera preview, debug mode
    * evaluate mode
# What this project DO NOT DO
* Creating entire system
* Headpose handled webcam eye tracking system(future work)
* creating Deep learning based eye tracking system from zero

# Experimental Conditions
| Condition | Gaze shared with partner | IR running |
|-----------|--------------------------|------------|
| IR | IR gaze | yes |
| Webcam | Raw webcam gaze | yes (ground truth) |
| Webcam Filtered | Filtered webcam gaze | yes (ground truth) |

IR runs in all conditions. In Webcam and Webcam Filtered conditions, IR data is recorded simultaneously to characterize the actual noise present during the experiment (i.e., how far the webcam estimate deviated from where the participant truly looked).

# Open Challenges

## Core system
1. **OSC layer definition** — message format, port, timing/frequency
2. **Tobii EyeX 32-bit DLL constraint** — the SDK DLL is 32-bit only; Tobii has discontinued EyeX updates. Options to investigate:
    * Run 32-bit Python on 64-bit OS
    * Search for an unofficial 64-bit build of the DLL
    * Bridge: run 32-bit subprocess and pipe gaze data to a 64-bit main process
3. **Webcam gaze implementation** — geometric model (MediaPipe iris landmarks → 2nd-order polynomial calibration) preferred over DL; CPU-only, face-fixed setup, noise must be interpretable for IR comparison
4. **Timestamp design** — store both wall-clock ms (absolute time) and monotonic ns (cross-stream alignment) at capture time; Windows wall clock can drift between streams
5. **Pipeline latency offset** — MediaPipe + OSC adds ~30–80 ms lag relative to IR; measure and correct per session if automatic estimation is feasible

## Calibration
6. **Calibration design** — point count, spatial distribution, and hold-out validation to avoid overfitting
7. **Calibration UI** — re-calibration triggerable at any time via button; show pass/fail verdict (not raw numbers) so experimenter can judge quickly under pressure; each dot triggered by experimenter, not auto-timed

## Session management
8. **Session start UI** — participant ID and condition set before data flows; condition locked once recording starts, requiring explicit confirmation to change
9. **Pre-calibration data flagging** — data before successful calibration flagged in CSV for easy exclusion in post-hoc analysis

## Data logging
10. **CSV logging** — independent of OSC; every row includes condition, participant ID, session ID; flush per row or small batch to survive crashes

## Monitoring
11. **OSC connection status** — persistent live/dead indicator; dropped connection visually distinct from idle

## Performance
12. **Thread architecture** — MediaPipe, Tobii callback, CSV writer, and OSC sender must run on separate threads; use `queue.Queue(maxsize=2)` to drop stale frames rather than queuing; set thread priorities (Tobii: HIGHEST, MediaPipe: ABOVE_NORMAL); validate p99 frame time and Tobii inter-sample jitter before the experiment

## Security
13. **CSV formula injection** — strip leading `= + - @` from any user-supplied string (e.g. participant ID) before writing to CSV, to prevent formula execution when opened in Excel

# Hope
* accuracy: 1-7 deg