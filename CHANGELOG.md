# Changelog

## Voxtral Mini 4B experimental branch

- Replaced the previous speech-recognition stack with a single resident audio.cpp Voxtral Mini 4B Realtime 2602 Q4_K CUDA server, transcribing complete FireRedVAD utterances offline.
- Retained CPU FireRedVAD, original audio timing, Japanese-only filtering, PLaMo translation, nickname policies, translation memory, MPV playback controls, diagnostics, and owned-process cleanup.
- Simplified configuration, dependency installation, GUI controls, and diagnostic labels around the new backend.
- Added pinned Windows CUDA runtime and checkpoint integrity verification to the independent fresh-clone installer.
- Updated Windows executable packaging and the offline regression workflow for the experimental branch.

Automated protocol and lifecycle tests do not establish real Japanese transcription accuracy, RTX 3070 VRAM headroom, playback stability, or Windows executable availability. Those results must be reported with actual test/build evidence.
