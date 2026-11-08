# VB-Cable setup (interceptor mode)

Interceptor mode puts SpeechShield **between your call app and your
headphones**. Other people's voices are checked before you hear them; your
own microphone is not touched.

```
Discord / Zoom / game ──► CABLE Input ══(VB-Cable)══► CABLE Output ──► SpeechShield ──► your headphones
```

## 1. Install VB-Cable (free, ~2 minutes)

1. Download from <https://vb-audio.com/Cable/>
2. Extract the ZIP, right-click `VBCABLE_Setup_x64.exe` → **Run as administrator** → *Install Driver*
3. Restart your PC

## 2. Send the call audio into the cable

| App | Setting |
|---|---|
| Discord | Settings → Voice & Video → **Output Device** → `CABLE Input (VB-Audio Virtual Cable)` |
| Zoom | Settings → Audio → **Speaker** → `CABLE Input (VB-Audio Virtual Cable)` |
| Teams | Settings → Devices → **Speaker** → `CABLE Input` |
| A game / anything else | Windows Settings → System → Sound → **Volume mixer** → set that app's output to `CABLE Input` |

Leave your **Windows default output** on your real headphones: that is where
SpeechShield plays the cleaned audio.

## 3. Check, then start

```powershell
python main.py --doctor --mode interceptor     # verifies the cable, ffmpeg, models, GPU…
python main.py --mode interceptor              # bleep toxic words (default)
python main.py --mode interceptor --censor-mode voice_clone   # replace them in the speaker's voice
```

Useful flags: `--output-device "Headphones"` (play somewhere other than the
default), `--censor-mode mute|bleep|voice_clone|tts`, `--no-dashboard`.
`python main.py --list-devices` shows every device name.

The dashboard (<http://localhost:8501>) shows each utterance, what was
censored and how long it was held.

## What you will hear

- **Clean speech**: the original voice, about 1 second behind real time.
  With a supported NVIDIA GPU, audio streams out word by word as soon as each
  word is confirmed clean (typically 1.0–1.5 s behind). On CPU it is released
  per sentence (typically 1.5–3 s behind).
- **Toxic words**: bleeped, muted, or (voice_clone) swapped for a polite word in
  the same speaker's voice. The rest of the sentence is untouched.
- **Silence between sentences** is shortened automatically to win back the delay.
- **Fail-safe**: if the AI cannot judge something in time, it is muted rather
  than played (`fail_mode: closed`). Known-toxic audio is never passed through.

## Troubleshooting

| Problem | Fix |
|---|---|
| `Input device 'CABLE Output' not found` | VB-Cable isn't installed, or the PC wasn't restarted afterwards |
| You hear nothing at all | Your Windows default output is set to `CABLE Input`; set it back to your headphones |
| You hear people twice | The call app still outputs to your headphones too; set it to `CABLE Input` only |
| Crackling or gaps | Raise `interceptor.delay_seconds` (e.g. 1.2) or close other GPU-heavy apps |
| Long delays | Run `--doctor`: if ASR is on CPU, install `requirements-gpu.txt` to use your NVIDIA GPU |
| Your own voice isn't filtered | By design: only incoming audio is intercepted |
