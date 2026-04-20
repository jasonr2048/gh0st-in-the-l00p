Gh0st in the L00p – exhibition mode notes

This branch contains the current exhibition-system pass for dual-screen playback.

Key file:
- exhibition/text_payload.json
  Text pools for the exhibition system.

How to run playback:
python app.py --mode exhibition --duration-seconds 180

How to export:
python app.py --mode exhibition --export --duration-seconds 180

Output:
- exports/exhibition/screen_A.avi
- exports/exhibition/screen_B.avi

Notes:
- Screen A and Screen B are intended as separate synchronized outputs.
- The text system currently lives in exhibition/text_payload.json.