file name is obviously a joke because i have no life. 

## 2026-02-14 P2 loss check

| step | training loss | validation loss |
| --- | --- | --- |
| 1200 | 0.5797 | 0.6255 |
| 1400 | 0.638800	| 0.639327 |
| 1600 | 0.572200	| 0.646048 |
| 1800 | 0.610400	| 0.659436 |
| 2000 | 0.494300	| 0.667894 |

Starting to overfit at step 2000. Val loss has been climbing monotonically since step 1200 while train dropped. Gap tripled from ~0.05 to ~0.17. Not catastrophic, and LR is decaying on cosine schedule so it should slow down, but best checkpoint for generation is probably somewhere around step 1400-1600 where train/val were tightest. Letting it run to 3000 anyway to see the full curve.

1000 steps to go.

Fred Again dropped USB. [lights burn dimmer](https://soundcloud.com/fredagain/lights-burn-dimmer?in=fredagain/sets/usb-848345661&si=b37c390e419e463e8b1c12c5b237eee8&utm_source=clipboard&utm_medium=text&utm_campaign=social_sharing) is a fckn banger.
