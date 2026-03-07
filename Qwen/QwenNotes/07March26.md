# wake2vec devlog 2026-03-07

Spent some time touching grass today but all I could think about was Qwen. 

[Hardstyle 2](https://soundcloud.com/fredagain/hardstyle-2?in=houseof_kyri/sets/sal-paradise&si=d5d0c895a69241118740210396c590a2&utm_source=clipboard&utm_medium=text&utm_campaign=social_sharing)

## Qwen 2.5-14B P1 sesh 5

Resumed from `sentry_step_0180.pt` with `STEP_OFFSET=180`. 

GPU allocation: 3h20m today.

Sentry at step 200 landed clean (428MB). Training speed improved to ~115s/step (down from 135s in session 4).

### P1 loss table

| Step | Train | Val | Session |
|------|-------|-----|---------|
| 50 | 345.00 | 21.54 | 1 | 
| 100 | 321.48 | 20.98 | 1 |
| 150 | 303.07 | 20.64 | 4 |
| 200 | — | — | 5 |

## Notes

Currently reading Rhett Davis, *Arborescence*, so you're not just getting some Fred Again from this devlog but also a book recommendation. 

From page 64:

> "I would like to ask Ally if they are real or another advanced, self-replicating simulation, but it's against the company's People Policy, so I don't."
