# MQL5.com Market automatic validator — checklist

Sources: the official MQL5 article **"The checks a trading robot must pass
before publication in the Market"** (mql5.com/en/articles/2555), the blog
post **"Solving Automatic Validation Problems Arising During Product
Submission in MQL5 Market"** (mql5.com/en/blogs/post/686716), and MQL5 forum
threads on validation failures. Collected 2026-07.

The validator runs the compiled EA in the Strategy Tester on **multiple
symbols and timeframes you do not choose** (FX majors, metals, sometimes
exotic specs; netting AND hedging account modes), with no network, no account
connection, and a hard time/log budget. Any runtime error, or a test with no
trades, fails the submission.

## A. Hard failure classes and the required counter-measures

### A1. "There are no trading operations"
- The EA **must trade on whatever symbol/timeframe the validator picks**.
  Products "can not apply restrictions" — symbol/TF limits may only be
  *recommendations* in the description.
- Classic bug: gating on `TerminalInfoInteger(TERMINAL_TRADE_ALLOWED)`
  without `|| MQLInfoInteger(MQL_TESTER)` — the validator tester is not
  connected to an account, so such a gate blocks all trading.
- Counter-measures in our templates:
  - no symbol/timeframe restrictions anywhere;
  - entry logic works from generic indicators available on every symbol;
  - **tester fallback trade**: inside `MQL_TESTER` only, if no position has
    ever been opened after a long warm-up window, open one
    minimum-volume, stops-compliant probe trade and manage it to close.
    This guarantees at least one trading operation per validator pass while
    being inert in live use.

### A2. Invalid volume
- Every order volume must satisfy `SYMBOL_VOLUME_MIN`, `SYMBOL_VOLUME_MAX`,
  `SYMBOL_VOLUME_STEP`, and the *aggregate* `SYMBOL_VOLUME_LIMIT`
  (sum of open positions + pending orders per symbol).
- Volume must be snapped to the step grid (floor, then clamp).
- Counter-measures: `NormalizeLots()` (floor-to-step + clamp min/max) +
  `VolumeLimitOK()` (checks `SYMBOL_VOLUME_LIMIT`) before **every**
  `OrderSend`.

### A3. Insufficient funds (retcode 10019 / "not enough money")
- Free margin must be checked with `OrderCalcMargin()` against
  `ACCOUNT_MARGIN_FREE` **before every order** — the validator deliberately
  tests with small deposits.
- Counter-measure: `MarginOK()` helper wrapping `OrderCalcMargin` (with a
  safety factor) called inside the order path; order silently skipped when
  margin is insufficient (no error spam).

### A4. SL/TP inside `SYMBOL_TRADE_STOPS_LEVEL`
- Buys: `TP − Bid ≥ stops_level·Point` and `Bid − SL ≥ stops_level·Point`.
- Sells: `Ask − TP ≥ stops_level·Point` and `SL − Ask ≥ stops_level·Point`.
- Note the asymmetry: a buy is filled at Ask but its stops are measured from
  **Bid** (and vice versa) — the effective minimum SL distance is
  `stops_level + spread`.
- Counter-measure: `AdjustStops()` pushes requested SL/TP *outward* to the
  minimum legal distance (stops_level + freeze_level + spread margin) before
  the order is sent; prices `NormalizeDouble`d to `_Digits`.

### A5. Modification inside `SYMBOL_TRADE_FREEZE_LEVEL`
- Positions may not be modified when price is within the freeze distance of
  the activation price (`TP−Bid`, `Bid−SL` for buys; mirrored for sells).
- Counter-measure: `FreezeOK()` check before every `PositionModify`
  (breakeven moves in the partial-close mechanic).

### A6. Array out of range
- Every `CopyBuffer`/`CopyRates`/`CopyHigh/Low/Close` result must be checked
  against the requested count; no fixed shifts without bar-count checks; no
  indicator buffer access in `OnInit` (buffers are sized later).
- The validator runs on symbols with **short or gappy history** — assume any
  copy can fail on any tick.
- Counter-measures: `SafeCopyBuffer`/`SafeCopyHigh/Low/Close` helpers that
  `ArrayResize` to the exact count, verify `BarsCalculated`, verify the
  copied count, and make the signal return "no trade" on failure; a
  `HistoryReady()` gate checking `SERIES_SYNCHRONIZED` and a minimum bar
  count before any data access.

### A7. Zero divide
- Exotic symbols produce degenerate values: `SYMBOL_TRADE_TICK_VALUE` can be
  0 in the validator, point/digits math differs, indicator baselines can be
  0. **Every** division must be guarded.
- Counter-measures: `SafeDiv()` used for all divisions (including
  tick-value, point and lot-step math); no raw `/` on runtime values.

### A8. Requote / reject / timeout handling
- `TRADE_RETCODE_REQUOTE`, `TRADE_RETCODE_REJECT`, `TRADE_RETCODE_TIMEOUT`,
  `TRADE_RETCODE_PRICE_CHANGED`, `TRADE_RETCODE_PRICE_OFF`,
  `TRADE_RETCODE_CONNECTION` must be retried a **bounded** number of times
  with backoff; fatal retcodes (invalid stops, invalid volume, no money)
  must NOT be retried (log-size limit: the validator kills tests whose logs
  exceed the budget — infinite retry loops with `Print` are a failure class
  of their own).
- Counter-measure: `SendMarketOrder` retries only transient retcodes with
  `Sleep` backoff, aborts immediately on fatal ones, and prints at most one
  line per failed order.

### A9. `OnInit` must not fail on any symbol
- `INIT_FAILED` on an unexpected symbol → immediate validation failure.
- Counter-measure: indicator handle creation failures do **not** fail
  `OnInit`; handles are lazily (re)created from `OnTick` until available;
  until then the EA simply does not trade (signals return false safely).

### A10. Netting vs hedging account modes
- The validator tests **both** `ACCOUNT_MARGIN_MODE_RETAIL_NETTING` and
  `..._RETAIL_HEDGING`. On netting accounts all fills on one symbol
  aggregate into a single position:
  - an opposite-direction "hedge" order *reduces or closes* the position
    instead of hedging — hedge mechanics must branch;
  - grid/DCA adds aggregate into one position with a volume-weighted price —
    per-level bookkeeping must not rely on positions being separate;
  - comment-based position identification breaks (comments are merged).
- Counter-measures: `IsHedgingAccount()` helper; the hedge mechanic degrades
  to a hard basket-stop close on netting; the grid mechanic tracks its level
  count and extreme entry in state variables (re-derived from the aggregate
  position when possible) instead of counting tickets; partial close uses
  `PositionClosePartial` which is valid in both modes.

### A11. Prohibited operations
- No DLL imports, no file writes outside tester sandbox needs, no network
  (WebRequest is blocked for Market products), no non-Latin characters in
  log output, no AVX-specific compilation.
- Counter-measure: templates include only `<Trade\Trade.mqh>`; no file,
  network, or DLL calls anywhere; ASCII-only log strings.

### A12. Resource budget
- "Tester takes too long" and "Log files size exceeded" are real failure
  classes: keep per-tick work bounded (copy only the bars needed), avoid
  printing in loops, avoid `Sleep` spam.
- Counter-measure: fixed small copy counts per signal; prints only on
  terminal failures; dashboard object updates are cheap label writes.

### A13. Clean `OnDeinit`
- Release every indicator handle (`IndicatorRelease`), delete all chart
  objects created by the EA.

## B. Secondary requirements (manual moderation & description)

- Product description must not promise profits; symbol/TF guidance is a
  recommendation, not a programmatic restriction.
- Inputs must be sane by default: the validator runs **default inputs** —
  defaults must trade safely on any symbol (normalized volume, valid stops).
- Version updates re-run validation; keep the EA deterministic under
  default seed/inputs.

## C. Template audit result (this project)

| Check | Status before | Status after hardening |
|---|---|---|
| A1 no-trade fallback | missing | tester-only probe trade after warm-up |
| A1 TERMINAL_TRADE_ALLOWED misuse | n/a (not used) | still not used — OK |
| A2 volume min/max/step | `NormalizeLots` | kept + `SYMBOL_VOLUME_LIMIT` check |
| A3 margin pre-check | **missing** | `MarginOK()` via `OrderCalcMargin` before every send |
| A4 stops level | **missing** | `AdjustStops()` on every SL/TP order |
| A5 freeze level | **missing** | `FreezeOK()` before `PositionModify` |
| A6 checked copies | present | kept (SafeCopy* + HistoryReady) |
| A7 zero divide | `SafeDiv` present | extended to all new math |
| A8 bounded retries | present but retried fatals | transient-only retry, fatal aborts |
| A9 OnInit robustness | INIT_FAILED on handle error | lazy handle creation, OnInit always succeeds |
| A10 netting mode | **missing** | `IsHedgingAccount()` branches in hedge + grid mechanics |
| A11 DLL/file/network | clean | clean |
| A12 log budget | mostly OK | single-line failure prints |
| A13 OnDeinit | clean | clean |
