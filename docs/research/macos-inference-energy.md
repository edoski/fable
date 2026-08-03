# Measuring incremental inference energy on the M2 Max

Date: 2026-08-01

## Decision

Use Apple's `powermetrics` to estimate the incremental CPU+GPU+ANE energy of repeated,
batch-1 inference on this MacBook Pro. Measure equal-duration busy and model-loaded idle blocks,
subtract the idle estimate, then divide by the number of completed inferences.

The defensible thesis claim is **powermetrics-estimated incremental SoC energy per inference on one
MacBook Pro M2 Max**. It is not direct physical measurement, whole-machine energy, wall-plug
energy, or a result transferable to another device. An external meter would be required to make
those stronger claims.

## Instrument boundary

The inspected machine is a `Mac14,5` MacBook Pro with an Apple M2 Max, 12 CPU cores, and 96 GB of
memory, running macOS 26.5 (25F71) and Xcode 26.5 (17F42).

The local Apple `powermetrics(1)` manual and `/usr/bin/powermetrics --help` establish that:

- the tool requires superuser privileges;
- `cpu_power`, `gpu_power`, and `ane_power` are supported samplers;
- text output labels CPU, GPU, ANE, and combined CPU+GPU+ANE power in milliwatts;
- `--sample-rate` sets the interval, while plist output is machine-readable and NUL-separated;
- reported average power is estimated, may be inaccurate, and must not be compared between
  devices;
- per-process “energy impact” is only a rough, platform-specific proxy, not joules;
- short-interval battery telemetry can be aliased because controller updates may arrive out of
  phase.

Therefore, use the explicitly labelled power values, not Energy Impact or battery discharge, and
integrate them over time:

\[
E_b\,[\mathrm J] = 10^{-3}\sum_j P_{b,j}\,[\mathrm{mW}]\,\Delta t_j\,[\mathrm s].
\]

Prefer combined CPU+GPU+ANE power for the primary result and retain the three rails as diagnostics.
This combined value excludes display, storage, networking, fans, charger loss, and other board
loads. Its absolute error is not quantified by Apple.

Instruments is not a better Mac energy meter. Although `xctrace list templates` lists Power
Profiler, a local recording attempt fails with “The Power Profiler instrument is not supported on
macOS. Record on iOS or iPadOS instead.” Apple's documentation likewise limits Power Profiler to
iPhone on iOS 26 or later and iPad on iPadOS 26 or later. Its app-level values are relative power
impact, not calibrated joules. See [Apple Power
Profiler](https://developer.apple.com/documentation/xcode/measuring-your-app-s-power-use-with-power-profiler).

## Lean protocol

1. Freeze the final artifact, runtime, thread count, input IDs, macOS version, and power mode.
   Connect AC power, pause charging, disable Low Power Mode, fix the display and peripheral state,
   close unrelated applications, and begin only at nominal thermal pressure. Load the model and
   inputs before measurement; treat cold start as a separate endpoint.
2. Warm the model until latency stabilizes. Run one short preflight in both text and plist formats
   to bind the current plist fields to the text fields labelled in milliwatts. Preserve both raw
   outputs. Do not infer units for undocumented energy-counter fields.
3. Collect one continuous trace across all phases, avoiding collector start-up at every boundary:

   ```sh
   sudo /usr/bin/powermetrics \
     --samplers cpu_power,gpu_power,ane_power,thermal \
     --sample-rate 1000 --poweravg 1 \
     --format plist --output powermetrics.plist
   ```

   The command requests one-second samples and requires an interactively entered administrator
   password. The benchmark runner should write monotonic phase boundaries and inference counts to
   a separate log. Discard samples that cross a phase boundary.
4. Run 20 randomized `idle→busy` or `busy→idle` pairs. Give both blocks the same 60–120 second
   duration. In idle blocks, keep the same process, model, tensors, and runtime loaded but perform
   no inference. In busy blocks, perform only batch-1 inference. Insert a fixed washout and require
   nominal thermal pressure before the next pair.
5. For pair \(r\), compute

   \[
   e_r = \frac{E_{busy,r}-(T_{busy,r}/T_{idle,r})E_{idle,r}}{N_r},
   \]

   where \(N_r\) is the completed inference count. Keep negative pair estimates rather than
   clipping measurement noise. Report raw busy and idle power, durations, counts, throughput,
   incremental joules per inference, and thermal state.
6. Compute the mean and 95% confidence interval over the 20 pair-level \(e_r\) values. Use a
   Student-t interval when pair estimates are reasonably symmetric; otherwise bootstrap whole
   pairs. Samples inside one block and inferences inside one repeated loop are not independent
   replicates. The interval measures run-to-run repeatability, not systematic error in Apple's
   power model. See the [NIST confidence-limit
   method](https://itl.nist.gov/div898/handbook/eda/section3/eda352.htm) and [NIST bootstrap
   guidance](https://www.itl.nist.gov/div898/handbook/eda/section3/bootplot.htm).

Run a small timing-only pilot with and without `powermetrics` and report any collector overhead.
For production realism, add a separately named rolling-policy workload with its real bursts and
gaps; continuous busy blocks estimate marginal inference energy but omit wake and idle-transition
effects.

## Held-out inputs

Using every sealed held-out decision origin at true batch size 1 is scientifically fair and avoids
cherry-picking. Cycle through the complete set repeatedly inside each busy block and rotate the
starting origin between blocks so exposure remains balanced. Input preparation must remain outside
the measured phase if the estimand is model inference alone.

Downscope only if one full pass exceeds the chosen block duration or prevents enough independent
pairs. Before viewing energy results, freeze a deterministic stratified sample for every final
`(chain,K)` artifact. Preserve the testing window's chronological spread, publish the seed and
selected origin IDs, and reuse exactly that sample in every pair. Repeat the sample to fill each
busy block. The pair, not each reused input, remains the statistical unit.

## Rejected substitutes

AlDente 1.38 can hold the machine at “AC attached, charging paused,” so it is useful only as an
experimental control. Its official material describes a live Power Flow display and a smoothed
Power Consumption graph, but provides no calibration specification, accuracy bound, raw
timestamped export, or documented joule counter. The current application is proprietary. Do not
use its displayed statistics as thesis measurements. See [AlDente Power
Flow](https://apphousekitchen.com/aldente-overview/features/), [AlDente 1.35 release
notes](https://github.com/AppHouseKitchen/AlDente-Battery_Care_and_Monitoring/discussions/1604),
and the vendor's [closed-source
notice](https://github.com/AppHouseKitchen/AlDente-Battery_Care_and_Monitoring).

Third-party CLIs do not strengthen the evidence. [`asitop`](https://github.com/tlkh/asitop) parses
`powermetrics`, inheriting the same estimator and permission boundary. Tools such as
[`macmon`](https://github.com/vladkens/macmon) avoid `sudo` by using private macOS APIs to expose
similar values, but Apple provides no public calibration or stability contract for those APIs.
Use the first-party tool and preserve its raw output.

If a monetary illustration is needed, convert the partial estimate with
\(c=e/(3.6\times10^6)\times p\), where \(p\) is a dated electricity price in currency/kWh. Label
this “SoC-estimated energy cost,” not electricity consumed at the wall.

## Primary references

- Apple, `powermetrics(1)` and `/usr/bin/powermetrics --help`, inspected on macOS 26.5.
- [Measuring your app's power use with Power Profiler](https://developer.apple.com/documentation/xcode/measuring-your-app-s-power-use-with-power-profiler)
- [AlDente features: Power Flow](https://apphousekitchen.com/aldente-overview/features/)
- [AlDente 1.35 release notes](https://github.com/AppHouseKitchen/AlDente-Battery_Care_and_Monitoring/discussions/1604)
- [AlDente closed-source notice](https://github.com/AppHouseKitchen/AlDente-Battery_Care_and_Monitoring)
- [NIST confidence limits for the mean](https://itl.nist.gov/div898/handbook/eda/section3/eda352.htm)
- [NIST bootstrap plot](https://www.itl.nist.gov/div898/handbook/eda/section3/bootplot.htm)
