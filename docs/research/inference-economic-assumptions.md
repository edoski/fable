# Inference economic assumptions

Research frozen on 2026-08-01. The lean primary study needs one economic input: an electricity
price in EUR/kWh. Native-token prices are not inputs to inference cost. They are needed only if a
separate analysis converts KAIROS's native-unit fee savings into euros and compares them with the
inference electricity cost.

## Recommended primary contract

Use **0.2966 EUR/kWh** as a static Italian household electricity-price proxy. This is Eurostat's
latest complete observation for Italy at the research date: 2025 semester 2, household consumption
band DC (2,500–4,999 kWh/year), all taxes and levies included. The exact filtered
[Eurostat API response](https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/nrg_pc_204?lang=en&freq=S&siec=E7000&nrg_cons=KWH2500-4999&unit=KWH&tax=I_TAX&currency=EUR&geo=IT&time=2025-S2)
returns `0.2966`. Eurostat describes the data as consumption-weighted prices actually paid by end
users, reported in EUR/kWh, and identifies DC as the 2,500–4,999 kWh household band
([dataset methodology](https://ec.europa.eu/eurostat/cache/metadata/EN/nrg_pc_204_sims_me.htm)).

For measured incremental energy `e` in joules per cascade, calculate:

```text
cost_EUR_per_cascade = e / 3,600,000 * 0.2966
cost_EUR_per_million = cost_EUR_per_cascade * 1,000,000
```

Equivalently, each measured joule per cascade contributes `8.2389e-8 EUR/cascade`, or
`0.0823889 EUR` per million cascades. Use the exact `0.2966` input and do not round intermediate
calculations. Present the final monetary estimates and their linearly transformed confidence
limits to three significant figures.

This is an **all-in national average**, not the MacBook owner's marginal tariff. Strict marginal
cost would use only the per-kWh charges on the owner's actual contract because fixed annual charges
do not change with one more inference. That value would be private, tariff-specific, and less
generalizable. Eurostat's all-in value is therefore the cleaner thesis reference, but the result
must be named an electricity-cost **proxy**, not an observed change in the electricity bill. This
qualification is especially important because `powermetrics` estimates selected SoC rails rather
than calibrated wall-plug energy.

ARERA is useful corroboration but is not the preferred primary input. Its current quarterly
reference applies only to roughly three million vulnerable customers in *Maggior Tutela*, rather
than the Italian household market as a whole
([ARERA, 2026 Q3](https://www.arera.it/comunicati-stampa/dettaglio/elettricita-maggior-tutela-46-nel-iii-trimestre-2026-per-i-clienti-vulnerabili)).
Its Q3 2026 all-in reference price is 0.3163 EUR/kWh
([ARERA calculation sheet](https://www.arera.it/fileadmin/allegati/com_stampa/26/Aggiornamento_Maggior_Tutela_III_trimestre_2026.pdf)); substituting it would increase the derived
cost by 6.6%. This is a useful sensitivity check, not a second primary assumption.

The inference-cost subsection should report joules per cascade, the energy confidence interval,
EUR per cascade, and EUR per million cascades. It may cross-reference the existing held-out
optimizer results, but it should not add energy or token-price fields to held-out evaluation
implementation, results, or tables.

## Why token prices should be omitted

The conversion above ends in euros and requires no cryptocurrency price. A token price becomes
necessary only for a different question: whether the euro value of native-token fee savings exceeds
the euro inference cost, or how much transaction gas would make the two equal. That comparison adds
a volatile valuation date and a market-data source without improving the measurement of inference
cost itself.

The primary issue should therefore omit token-price assumptions, raw fee-to-EUR conversion, and
break-even gas. Slice 3 can reduce latency and energy, apply the electricity price, and generate the
inference-cost subsection. The existing held-out evaluation remains the authority for optimizer
savings.

## Correct optional token path

If a later appendix explicitly requests cross-currency break-even, use the following current chain
identities. All three EVM gas-price fields use 18-decimal atomic units.

| Chain | Native gas token | Symbol | Atomic scale | Primary evidence |
| --- | --- | --- | ---: | --- |
| Ethereum | Ether | ETH | `10^18 wei/ETH` | Ethereum states that gas is paid in ETH and wei is `10^-18` ETH ([gas](https://ethereum.org/developers/docs/gas/), [denominations](https://ethereum.org/developers/docs/intro-to-ether/)). |
| Avalanche C-Chain | AVAX | AVAX | `10^18 wei/AVAX` | Avalanche identifies AVAX as C-Chain's gas token and documents `1 AVAX = 10^18 wei` ([native token](https://build.avax.network/academy/avalanche-l1/l1-native-tokenomics/01-tokens-fundamentals/03-native-tokens), [unit conversion](https://build.avax.network/docs/tooling/avalanche-sdk/client/utils)). |
| Polygon PoS | Polygon Ecosystem Token | POL | `10^18 wei/POL` | Polygon identifies POL as the mainnet gas token and as MATIC's replacement ([network reference](https://docs.polygon.technology/pos/reference/rpc-endpoints), [migration](https://docs.polygon.technology/pos/concepts/tokens/matic-to-pol)); Polygon's official token list records 18 decimals ([token record](https://github.com/0xPolygon/polygon-token-list/blob/dev/src/tokens/defaultTokens.json)). |

Polygon should use **POL, not MATIC**, for a current valuation. Polygon PoS automatically converted
native MATIC to POL 1:1 on 4 September 2024
([Polygon announcement](https://polygon.technology/blog/matic-to-pol-migration-is-now-live-everything-you-need-to-know)),
and its current network reference names POL as the gas token. MATIC is relevant only to a
deliberately historical valuation of pre-migration market prices.

Kraken is a single first-party exchange source with online `ETH/EUR`, `AVAX/EUR`, and `POL/EUR`
markets, as exposed by its public
[AssetPairs endpoint](https://api.kraken.com/0/public/AssetPairs?pair=ETHEUR,AVAXEUR,POLEUR).

For a reproducible optional valuation, freeze the last completed 1,440-minute UTC candle on a named
date and use its closing price. Kraken defines close as the final traded price in a candle
([OHLCVT definition](https://support.kraken.com/articles/360047124832-downloadable-historical-ohlcvt-open-high-low-close-volume-trades-data)); its API warns that the final
returned row is the still-open candle, which must be excluded
([OHLC API](https://docs.kraken.com/api-reference/market-data/get-ohlc-data)). Preserve the exact
decimal strings returned by the API, perform calculations without intermediate rounding, and show
prices at the exchange pair precision.

For reference only, the committed 2026-07-31 UTC Kraken daily closes were:

| Pair | Close (EUR) | Source |
| --- | ---: | --- |
| ETH/EUR | `1614.35` | [Kraken OHLC](https://api.kraken.com/0/public/OHLC?pair=ETHEUR&interval=1440&since=1785456000) |
| AVAX/EUR | `5.544` | [Kraken OHLC](https://api.kraken.com/0/public/OHLC?pair=AVAXEUR&interval=1440&since=1785456000) |
| POL/EUR | `0.06182` | [Kraken OHLC](https://api.kraken.com/0/public/OHLC?pair=POLEUR&interval=1440&since=1785456000) |

These prices demonstrate a viable common source; they are not recommended inputs to the primary
inference-cost study.

## Thesis-ready wording

> Incremental model-compute energy was converted to a monetary proxy using an electricity price of
> 0.2966 EUR/kWh. This value is Eurostat's Italian household price for consumption band DC in the
> second semester of 2025, including taxes and levies, and was the latest complete national
> observation available when the assumptions were frozen on 1 August 2026. Because the energy
> estimate covers CPU, GPU, and ANE rails rather than calibrated wall-plug consumption, the result
> is interpreted as a reference electricity-cost proxy rather than an observed change in the
> household bill. Cryptocurrency prices are not required to quantify inference cost and were not
> introduced into this experiment.
