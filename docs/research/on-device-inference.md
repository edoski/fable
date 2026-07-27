# FABLE on-device inference decision and acceptance plan

Date: 2026-07-26

## Decision

FABLE mobile inference is a self-contained application path:

```text
public EVM JSON-RPC
        |
        v
closed-block context -> TypeScript feature transform -> bundled XNNPACK model
                                                       |
                                                       v
                                      recommendation and fee prediction
                                                       |
                                                       v
                                      local history and RPC outcomes
```

RPC supplies raw blockchain observations. Feature preparation, neural inference, decoding,
history, outcome resolution, and analytics run in the Expo app. There is no Python inference
server, remote model download, or fallback path.

This keeps the existing cross-platform React Native interface while removing the project-specific
service. A native Swift/Core ML rewrite would replace working application code and introduce a
second model interface without reducing the thesis scope.

## Implemented repository contract

The canonical implemented exporter and app runtime contract is owned by
[Mobile deployment](../FABLE.md#mobile-deployment). This research note records the decision
rationale and the real-artifact evidence still required.

## Current boundary

The code and non-asset tests implement the target architecture. Real-artifact acceptance has not
run.

The twelve final artifact UUIDs do not exist, so the repository intentionally has:

- no real `MOBILE.yaml`;
- no generated `manifest.json`;
- no placeholder `.pte` files.

Therefore no claim is made that a final FABLE model exports, bundles, loads, or matches the Python
model in the simulator.

## Developer flow

The current asset-generation and custom native build instructions are owned by the
[README mobile demo](../../README.md#mobile-demo).

## Deferred real-artifact acceptance

Once the exporter produces the trusted manifest and all twelve real `.pte` files, execute every
`(chain,K)` cell against fixed parity inputs in a custom native iOS simulator build. Compare both
outputs, the selected action, and decoded fee with the Python oracle within the exporter's existing
tolerances.

Until these steps pass, documentation must describe the mobile path as implemented code under an
unfulfilled generated-asset prerequisite, not as a validated runtime result.

## Primary references

- [Expo SDK 55](https://docs.expo.dev/versions/v55.0.0/)
- [Expo Continuous Native Generation](https://docs.expo.dev/workflow/continuous-native-generation/)
- [React Native ExecuTorch compatibility](https://docs.swmansion.com/react-native-executorch/docs/other/compatibility)
- [ExecuTorch model loading](https://docs.swmansion.com/react-native-executorch/docs/fundamentals/loading-models)
- [ExecuTorch 1.2 export](https://docs.pytorch.org/executorch/1.2/using-executorch-export.html)
- [Viem Public Client](https://viem.sh/docs/clients/public)
- [Viem fee history](https://viem.sh/docs/actions/public/getFeeHistory)
- [Ethereum execution API](https://ethereum.github.io/execution-apis/api/methods/eth_getBlockByNumber/)
