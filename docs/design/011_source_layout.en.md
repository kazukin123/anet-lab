<!-- translated-from: 011_source_layout.jp.md blob:870df61c32c979f106d8ca29930f751a2960af97 date:2026-11-20 progress:done -->
# Source Layout and Library Boundaries

## Libraries and Applications

| Location | Responsibilities and dependencies |
|---|---|
| `src/core/base/` | Contracts and shared implementations for Agent, Env, NN, ReplayBuffer, and Observer. Its direct dependencies are external libraries |
| `src/algo/{dqn,muzero,imagecls}/` | Agent implementations by Agent family. They use the public headers of the base library and expose an initialization function for each family |
| `src/env/` | Implementations and initialization functions for each environment. They use the public headers of the base library |
| `src/apps/runner/` | Build inputs for Runner. Runner calls the base library, each Agent family, and the public initialization functions of the Envs |
| `src/apps/metrics-viewer/` | Metrics Viewer sources and Maven build inputs |

The contracts of the public headers placed in `include/anet/` are listed in the [extension point table](010_framework_overview.en.md#43-extension-points). Each library's `src/` holds its implementation and internal headers.

## Tests

Unit tests are placed in the library they test. When a test uses another library, it goes through that library's public headers and test-only interface.
Headers for the test-only interface live in `src/core/base/include/anet/test/`, and its implementation lives in `src/core/base/src_testing/`.
`anet-core-testing` defines `ANET_TEST_SUPPORT` for the tests that link it, and provides small Network assembly, reference computations, and stand-in Agents.

A test that uses multiple libraries is placed in the library that is its subject, with the other parties composed of stand-ins. When the combination itself is the subject, a dedicated test application is placed under `src/apps/`. Shipped configuration and validation resources are readable from the repository root.

Check target and include dependencies with the [PRD182 validation resources](../../src/core/base/testdata/rfd0182/README.md#%E3%83%A9%E3%82%A4%E3%83%96%E3%83%A9%E3%83%AA%E3%81%AE%E4%BE%9D%E5%AD%98).

Check data migration and placement with the [PRD182 migration validation](../../src/core/base/testdata/rfd0182/README.md#%E5%AE%9F%E6%A9%9F%E3%81%AE%E3%83%87%E3%83%BC%E3%82%BF%E3%82%92%E7%A7%BB%E3%81%99%E5%89%8D%E3%81%AE%E7%A2%BA%E8%AA%8D).

## Output Locations

CMake and Maven intermediate files are placed in `out/build/`. Unit test executables and shared DLLs are written to `out/test/<configuration>/`.
Runner and GUI tests are written to `apps/runner/bin/<configuration>/`, and the Metrics Viewer jar is written to `apps/metrics-viewer/bin/`.
