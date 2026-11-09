{
  description = "ML-driven PVT variation aware leakage estimation, PTM 22nm LP";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs =
    { nixpkgs, ... }:
    let
      system = "x86_64-linux";
      # torch-bin ships the CUDA runtime, which is unfree.
      pkgs = import nixpkgs {
        inherit system;
        config.allowUnfree = true;
        # The CUDA 13.0 NCCL that torch-bin links is not in any binary cache, so it compiles locally.
        # Building it only for our GPUs (RTX 3050 Ti 8.6, RTX 4070 8.9) instead of every architecture
        # cuts that compile. Our code does not call NCCL, torch-bin only needs it to load.
        config.cudaCapabilities = [ "8.6" "8.9" ];
      };

      pythonPackages =
        ps: with ps; [
          numpy
          pandas
          scikit-learn
          xgboost
          lightgbm
          matplotlib
          joblib
        ];

      tools = [
        pkgs.ngspice
        pkgs.typst
      ];

      # torch-bin 2.13 is the cu130 wheel, and nixpkgs marks it broken against the default
      # cudaPackages (12.9). Build it and cuda-bindings against CUDA 13.0 instead. The GPU machine's
      # driver (595) supports CUDA up to 13.2.
      # libnvshmem is also built locally, and nixpkgs turns on its test suite and examples, which
      # take most of that build. torch-bin only links the library, so both are off here.
      cudaPackages = pkgs.cudaPackages_13_0.overrideScope (
        cudaFinal: cudaPrevious: {
          libnvshmem = cudaPrevious.libnvshmem.overrideAttrs (previousAttrs: {
            cmakeFlags = previousAttrs.cmakeFlags ++ [
              "-DNVSHMEM_BUILD_TESTS=OFF"
              "-DNVSHMEM_BUILD_EXAMPLES=OFF"
            ];
          });
        }
      );
      gpuPython = pkgs.python3.override {
        packageOverrides = pythonFinal: pythonPrevious: {
          cuda-bindings = pythonPrevious.cuda-bindings.override { inherit cudaPackages; };
          torch-bin = pythonPrevious.torch-bin.override { inherit cudaPackages; };
        };
      };
    in
    {
      devShells.${system} = {
        # SPICE, data handling, the CPU models and the report. Enough for everything except the
        # GPU training run.
        default = pkgs.mkShell {
          packages = tools ++ [ (pkgs.python3.withPackages pythonPackages) ];
        };

        # The default shell plus CUDA PyTorch, for training on the GPU machine.
        gpu = pkgs.mkShell {
          packages = tools ++ [
            (gpuPython.withPackages (ps: pythonPackages ps ++ [ ps.torch-bin ]))
          ];
          # NixOS puts the driver's libcuda.so here, outside the Nix store.
          shellHook = ''
            export LD_LIBRARY_PATH=/run/opengl-driver/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
          '';
        };
      };
    };
}
