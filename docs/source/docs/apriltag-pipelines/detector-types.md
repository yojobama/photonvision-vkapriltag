# AprilTag Pipeline Types

PhotonVision offers three different AprilTag pipeline types based on different implementations of the underlying algorithm. Each one has its advantages / disadvantages, which are detailed below.

:::{note}
Note that all of these pipeline types detect AprilTag markers and are just different algorithms for doing so.
:::

## AprilTag

The AprilTag pipeline type is based on the [AprilTag](https://april.eecs.umich.edu/software/apriltag.html) library from the University of Michigan and we recommend it for most use cases. It is (to our understanding) most accurate pipeline type, but is also ~2x slower than ArUco. This was the pipeline type used by teams in the 2023 season and is well tested.

## ArUco

The ArUco pipeline is based on the [ArUco](https://docs.opencv.org/4.8.0/d9/d6a/group__aruco.html) library implementation from OpenCV. It is ~2x higher fps and ~2x lower latency than the AprilTag pipeline type, but is less accurate. We recommend this pipeline type for teams that need to run at a higher framerate or have a lower powered device. This pipeline type was new for the 2024 season.

## AprilTag (Vulkan) - Beta

:::{warning}
This pipeline type is **beta** and may be removed in a future release if it can't be kept up to date. It has no guaranteed support window.
:::

The Vulkan AprilTag pipeline runs the same [AprilTag](https://april.eecs.umich.edu/software/apriltag.html) decoding algorithm as the standard AprilTag pipeline, but does the GPU-parallelizable stages of detection (decimation, thresholding, connected-component labeling, and boundary extraction) as Vulkan compute shaders instead of on the CPU, via the [vkapriltag](https://github.com/yojobama/vkapriltag) library. This targets low-powered ARM single-board computers with a capable GPU (e.g. the Orange Pi 5's Mali G610) where the CPU AprilTag pipeline is the bottleneck and a CUDA-based accelerator isn't an option.

Requirements and known limitations:

- Requires a Vulkan 1.1+ capable GPU and driver. Only `linuxarm64` and `linuxx86-64` builds include the native detector at all.
- **Falls back to the CPU AprilTag detector automatically** if Vulkan isn't available, the requested camera resolution isn't a multiple of 8 pixels in both dimensions, or the native detector otherwise fails to initialize. There is no separate indication in the UI today when this fallback is active beyond the resulting frame rate.
- This pipeline type has no `Decimate`, `Blur`, or `Refine Edges` controls, unlike the standard AprilTag pipeline: the Vulkan detector always runs a fixed 2x decimation, has no Gaussian pre-blur stage, and does not perform edge refinement. These aren't configurable because they aren't implemented, not because they're hidden.
- A device picker lets you choose a specific GPU when more than one Vulkan-capable device is present; "Automatic" defers to the library's own scoring (discrete GPU > integrated GPU > virtual GPU), which is a reasonable default on almost every coprocessor.
