/*
 * Copyright (C) Photon Vision.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

package org.photonvision.vision.pipeline;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.photonvision.common.LoadJNI;
import org.photonvision.common.configuration.ConfigManager;
import org.photonvision.common.util.TestUtils;
import org.photonvision.vision.apriltag.AprilTagFamily;
import org.photonvision.vision.apriltag.VkAprilTagAvailability;
import org.photonvision.vision.calibration.CameraCalibrationCoefficients;
import org.photonvision.vision.camera.QuirkyCamera;
import org.photonvision.vision.frame.provider.FileFrameProvider;
import org.photonvision.vision.opencv.CVMat;
import org.photonvision.vision.pipe.impl.VkAprilTagDetectionPipe;
import org.photonvision.vision.pipe.impl.VkAprilTagDetectionPipe.VkAprilTagDetectionPipeParams;
import org.photonvision.vision.pipeline.result.CVPipelineResult;
import org.photonvision.vision.target.TrackedTarget;

/**
 * "Same as or better than libapriltag" parity test for the Vulkan detector - explicitly requested
 * by the maintainers (see the integration plan's §1.1, Chris Gerth's point 4) as a prerequisite
 * for review, not just good practice.
 *
 * <p>Mirrors {@link AprilTagTest}'s structure and fixtures. The comparison itself ports
 * vkapriltag's own {@code tools/validate_against_libapriltag/validate_common.h}
 * (CompareCorners/ExtractDetections/SortedIds): sort both detection sets by tag ID, walk them in
 * lockstep, and for every ID present in both compute the RMS distance over the 4 corners. That
 * tool measured ~0.85px mean RMS on real Orange Pi 5 hardware against the same
 * {@code tag1_640_480.jpg} fixture used below; the tolerances here are set well above that so
 * this test is checking for a real regression, not chasing noise.
 *
 * <p>The GPU-comparison tests below {@code assumeTrue(VkAprilTagAvailability.isSupported())} and
 * skip (not fail) when no Vulkan-capable device is present - true of every GitHub-hosted CI
 * runner today. {@link #testVulkanFallsBackToCpuOnUnsupportedFrameSize} does not skip: it drives
 * {@link VkAprilTagDetectionPipe} directly with a frame size vkapriltag can never support
 * (width not a multiple of 8), so it exercises the fallback path identically on every machine,
 * GPU or not.
 */
public class VkAprilTagTest {
    // vkapriltag's own validate_common.h::CompareCorners; both values well above the ~0.85px
    // mean RMS measured against a real reference detector on real hardware, so a regression well
    // short of "clearly broken" still trips these.
    private static final double MEAN_CORNER_RMS_TOLERANCE_PX = 3.0;
    private static final double MAX_CORNER_RMS_TOLERANCE_PX = 8.0;

    @BeforeEach
    public void setup() {
        LoadJNI.loadLibraries();
        VkAprilTagAvailability.probe();
        ConfigManager.getInstance().load();
    }

    private record DetectionInfo(int id, double[][] corners) {}

    private static List<DetectionInfo> extract(List<TrackedTarget> targets) {
        List<DetectionInfo> out = new ArrayList<>();
        for (TrackedTarget t : targets) {
            List<Point> corners = t.getTargetCorners();
            double[][] p = new double[4][2];
            for (int c = 0; c < 4 && c < corners.size(); c++) {
                p[c][0] = corners.get(c).x;
                p[c][1] = corners.get(c).y;
            }
            out.add(new DetectionInfo(t.getFiducialId(), p));
        }
        out.sort(Comparator.comparingInt(DetectionInfo::id));
        return out;
    }

    /** Ports vkapriltag's own CompareCorners - see the class Javadoc. */
    private static void assertSameDetections(
            List<TrackedTarget> vulkan, List<TrackedTarget> cpu, boolean requireAtLeastOne) {
        List<DetectionInfo> vulkanSorted = extract(vulkan);
        List<DetectionInfo> cpuSorted = extract(cpu);

        List<Integer> vulkanIds = vulkanSorted.stream().map(DetectionInfo::id).toList();
        List<Integer> cpuIds = cpuSorted.stream().map(DetectionInfo::id).toList();
        assertEquals(cpuIds, vulkanIds, "Vulkan and CPU decoded different tag ID sets");

        int compared = 0;
        double sumRms = 0.0;
        double maxRms = 0.0;
        int vi = 0, ci = 0;
        while (vi < vulkanSorted.size() && ci < cpuSorted.size()) {
            DetectionInfo v = vulkanSorted.get(vi);
            DetectionInfo c = cpuSorted.get(ci);
            if (v.id() < c.id()) {
                vi++;
                continue;
            }
            if (c.id() < v.id()) {
                ci++;
                continue;
            }
            double sqSum = 0.0;
            for (int i = 0; i < 4; i++) {
                double dx = v.corners()[i][0] - c.corners()[i][0];
                double dy = v.corners()[i][1] - c.corners()[i][1];
                sqSum += dx * dx + dy * dy;
            }
            double rms = Math.sqrt(sqSum / 4.0);
            sumRms += rms;
            maxRms = Math.max(maxRms, rms);
            compared++;
            vi++;
            ci++;
        }

        if (requireAtLeastOne) {
            assertTrue(compared > 0, "No tags were actually compared - fixture or pipeline is broken");
        }
        if (compared > 0) {
            double meanRms = sumRms / compared;
            assertTrue(
                    meanRms <= MEAN_CORNER_RMS_TOLERANCE_PX,
                    "Mean corner RMS " + meanRms + "px exceeds " + MEAN_CORNER_RMS_TOLERANCE_PX + "px over "
                            + compared + " tag(s)");
            assertTrue(
                    maxRms <= MAX_CORNER_RMS_TOLERANCE_PX,
                    "Worst-case corner RMS " + maxRms + "px exceeds " + MAX_CORNER_RMS_TOLERANCE_PX + "px");
        }
    }

    private static List<TrackedTarget> runCpu(
            TestUtils.ApriltagTestImages image,
            AprilTagFamily family,
            CameraCalibrationCoefficients calibration,
            int outputMaximumTargets) {
        try (var pipeline = new AprilTagPipeline()) {
            pipeline.getSettings().tagFamily = family;
            pipeline.getSettings().solvePNPEnabled = false;
            pipeline.getSettings().outputMaximumTargets = outputMaximumTargets;
            try (var frameProvider =
                    new FileFrameProvider(
                            TestUtils.getApriltagImagePath(image, false), TestUtils.WPI2020Image.FOV, calibration)) {
                frameProvider.requestFrameThresholdType(pipeline.getThresholdType());
                try (CVPipelineResult result = pipeline.run(frameProvider.get(), QuirkyCamera.DefaultCamera)) {
                    return List.copyOf(result.targets);
                }
            }
        }
    }

    private static List<TrackedTarget> runVulkan(
            TestUtils.ApriltagTestImages image,
            AprilTagFamily family,
            CameraCalibrationCoefficients calibration,
            int outputMaximumTargets) {
        try (var pipeline = new VkAprilTagPipeline()) {
            pipeline.getSettings().tagFamily = family;
            pipeline.getSettings().solvePNPEnabled = false;
            pipeline.getSettings().outputMaximumTargets = outputMaximumTargets;
            try (var frameProvider =
                    new FileFrameProvider(
                            TestUtils.getApriltagImagePath(image, false), TestUtils.WPI2020Image.FOV, calibration)) {
                frameProvider.requestFrameThresholdType(pipeline.getThresholdType());
                try (CVPipelineResult result = pipeline.run(frameProvider.get(), QuirkyCamera.DefaultCamera)) {
                    assertTrue(
                            pipeline.isVulkanActive(),
                            "Vulkan reported available but the pipeline silently fell back to CPU - "
                                    + "check getLastError()/logs, this test only proves something if the "
                                    + "GPU path actually ran");
                    return List.copyOf(result.targets);
                }
            }
        }
    }

    @Test
    public void testVulkanMatchesLibapriltag_singleTag36h11() {
        assumeTrue(VkAprilTagAvailability.isSupported(), "No Vulkan-capable device on this machine");

        List<TrackedTarget> cpu =
                runCpu(
                        TestUtils.ApriltagTestImages.kTag1_640_480,
                        AprilTagFamily.kTag36h11,
                        TestUtils.get2020LifeCamCoeffs(false),
                        127);
        List<TrackedTarget> vulkan =
                runVulkan(
                        TestUtils.ApriltagTestImages.kTag1_640_480,
                        AprilTagFamily.kTag36h11,
                        TestUtils.get2020LifeCamCoeffs(false),
                        127);

        assertSameDetections(vulkan, cpu, true);
    }

    @Test
    public void testVulkanMatchesLibapriltag_16h5Family() {
        assumeTrue(VkAprilTagAvailability.isSupported(), "No Vulkan-capable device on this machine");

        List<TrackedTarget> cpu =
                runCpu(
                        TestUtils.ApriltagTestImages.kTag1_16h5_1280,
                        AprilTagFamily.kTag16h5,
                        TestUtils.get2023LifeCamCoeffs(false),
                        127);
        List<TrackedTarget> vulkan =
                runVulkan(
                        TestUtils.ApriltagTestImages.kTag1_16h5_1280,
                        AprilTagFamily.kTag16h5,
                        TestUtils.get2023LifeCamCoeffs(false),
                        127);

        assertSameDetections(vulkan, cpu, true);
    }

    @Test
    public void testVulkanMatchesLibapriltag_manyTags() {
        assumeTrue(VkAprilTagAvailability.isSupported(), "No Vulkan-capable device on this machine");

        // outputMaximumTargets defaults to 127 (Byte.MAX_VALUE) and both pipelines clip to it
        // identically after detection (see AbstractAprilTagPipeline.process) - raised here so the
        // ID-set comparison below isn't just checking that both pipelines clip the same way.
        List<TrackedTarget> cpu =
                runCpu(
                        TestUtils.ApriltagTestImages.k36h11_stress_test,
                        AprilTagFamily.kTag36h11,
                        TestUtils.getCoeffs(TestUtils.LIMELIGHT_480P_CAL_FILE, false),
                        300);
        List<TrackedTarget> vulkan =
                runVulkan(
                        TestUtils.ApriltagTestImages.k36h11_stress_test,
                        AprilTagFamily.kTag36h11,
                        TestUtils.getCoeffs(TestUtils.LIMELIGHT_480P_CAL_FILE, false),
                        300);

        assertSameDetections(vulkan, cpu, true);
    }

    /**
     * Drives {@link VkAprilTagDetectionPipe} directly with a frame width that isn't a multiple of
     * 8 - vkapriltag's own hard requirement (GpuDetector.cpp), unrelated to whether a GPU is
     * present. Runs on every CI runner unconditionally: no {@code assumeTrue}, since the fallback
     * this exercises has nothing to do with Vulkan actually being available.
     */
    @Test
    public void testVulkanFallsBackToCpuOnUnsupportedFrameSize() {
        try (var pipe = new VkAprilTagDetectionPipe()) {
            pipe.setParams(
                    new VkAprilTagDetectionPipeParams(AprilTagFamily.kTag36h11, 641, 480, -1, 0));
            assertFalse(pipe.isVulkanActive(), "641 is not a multiple of 8; Vulkan must not activate");

            // The fallback CPU AprilTagDetector must still be usable - process() should not throw
            // even though the "requested" width doesn't match vkapriltag's constraints, since the
            // fallback ignores that constraint entirely.
            Mat blank = new Mat(480, 641, org.opencv.core.CvType.CV_8UC1, new org.opencv.core.Scalar(0));
            try (CVMat in = new CVMat(blank)) {
                assertTrue(pipe.run(in).output.isEmpty(), "A blank frame should decode zero tags");
            }
        }
    }
}
