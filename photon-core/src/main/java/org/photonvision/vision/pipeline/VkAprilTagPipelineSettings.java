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

import org.photonvision.vision.target.TargetModel;

/**
 * BETA. Settings for the Vulkan-accelerated (vkapriltag) AprilTag detector.
 *
 * <p>Deliberately does NOT have decimate/blur/refineEdges fields, unlike {@link
 * AprilTagPipelineSettings}: the Vulkan pipeline runs a fixed 2x decimation with no pre-blur stage
 * (vkapriltag's GpuDetector hardcodes both), and edge refinement is explicitly out of scope for
 * that library (see vkapriltag's TagDecoder.h). Rather than exposing controls that silently do
 * nothing, this settings class simply doesn't have them - see the integration plan's blocker
 * write-up for why a toggle inside AprilTagPipelineSettings would have had to grey these out
 * instead.
 */
public class VkAprilTagPipelineSettings extends AprilTagPipelineSettingsBase {
    /**
     * -1 selects vkapriltag's own scored auto-select (discrete &gt; integrated &gt; virtual &gt;
     * CPU, the last of which is never actually chosen - see VkAprilTagAvailability). Otherwise an
     * index from {@code VkAprilTagAvailability.getDevices()}.
     */
    public int vulkanDeviceIndex = -1;

    /**
     * Degree of parallelism for the CPU tail (per-blob quad fitting). 0 selects {@code
     * std::thread::hardware_concurrency()}.
     */
    public int cpuThreads = 0;

    public VkAprilTagPipelineSettings() {
        super();
        pipelineType = PipelineType.AprilTagVulkan;
        targetModel = TargetModel.kAprilTag6p5in_36h11;
        cameraExposureRaw = 20;
        cameraAutoExposure = false;
        ledMode = false;
    }

    @Override
    public int hashCode() {
        final int prime = 31;
        int result = super.hashCode();
        result = prime * result + vulkanDeviceIndex;
        result = prime * result + cpuThreads;
        return result;
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (!super.equals(obj)) return false;
        if (getClass() != obj.getClass()) return false;
        VkAprilTagPipelineSettings other = (VkAprilTagPipelineSettings) obj;
        if (vulkanDeviceIndex != other.vulkanDeviceIndex) return false;
        if (cpuThreads != other.cpuThreads) return false;
        return true;
    }
}
