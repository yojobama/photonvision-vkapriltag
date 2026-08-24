<script setup lang="ts">
import { PipelineType, type VkAprilTagPipelineSettings, AprilTagFamily } from "@/types/PipelineTypes";
import PvSelect from "@/components/common/pv-select.vue";
import PvSlider from "@/components/common/pv-slider.vue";
import { computed } from "vue";
import { useStateStore } from "@/stores/StateStore";
import { useCameraSettingsStore } from "@/stores/settings/CameraSettingsStore";
import { useSettingsStore } from "@/stores/settings/GeneralSettingsStore";
import { useDisplay } from "vuetify";

// TODO fix pipeline typing in order to fix this, the store settings call should be able to infer that only valid pipeline type settings are exposed based on pre-checks for the entire config section
// Defer reference to store access method
const currentPipelineSettings = computed<VkAprilTagPipelineSettings>(
  () => useCameraSettingsStore().currentPipelineSettings as VkAprilTagPipelineSettings
);
const { mdAndDown } = useDisplay();
const interactiveCols = computed(() =>
  mdAndDown.value && (!useStateStore().sidebarFolded || useCameraSettingsStore().isDriverMode) ? 8 : 7
);

// "Automatic" (-1) first, then every device the backend enumerated - device.description already
// comes from vkapriltag's own Context::DescribeDevice(), so the formatting stays owned by that
// library, not duplicated here.
const vulkanDeviceItems = computed(() => [
  { value: -1, name: "Automatic" },
  ...useSettingsStore().general.vulkanDevices.map((device) => ({
    value: device.index,
    name: device.description
  }))
]);
</script>

<template>
  <div v-if="currentPipelineSettings.pipelineType === PipelineType.AprilTagVulkan">
    <v-alert
      density="compact"
      variant="tonal"
      type="warning"
      class="mb-3"
      text="BETA: this backend runs at a fixed 2x decimation with no blur or edge-refinement tuning,
        and falls back to the CPU detector automatically if Vulkan isn't usable on this device."
    />
    <pv-select
      v-model="currentPipelineSettings.tagFamily"
      label="Target family"
      :items="[
        { value: AprilTagFamily.Family36h11, name: 'AprilTag 36h11 (6.5in)' },
        { value: AprilTagFamily.Family16h5, name: 'AprilTag 16h5 (6in)' }
      ]"
      :select-cols="interactiveCols"
      @update:modelValue="(value) => useCameraSettingsStore().changeCurrentPipelineSetting({ tagFamily: value }, false)"
    />
    <pv-select
      v-model="currentPipelineSettings.vulkanDeviceIndex"
      label="Vulkan Device"
      tooltip="Which GPU (or vkapriltag's own automatic scoring) to run detection on"
      :items="vulkanDeviceItems"
      :select-cols="interactiveCols"
      @update:modelValue="
        (value) => useCameraSettingsStore().changeCurrentPipelineSetting({ vulkanDeviceIndex: value }, false)
      "
    />
    <pv-slider
      v-model="currentPipelineSettings.cpuThreads"
      :slider-cols="interactiveCols"
      label="CPU Threads"
      tooltip="Threads for the CPU tail (per-blob quad fitting); 0 uses all available cores"
      :min="0"
      :max="8"
      @update:modelValue="(value) => useCameraSettingsStore().changeCurrentPipelineSetting({ cpuThreads: value }, false)"
    />
    <pv-slider
      v-model="currentPipelineSettings.decisionMargin"
      :slider-cols="interactiveCols"
      label="Decision Margin Cutoff"
      tooltip="Tags with a 'margin' (decoding quality score) less than this wil be rejected. Increase this to reduce the number of false positive detections"
      :min="0"
      :max="250"
      @update:modelValue="
        (value) => useCameraSettingsStore().changeCurrentPipelineSetting({ decisionMargin: value }, false)
      "
    />
    <pv-slider
      v-model="currentPipelineSettings.numIterations"
      :slider-cols="interactiveCols"
      label="Pose Estimation Iterations"
      tooltip="Number of iterations the pose estimation algorithm will run, 50-100 is a good starting point"
      :min="0"
      :max="500"
      @update:modelValue="
        (value) => useCameraSettingsStore().changeCurrentPipelineSetting({ numIterations: value }, false)
      "
    />
  </div>
</template>
