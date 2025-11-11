package io.github.kazukin123.anetlab.metricsviewer.view.model;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.annotation.JsonProperty;

import io.github.kazukin123.anetlab.metricsviewer.util.MetricTraceEncoder;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;

@Data
@Builder
@AllArgsConstructor
public class TagTrace {
    private final String runId;
    private final String tagKey;
    private final String type;
    private final TagStats stats;

    @JsonIgnore
    private final int[] steps;
    @JsonIgnore
    private final float[] values;
    
    @JsonProperty("encodedSteps")
    public String getEncodedSteps() {
        return MetricTraceEncoder.encodeIntArray(steps);
    }

    @JsonProperty("encodedValues")
    public String getEncodedValues() {
        return MetricTraceEncoder.encodeFloatArray(values);
    }
}
