package io.github.kazukin123.anetlab.metricsviewer.dto;

import java.util.List;

import lombok.Data;

@Data
public class GetMetricsRequest {
    private List<String> runIds;
    private List<String> tagKeys;
}
