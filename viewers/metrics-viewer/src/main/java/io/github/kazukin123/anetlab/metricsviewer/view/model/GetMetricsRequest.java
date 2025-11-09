package io.github.kazukin123.anetlab.metricsviewer.view.model;

import java.util.ArrayList;
import java.util.List;

import lombok.Data;

@Data
public class GetMetricsRequest {
    private List<String> runIds = new ArrayList<String>();
    private List<String> tagKeys = new ArrayList<String>();
}
