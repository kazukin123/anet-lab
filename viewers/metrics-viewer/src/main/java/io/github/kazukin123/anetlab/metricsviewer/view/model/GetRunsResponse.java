package io.github.kazukin123.anetlab.metricsviewer.view.model;

import java.util.List;

import lombok.Data;

@Data
public class GetRunsResponse {
    private List<RunInfo> runs;
}
