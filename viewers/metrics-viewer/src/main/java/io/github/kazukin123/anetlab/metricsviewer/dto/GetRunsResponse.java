package io.github.kazukin123.anetlab.metricsviewer.dto;

import java.util.List;

import io.github.kazukin123.anetlab.metricsviewer.model.Run;
import lombok.Data;

@Data
public class GetRunsResponse {
    private List<Run> runs;
}
